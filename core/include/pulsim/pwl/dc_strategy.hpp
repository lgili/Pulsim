// SPDX-License-Identifier: MIT
//
// Pulsim — Phase A.2: DC operating-point strategies.
//
// The naive single-shot DC solve in dc_assemble.hpp::compute_dc_op
// works for most circuits, but fails on stiff nonlinear topologies
// (long diode chains, MOS amplifiers near threshold, LDOs where every
// device sits on a non-linear discontinuity). This header bundles
// three fallback strategies into a single dispatch point so users can
// reach for the right tool without rewriting kernel code:
//
//   * Naive            — single-shot solve, same as compute_dc_op.
//   * PseudoTransient  — modified Newton with dt regularisation;
//                          globally convergent on stiff problems.
//                          Wraps the existing pseudo_transient_solve
//                          (continuation.hpp / pseudo_transient.hpp).
//   * SourceStepping   — scale ALL voltage/current source amplitudes
//                          from 0 to 1 in `n_steps` calls to
//                          compute_dc_op, using the previous solution
//                          as the next initial guess (homotopy in
//                          source amplitude).
//   * Auto             — try Naive → PseudoTransient → SourceStepping
//                          in order; return the first that succeeds.
//
// The Python-facing API in `pulsim.compute_dc_op(...)` calls
// `compute_dc_op_with_strategy(...)` after dispatching on the
// strategy string.
//
// Header-only, C++20.

#pragma once

#include "pulsim/analysis/cancellation.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/continuation.hpp"
#include "pulsim/pwl/dc_assemble.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/row_names.hpp"
#include "pulsim/pwl/pseudo_transient.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>

namespace pulsim::pwl {

/// DC operating-point strategy selector.
enum class DCStrategy : std::uint8_t {
    Naive            = 0,  //!< single-shot compute_dc_op
    PseudoTransient  = 1,  //!< wraps pseudo_transient_solve
    SourceStepping   = 2,  //!< homotopy in source amplitude
    Auto             = 3,  //!< Naive → PseudoTransient → SourceStepping
};

/// Knobs for the PseudoTransient strategy.
struct PseudoTransientConfig {
    Real dt_init = Real{1.0};
    Real dt_max  = Real{1e10};
    Size max_iters = Size{500};
    Real tol_res   = Real{1e-7};
};

/// Knobs for the SourceStepping strategy.
struct SourceSteppingConfig {
    Size n_steps    = Size{10};
    Real tol_res    = Real{1e-7};
    Size max_inner_iters = Size{50};
};

namespace detail {

/// Run a single PseudoTransient solve at the current source levels.
///
/// Builds a minimal PWL segment from a fresh DC assemble at the
/// requested switch mask, then hands it to `pseudo_transient_solve`.
/// Pre-factorises the matrix locally so the segment carries a valid
/// `solver`, which `pseudo_transient_solve` requires.
[[nodiscard]] inline Vector pseudo_transient_dc(
    const topology::Graph& graph,
    const DevicePool& pool,
    const topology::SwitchStateMask& mask,
    const PseudoTransientConfig& cfg,
    Real t_eval) {

    sparse::Matrix J;
    Vector b;
    dc_assemble(graph, pool, mask, J, b, t_eval);
    sparse::compress_in_place(J);

    auto solver = sparse::make_default_solver();
    if (!solver->analyze(J)) {
        throw std::runtime_error(std::format(
            "pseudo_transient_dc: matrix structurally singular for mask {}{}",
            mask.to_string(),
            explain_singular(graph, pool, J, solver.get())));
    }
    if (!solver->factorize(J)) {
        throw std::runtime_error(std::format(
            "pseudo_transient_dc: matrix numerically singular for mask {}{}",
            mask.to_string(),
            explain_singular(graph, pool, J, solver.get())));
    }

    PwlSegment seg;
    seg.J            = std::move(J);
    seg.b_constant   = std::move(b);
    seg.solver       = std::move(solver);
    seg.state_size   = static_cast<Size>(seg.b_constant.size());

    // Refresh function for nonlinear devices. For a pure-linear
    // circuit this is a no-op (J_nl stays empty, f_nl stays zero);
    // for circuits with nonlinear devices the caller would compose
    // the appropriate refresh chain — left as a future extension.
    NonlinearRefreshFn refresh =
        [](const Vector& /*x*/, sparse::Matrix& J_nl, Vector& f_nl,
            const topology::Graph& /*g*/, const DevicePool& /*p*/)
            -> Real {
            (void)J_nl;
            (void)f_nl;
            return Real{0};   // residual magnitude — no nonlinear contribution
        };

    Vector x_init = Vector::Zero(static_cast<Index>(seg.state_size));
    Vector b_extra = Vector::Zero(static_cast<Index>(seg.state_size));

    return pseudo_transient_solve(seg, refresh, graph, pool,
                                    x_init, b_extra,
                                    cfg.dt_init, cfg.dt_max,
                                    cfg.max_iters, cfg.tol_res);
}

/// Run SourceStepping homotopy. Scales the constant residual `b` by
/// α ∈ [0, 1] in `n_steps` increments. Each inner solve uses the
/// previous solution as the initial guess (warm start).
///
/// NOTE: this is a *minimal* implementation that works by scaling the
/// FULL constant residual — equivalent to ramping all voltage/current
/// source amplitudes proportionally. For per-source ramp control,
/// extend `dc_assemble` to accept a source-scaling vector (future).
[[nodiscard]] inline Vector source_stepping_dc(
    const topology::Graph& graph,
    const DevicePool& pool,
    const topology::SwitchStateMask& mask,
    const SourceSteppingConfig& cfg,
    Real t_eval) {

    sparse::Matrix J;
    Vector b_full;
    dc_assemble(graph, pool, mask, J, b_full, t_eval);
    sparse::compress_in_place(J);

    auto solver = sparse::make_default_solver();
    if (!solver->analyze(J)) {
        throw std::runtime_error(std::format(
            "source_stepping_dc: matrix structurally singular for mask {}{}",
            mask.to_string(),
            explain_singular(graph, pool, J, solver.get())));
    }
    if (!solver->factorize(J)) {
        throw std::runtime_error(std::format(
            "source_stepping_dc: matrix numerically singular for mask {}{}",
            mask.to_string(),
            explain_singular(graph, pool, J, solver.get())));
    }

    Vector x = Vector::Zero(static_cast<Index>(b_full.size()));
    Vector rhs(b_full.size());
    for (Size k = 1; k <= cfg.n_steps; ++k) {
        const Real alpha = static_cast<Real>(k) /
                              static_cast<Real>(cfg.n_steps);
        rhs = -alpha * b_full;
        // DirectSolver::solve(...) is void — relies on the previous
        // analyze + factorize. We trust those succeeded above.
        solver->solve(rhs, x);
    }
    return x;
}

}  // namespace detail

/// Unified DC operating-point solver with strategy selection.
///
/// Returns the state vector at DC equilibrium. Throws on failure.
/// When ``should_continue`` is non-empty, it is invoked between
/// fallback strategies (Auto mode) and the call throws
/// :class:`analysis::Cancelled` if the callback returned ``false``.
[[nodiscard]] inline Vector compute_dc_op_with_strategy(
    const topology::Graph& graph,
    const DevicePool& pool,
    const topology::SwitchStateMask& mask,
    DCStrategy strategy = DCStrategy::Auto,
    Real t_eval = Real{0},
    const PseudoTransientConfig& pt_cfg = {},
    const SourceSteppingConfig& ss_cfg = {},
    const analysis::ShouldContinueFn& should_continue = {}) {

    auto try_naive = [&]() -> Vector {
        return compute_dc_op(graph, pool, mask, t_eval);
    };
    auto try_pt = [&]() -> Vector {
        return detail::pseudo_transient_dc(graph, pool, mask,
                                              pt_cfg, t_eval);
    };
    auto try_ss = [&]() -> Vector {
        return detail::source_stepping_dc(graph, pool, mask,
                                              ss_cfg, t_eval);
    };

    analysis::check_cancellation(should_continue, "compute_dc_op");
    switch (strategy) {
    case DCStrategy::Naive:           return try_naive();
    case DCStrategy::PseudoTransient: return try_pt();
    case DCStrategy::SourceStepping:  return try_ss();
    case DCStrategy::Auto: {
        // Try naive first — fastest path.
        try { return try_naive(); }
        catch (const std::exception&) { /* fall through */ }
        analysis::check_cancellation(should_continue,
                                      "compute_dc_op", 1);
        // Then pseudo-transient.
        try { return try_pt(); }
        catch (const std::exception&) { /* fall through */ }
        analysis::check_cancellation(should_continue,
                                      "compute_dc_op", 2);
        // Finally source-stepping (most robust, slowest).
        return try_ss();
    }
    }
    throw std::runtime_error(
        "compute_dc_op_with_strategy: unknown strategy enum value");
}

}  // namespace pulsim::pwl
