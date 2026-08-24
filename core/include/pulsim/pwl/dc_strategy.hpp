// SPDX-License-Identifier: MIT
//
// Pulsim — DC operating-point strategies.
//
// The naive single-shot DC solve in dc_assemble.hpp::compute_dc_op
// works for most circuits and fails on the stiff nonlinear ones —
// long diode chains, MOS amplifiers near threshold, LDOs where every
// device sits on a discontinuity. This header is the cascade of
// fallbacks, tried in order until one produces an answer:
//
//   1. Naive          — single-shot solve (Newton, if the caller
//                        supplied a refresh). Fastest; the common
//                        case never leaves this rung.
//   2. GminStepping   — ramp a large conductance to ground down by
//                        decades, warm-starting each solve. Fixes
//                        BOTH a badly-pivoted matrix and a Newton
//                        that cannot find the basin from x = 0.
//   3. SourceStepping — ramp every independent source amplitude from
//                        0 to nominal. Fixes a Newton basin problem;
//                        cannot fix a singular matrix, because the
//                        matrix is the same at every α.
//   4. PseudoTransient— integrate dx/dt = -F to equilibrium. Last,
//                        not second: pseudo_transient.hpp documents
//                        its own limitation — Pulsim's MNA systems
//                        have mixed-sign eigenvalues on the
//                        voltage-source constraint rows, where the
//                        artificial dynamics is UNSTABLE. It earns
//                        its place only on the constraint-free
//                        resistive-nonlinear problems.
//
// v2.0 Phase 2 (B.2) rebuilt rungs 2-4. Before it, in Auto mode:
//
//   * `pseudo_transient_dc` pre-factorised the raw matrix and threw
//     if it was singular — i.e. it rejected exactly the inputs the
//     naive rung had just failed on, so it could never rescue one;
//   * `source_stepping_dc` factorised once and re-solved the SAME
//     system with a scaled right-hand side, so it returned the naive
//     answer after n_steps redundant solves, or threw identically;
//   * both passed a no-op NonlinearRefreshFn, so reaching either one
//     directly on a circuit with diodes or MOSFETs returned the
//     operating point of the circuit with those devices OPEN — with
//     no warning.
//
// Header-only, C++20.

#pragma once

#include "pulsim/analysis/cancellation.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/continuation.hpp"
#include "pulsim/pwl/dc_assemble.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/gmin.hpp"
#include "pulsim/pwl/row_names.hpp"
#include "pulsim/pwl/pseudo_transient.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulsim::pwl {

/// DC operating-point strategy selector.
enum class DCStrategy : std::uint8_t {
    Naive            = 0,  //!< single-shot compute_dc_op
    PseudoTransient  = 1,  //!< wraps pseudo_transient_solve
    SourceStepping   = 2,  //!< homotopy in source amplitude
    Auto             = 3,  //!< the cascade, in the order above
    GminStepping     = 4,  //!< homotopy in conductance-to-ground
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

/// Which rung of the cascade produced the answer, and what it cost.
struct DCSolveReport {
    DCStrategy strategy = DCStrategy::Naive;
    Size rungs_attempted = Size{0};   //!< homotopy rungs, incl. retries
    Real final_gmin = Real{0};        //!< conductance left in the answer
    Real residual = Real{0};          //!< ||f(x)||_inf, UN-augmented
    std::string detail;               //!< human-readable trace

    [[nodiscard]] std::string summary() const {
        const char* name = "naive";
        switch (strategy) {
        case DCStrategy::Naive:           name = "naive"; break;
        case DCStrategy::GminStepping:    name = "gmin stepping"; break;
        case DCStrategy::SourceStepping:  name = "source stepping"; break;
        case DCStrategy::PseudoTransient: name = "pseudo-transient"; break;
        case DCStrategy::Auto:            name = "auto"; break;
        }
        return std::format(
            "DC operating point solved by {} ({} rung(s), "
            "residual {:.3e})", name, rungs_attempted, residual);
    }
};

namespace detail {

/// Residual of the ORIGINAL (un-augmented, full-amplitude) system at
/// `x`, and the row where it is worst.
struct DCResidual {
    Real norm = Real{0};
    Index worst_row = kInvalidIndex;
};

[[nodiscard]] inline DCResidual dc_residual(
    const topology::Graph& graph,
    const DevicePool& pool,
    const topology::SwitchStateMask& mask,
    const NonlinearRefreshFn& refresh,
    const Vector& x,
    Real t_eval) {
    sparse::Matrix J;
    Vector b;
    dc_assemble(graph, pool, mask, J, b, t_eval, Real{0}, Real{1});
    sparse::compress_in_place(J);
    Vector f = J * x + b;
    if (refresh) {
        const Index n = static_cast<Index>(f.size());
        sparse::Matrix J_nl(n, n);
        Vector f_nl = Vector::Zero(n);
        (void)refresh(x, J_nl, f_nl, graph, pool);
        f += f_nl;
    }
    if (f.size() == 0) {
        return {};
    }
    Eigen::Index at = 0;
    const Real norm = f.cwiseAbs().maxCoeff(&at);
    return {norm, static_cast<Index>(at)};
}

/// One DC solve of the system assembled at (`gmin`, `source_scale`),
/// warm-started from `x_warm`. Newton when a refresh is supplied,
/// a single factor-and-solve when it is not.
[[nodiscard]] inline Vector dc_solve_at(
    const topology::Graph& graph,
    const DevicePool& pool,
    const topology::SwitchStateMask& mask,
    const NonlinearRefreshFn& refresh,
    const Vector& x_warm,
    Real t_eval,
    Real gmin,
    Real source_scale,
    Size max_newton_iters,
    Real tol_dx,
    Real tol_res,
    bool enable_line_search = false,
    bool enable_lm = false) {

    PwlSegment seg;
    dc_assemble(graph, pool, mask, seg.J, seg.b_constant, t_eval,
                 gmin, source_scale);
    sparse::compress_in_place(seg.J);
    seg.state_size = static_cast<Size>(seg.b_constant.size());

    if (!refresh) {
        auto solver = sparse::make_default_solver();
        if (!solver->analyze(seg.J)) {
            throw std::runtime_error(std::format(
                "dc_solve_at: matrix structurally singular at "
                "gmin = {:.3e}, source scale = {:.3f}{}",
                gmin, source_scale,
                explain_singular(graph, pool, seg.J, solver.get())));
        }
        if (!solver->factorize(seg.J)) {
            throw std::runtime_error(std::format(
                "dc_solve_at: matrix numerically singular at "
                "gmin = {:.3e}, source scale = {:.3f}{}",
                gmin, source_scale,
                explain_singular(graph, pool, seg.J, solver.get())));
        }
        Vector out;
        Vector rhs = -seg.b_constant;
        solver->solve(rhs, out);
        return out;
    }

    const Vector b_extra =
        Vector::Zero(static_cast<Index>(seg.state_size));
    return solve_with_newton_b_extra(
        seg, refresh, graph, pool, x_warm, b_extra,
        max_newton_iters, tol_dx, tol_res,
        enable_line_search, enable_lm);
}

/// Reject an operating point that only satisfies the AUGMENTED
/// system. Whatever homotopy produced `x`, the answer handed back
/// must solve the circuit the user actually described.
///
/// What this CAN catch: a homotopy that returned at α < 1, or at a
/// conductance rung above the floor, or a rung that reported success
/// on a system it had modified. What it CANNOT catch is a
/// load-bearing floor: augmented and un-augmented differ by exactly
/// gmin·v on the node rows, so at gmin = 1e-12 the gate would only
/// trip above 1e6 V. Detecting a floor that is holding the answer up
/// is a STRUCTURAL question, and `dc_structural_defect` is what
/// answers it — before any conductance is stamped.
inline void require_unaugmented_residual(
    const topology::Graph& graph,
    const DevicePool& pool,
    const topology::SwitchStateMask& mask,
    const NonlinearRefreshFn& refresh,
    const Vector& x,
    Real t_eval,
    Real tol,
    const char* who,
    DCSolveReport* report) {
    const auto r = dc_residual(graph, pool, mask, refresh, x, t_eval);
    if (report != nullptr) {
        report->residual = r.norm;
    }
    if (!(r.norm <= tol)) {
        throw std::runtime_error(std::format(
            "{}: converged on the regularized system but the answer "
            "does not satisfy the original circuit (||f||_inf = "
            "{:.3e} > {:.3e}, worst at {})",
            who, r.norm, tol,
            row_equation_label(graph, pool, r.worst_row)));
    }
}

}  // namespace detail

// =============================================================================
// Rung 2 — gmin stepping
// =============================================================================

/// Conductance homotopy: solve with a large conductance from every
/// node to ground, then walk it down by decades, warm-starting each
/// solve from the last, until only the floor remains.
///
/// WHAT IT FIXES. At the top of the ramp every node is clamped to
/// ground through 10 mS, so the Jacobian is diagonally dominant and
/// Newton converges from anywhere. Each decade down relaxes the
/// clamp by 10×, and the previous answer is a good enough guess to
/// stay in the basin. This is the standard SPICE recovery, and for a
/// nonlinear circuit it is the difference between an answer and
/// "Newton did not converge".
///
/// WHAT IT DOES NOT FIX. Structural singularity. A node with no
/// equation is a topology defect, not a conditioning one, and the
/// probe below refuses to paper over it — `preflight.hpp` is the
/// pass that repairs those, with a report.
///
/// For a circuit with no nonlinear devices (`refresh` empty) the
/// ramp is pointless — a direct solve has no basin to miss — so it
/// collapses to a single solve at the floor, which is still a
/// genuine rescue when the naive matrix pivoted badly.
[[nodiscard]] inline Vector compute_dc_op_gmin_stepped(
    const topology::Graph& graph,
    const DevicePool& pool,
    const topology::SwitchStateMask& mask,
    const NonlinearRefreshFn& refresh = {},
    Real t_eval = Real{0},
    const GminConfig& cfg = {},
    DCSolveReport* report = nullptr) {

    const Index n = static_cast<Index>(pool.state_size(graph));

    // Structural probe on the UN-augmented matrix, before any
    // conductance goes in. gmin gives every node row a diagonal; if
    // the row was empty on its own merits the solve would succeed
    // and report a fabricated 0 V for a node that has no defined
    // voltage at all.
    {
        const auto defect =
            dc_structural_defect(graph, pool, mask, refresh, t_eval);
        if (defect.present) {
            throw std::runtime_error(std::format(
                "compute_dc_op_gmin_stepped: DC system structurally "
                "singular for mask {} — gmin cannot substitute for a "
                "missing equation{}",
                mask.to_string(), defect.detail));
        }
    }

    const Real target = std::max(cfg.floor, Real{0});
    const std::vector<Real> ramp =
        refresh ? gmin_ramp(cfg) : std::vector<Real>{target};

    // Seed rung 1 from the LINEAR solve at the same conductance,
    // not from x = 0. Newton on a smooth diode started at zero bias
    // is the exact failure `initial_guess.hpp` exists to work around
    // — the exponential's Jacobian there bears no relation to the
    // solution — and a homotopy whose first rung cannot converge has
    // nothing to step down from. One extra linear solve buys the
    // whole ramp.
    Vector x = Vector::Zero(n);
    if (refresh) {
        try {
            x = detail::dc_solve_at(graph, pool, mask, {}, x,
                                     t_eval, ramp.front(), Real{1},
                                     Size{1}, cfg.tol_dx,
                                     cfg.tol_res);
        } catch (const std::exception&) {
            x = Vector::Zero(n);   // fall back to a cold start
        }
    }
    Real g = ramp.front();
    Real g_ok = std::numeric_limits<Real>::infinity();
    bool solved_any = false;
    Size next_idx = 1;
    Size attempts = 0;
    // Retries refine the ramp; the budget bounds how far that can go
    // before we admit the problem is not a ramp-resolution problem.
    const Size max_attempts = 4 * ramp.size() + 8;

    while (true) {
        if (++attempts > max_attempts) {
            throw std::runtime_error(std::format(
                "compute_dc_op_gmin_stepped: exhausted the rung "
                "budget ({}) at gmin = {:.3e} (last solved {:.3e}, "
                "target {:.3e}) — refining the ramp is no longer "
                "buying progress, so the obstacle is not the ramp",
                max_attempts, g, g_ok, target));
        }
        try {
            x = detail::dc_solve_at(graph, pool, mask, refresh, x,
                                     t_eval, g, Real{1},
                                     cfg.max_newton_iters,
                                     cfg.tol_dx, cfg.tol_res,
                                     cfg.enable_line_search,
                                     cfg.enable_lm);
        } catch (const std::exception& e) {
            if (!solved_any) {
                throw std::runtime_error(std::format(
                    "compute_dc_op_gmin_stepped: the first rung "
                    "(gmin = {:.3e}) already failed, so there is "
                    "nothing to step down from: {}",
                    g, e.what()));
            }
            // Halve the decade we just tried to cross.
            const Real mid = std::sqrt(g * g_ok);
            if (!(mid < g_ok) || !(mid > g)) {
                throw std::runtime_error(std::format(
                    "compute_dc_op_gmin_stepped: cannot get below "
                    "gmin = {:.3e} (last solved {:.3e}); the ramp is "
                    "already at floating-point resolution: {}",
                    g, g_ok, e.what()));
            }
            g = mid;
            continue;
        }
        solved_any = true;
        g_ok = g;
        if (g <= target) {
            break;
        }
        while (next_idx < ramp.size() && ramp[next_idx] >= g) {
            ++next_idx;
        }
        // Back to the scheduled decade rather than the width the
        // last bisection settled on — see the note in
        // `source_stepping_dc`: carrying a reduction forward without
        // ever letting it grow back converts one hard rung into a
        // step so small the budget runs out first.
        g = (next_idx < ramp.size()) ? ramp[next_idx] : target;
    }

    if (report != nullptr) {
        report->strategy = DCStrategy::GminStepping;
        report->rungs_attempted = attempts;
        report->final_gmin = g;
    }
    detail::require_unaugmented_residual(
        graph, pool, mask, refresh, x, t_eval,
        cfg.max_unaugmented_residual,
        "compute_dc_op_gmin_stepped", report);

    return x;
}

namespace detail {

/// Rung 4 — pseudo-transient continuation at the requested mask.
///
/// Deliberately does NOT pre-factorise the raw matrix: the whole
/// point of PTC is that `J + (1/dt)·I` is solvable when `J` alone is
/// not, so gating on `J` would reject exactly the inputs this rung
/// exists to rescue.
[[nodiscard]] inline Vector pseudo_transient_dc(
    const topology::Graph& graph,
    const DevicePool& pool,
    const topology::SwitchStateMask& mask,
    const NonlinearRefreshFn& refresh,
    const PseudoTransientConfig& cfg,
    Real t_eval) {

    PwlSegment seg;
    dc_assemble(graph, pool, mask, seg.J, seg.b_constant, t_eval);
    sparse::compress_in_place(seg.J);
    seg.state_size = static_cast<Size>(seg.b_constant.size());

    const Index n = static_cast<Index>(seg.state_size);
    const Vector x_init = Vector::Zero(n);
    const Vector b_extra = Vector::Zero(n);

    // A NULL refresh here would silently solve the circuit with every
    // nonlinear device open. Substitute the explicit no-op only when
    // the caller genuinely has no nonlinear devices to stamp.
    NonlinearRefreshFn eff = refresh;
    if (!eff) {
        eff = [](const Vector&, sparse::Matrix&, Vector&,
                  const topology::Graph&, const DevicePool&) -> Real {
            return Real{0};
        };
    }

    return pseudo_transient_solve(seg, eff, graph, pool,
                                    x_init, b_extra,
                                    cfg.dt_init, cfg.dt_max,
                                    cfg.max_iters, cfg.tol_res);
}

/// Rung 3 — source-amplitude homotopy.
///
/// Re-assembles at every α, so the nonlinear devices are re-stamped
/// at the new operating point and each solve warm-starts from the
/// last. (Scaling only the right-hand side of a fixed factorization,
/// as this did before v2.0 Phase 2, is a no-op: the matrix is the
/// same at every α, so the α = 1 solve *is* the naive answer.)
///
/// Note what this rung can and cannot do: ramping amplitudes changes
/// the excitation, never the matrix's structure, so it rescues a
/// Newton basin problem and never a singular system.
[[nodiscard]] inline Vector source_stepping_dc(
    const topology::Graph& graph,
    const DevicePool& pool,
    const topology::SwitchStateMask& mask,
    const NonlinearRefreshFn& refresh,
    const SourceSteppingConfig& cfg,
    Real t_eval,
    DCSolveReport* report) {

    const Index n = static_cast<Index>(pool.state_size(graph));
    const Size n_steps = cfg.n_steps > 0 ? cfg.n_steps : Size{1};

    Vector x = Vector::Zero(n);
    Real alpha_ok = Real{0};
    Real alpha = Real{1} / static_cast<Real>(n_steps);
    Size attempts = 0;
    const Size max_attempts = 4 * n_steps + 8;

    while (true) {
        if (++attempts > max_attempts) {
            throw std::runtime_error(std::format(
                "source_stepping_dc: exhausted the rung budget ({}) "
                "at {:.6f} of nominal source amplitude (last solved "
                "{:.6f}) — the circuit will not follow the ramp even "
                "in ever smaller steps",
                max_attempts, alpha, alpha_ok));
        }
        try {
            x = dc_solve_at(graph, pool, mask, refresh, x, t_eval,
                             Real{0}, alpha, cfg.max_inner_iters,
                             Real{1e-9}, cfg.tol_res);
        } catch (const std::exception& e) {
            const Real mid = Real{0.5} * (alpha + alpha_ok);
            if (!(mid > alpha_ok) || !(mid < alpha)) {
                throw std::runtime_error(std::format(
                    "source_stepping_dc: cannot get past {:.6f} of "
                    "nominal source amplitude (last solved {:.6f}): "
                    "{}", alpha, alpha_ok, e.what()));
            }
            alpha = mid;
            continue;
        }
        alpha_ok = alpha;
        if (alpha >= Real{1}) {
            break;
        }
        // Deliberately back to the FULL nominal increment, not the
        // one the last bisection settled on. A shrink-only step
        // controller turns one hard spot into a permanently tiny
        // step and exhausts the budget before reaching α = 1; the
        // re-bisection this costs is bounded by that same budget,
        // which is the cheaper of the two failure modes. (Review
        // finding HOMOTOPY-STEP-NOT-STICKY, declined on that
        // ground — a shrink-and-grow controller would be correct
        // but is more machinery than the saving justifies.)
        alpha = std::min(Real{1},
                          alpha + Real{1} / static_cast<Real>(n_steps));
    }

    if (report != nullptr) {
        report->strategy = DCStrategy::SourceStepping;
        report->rungs_attempted = attempts;
        // Explicitly zero, not left alone: a stale value inherited
        // from the gmin rung that ran before this one would claim a
        // conductance this answer does not carry.
        report->final_gmin = Real{0};
    }
    require_unaugmented_residual(graph, pool, mask, refresh, x,
                                  t_eval, cfg.tol_res,
                                  "source_stepping_dc", report);
    return x;
}

}  // namespace detail

/// Unified DC operating-point solver with strategy selection.
///
/// `refresh` is the nonlinear stamping chain. Leaving it empty means
/// "this circuit has no nonlinear devices"; supplying it wrongly (or
/// not at all, on a circuit that has diodes) yields the operating
/// point of a DIFFERENT circuit, which is why every rung now takes it.
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
    const analysis::ShouldContinueFn& should_continue = {},
    const NonlinearRefreshFn& refresh = {},
    const GminConfig& gmin_cfg = {},
    DCSolveReport* report = nullptr) {

    auto stamp_report = [&](DCStrategy s, Size rungs, Real gmin) {
        if (report != nullptr) {
            report->strategy = s;
            report->rungs_attempted = rungs;
            report->final_gmin = gmin;
        }
    };

    auto try_naive = [&]() -> Vector {
        Vector x = refresh
            ? compute_dc_op_newton(graph, pool, mask, refresh, t_eval,
                                    Size{50}, Real{1e-9}, Real{1e-9},
                                    /*line_search=*/false,
                                    /*lm=*/false, gmin_cfg.floor)
            : compute_dc_op(graph, pool, mask, t_eval,
                             gmin_cfg.floor);
        stamp_report(DCStrategy::Naive, Size{1}, gmin_cfg.floor);
        if (report != nullptr) {
            report->residual = detail::dc_residual(
                graph, pool, mask, refresh, x, t_eval).norm;
        }
        return x;
    };
    auto try_gmin = [&]() -> Vector {
        return compute_dc_op_gmin_stepped(graph, pool, mask, refresh,
                                            t_eval, gmin_cfg, report);
    };
    auto try_ss = [&]() -> Vector {
        return detail::source_stepping_dc(graph, pool, mask, refresh,
                                            ss_cfg, t_eval, report);
    };
    auto try_pt = [&]() -> Vector {
        Vector x = detail::pseudo_transient_dc(graph, pool, mask,
                                                 refresh, pt_cfg,
                                                 t_eval);
        stamp_report(DCStrategy::PseudoTransient, Size{1},
                      Real{0});
        detail::require_unaugmented_residual(
            graph, pool, mask, refresh, x, t_eval,
            pt_cfg.tol_res, "pseudo_transient_dc", report);
        return x;
    };

    // A structurally singular system is a TOPOLOGY defect, and no
    // rung may paper over one. This guard exists because
    // pseudo-transient will otherwise "succeed" on a floating node:
    // its (1/dt)·I regularization is a conductance on every row, so
    // it invents a value for an unknown that has no equation, and
    // the residual check cannot object because an all-zero row is
    // satisfied by anything. Probe once, up front, for every rung.
    {
        const auto defect =
            dc_structural_defect(graph, pool, mask, refresh, t_eval);
        if (defect.present) {
            throw std::runtime_error(std::format(
                "compute_dc_op: DC system structurally singular for "
                "mask {} — this is a topology defect, not a "
                "convergence one, so no fallback strategy will fix "
                "it{}",
                mask.to_string(), defect.detail));
        }
    }

    analysis::check_cancellation(should_continue, "compute_dc_op");
    switch (strategy) {
    case DCStrategy::Naive:           return try_naive();
    case DCStrategy::GminStepping:    return try_gmin();
    case DCStrategy::PseudoTransient: return try_pt();
    case DCStrategy::SourceStepping:  return try_ss();
    case DCStrategy::Auto: {
        std::string trace;
        auto note = [&](const char* who, const std::exception& e) {
            trace += std::format("\n  * {} failed: {}", who, e.what());
        };
        try { return try_naive(); }
        catch (const analysis::Cancelled&) { throw; }
        catch (const std::exception& e) { note("naive", e); }

        analysis::check_cancellation(should_continue,
                                      "compute_dc_op", 1);
        try { return try_gmin(); }
        catch (const analysis::Cancelled&) { throw; }
        catch (const std::exception& e) { note("gmin stepping", e); }

        analysis::check_cancellation(should_continue,
                                      "compute_dc_op", 2);
        try { return try_ss(); }
        catch (const analysis::Cancelled&) { throw; }
        catch (const std::exception& e) { note("source stepping", e); }

        analysis::check_cancellation(should_continue,
                                      "compute_dc_op", 3);
        try { return try_pt(); }
        catch (const analysis::Cancelled&) { throw; }
        catch (const std::exception& e) { note("pseudo-transient", e); }

        throw std::runtime_error(std::format(
            "compute_dc_op[auto]: every DC strategy failed.{}\n"
            "  If this circuit's steady state is a switching average "
            "rather than a fixed point, no DC solve can find it — "
            "run a transient instead (Python: "
            "compute_dc_op(..., strategy=\"settle\")).",
            trace));
    }
    }
    throw std::runtime_error(
        "compute_dc_op_with_strategy: unknown strategy enum value");
}

}  // namespace pulsim::pwl
