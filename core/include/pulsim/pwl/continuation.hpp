#pragma once

// =============================================================================
// Pulsim v2 — Layer 4 V8: continuation / homotopy Newton solve
// =============================================================================
//
// `pulsim-v2-continuation-newton` Phase 1.
//
// For stiff problems where Newton diverges from any reasonable warm
// start (the κ=20 sinusoidal rectifier is the motivating example),
// the classical fix is **continuation (homotopy)**: solve a
// SEQUENCE of progressively harder problems, warm-starting each
// from the previous step's converged solution.
//
//   Refresh 0: "easy" parameter (e.g. κ_easy = 2)  → Newton converges.
//   Refresh 1: closer to target                     → warm-start from x_0.
//   ...
//   Refresh N: target parameter (e.g. κ = 20)       → warm-start from x_{N-1}.
//
// Each step's solution is essentially the correct answer for the
// next step's parameter, just slightly mis-shaped — Newton converges
// easily in the local quadratic basin.
//
// This header ships a thin orchestrator that runs
// `solve_with_newton_b_extra` for each refresh in the supplied
// sequence. The CALLER constructs the sequence — typically via
// `make_kappa_override_refresh(kappa_i)` from
// `nonlinear_refresh_diode.hpp` for the smooth-blend IdealDiode.

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/nonlinear_solve.hpp"
#include "pulsim/pwl/segment.hpp"
#include "pulsim/topology/graph.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace pulsim::pwl {

/// Continuation / homotopy Newton solve.
///
/// Runs `solve_with_newton_b_extra` once per `NonlinearRefreshFn`
/// in `refresh_sequence`, warm-starting each call from the previous
/// step's converged solution. Returns the final converged `x` at
/// the end of the sequence (i.e. the solution at the TARGET
/// refresh — the LAST element of the sequence).
///
/// Semantics:
///   * `refresh_sequence` must be non-empty.
///   * `x_init` warm-starts the FIRST refresh; thereafter each
///     refresh receives the prior step's converged x.
///   * `seg`, `graph`, `pool`, `b_extra`, and the Newton tuning
///     knobs (`max_iters_per_step`, `tol_dx`, `tol_res`,
///     `enable_line_search`, `enable_lm`) are forwarded unchanged
///     to every inner solve.
///
/// Throws `std::runtime_error` if any inner Newton fails (the
/// exception message carries the index of the failing step).
///
/// Cost: N inner Newton solves vs 1 for single-shot Newton, where
/// N is the length of `refresh_sequence`. The robustness boost
/// justifies the linear blow-up for stiff problems; for easy
/// problems the caller simply doesn't enable continuation.
[[nodiscard]] inline Vector continuation_solve(
    const PwlSegment& seg,
    const std::vector<NonlinearRefreshFn>& refresh_sequence,
    const topology::Graph& graph,
    const DevicePool& pool,
    const Vector& x_init,
    const Vector& b_extra,
    Size max_iters_per_step = Size{100},
    Real tol_dx  = Real{1e-7},
    Real tol_res = Real{1e-5},
    bool enable_line_search = false,
    bool enable_lm = false) {
    if (refresh_sequence.empty()) {
        throw std::runtime_error(
            "continuation_solve: refresh_sequence is empty");
    }

    Vector x = x_init;
    for (Size step = 0; step < refresh_sequence.size(); ++step) {
        try {
            x = solve_with_newton_b_extra(
                seg,
                refresh_sequence[step],
                graph,
                pool,
                x,
                b_extra,
                max_iters_per_step,
                tol_dx,
                tol_res,
                enable_line_search,
                enable_lm);
        } catch (const std::runtime_error& e) {
            throw std::runtime_error(
                "continuation_solve: inner Newton failed at "
                "continuation step " +
                std::to_string(step) + " of " +
                std::to_string(refresh_sequence.size()) +
                " — " + e.what());
        }
    }
    return x;
}

}  // namespace pulsim::pwl
