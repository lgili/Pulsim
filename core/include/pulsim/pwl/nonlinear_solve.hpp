#pragma once

// =============================================================================
// Pulsim v2 — Layer 4 V3: Nonlinear Newton on cached linear factor
// =============================================================================
//
// `pulsim-v2-nonlinear-segment-newton` Phase 1.
//
// The Newton iteration on top of the LINEAR cached matrix.
// Layer 4 V0-V2 cached J_lin and b_lin (and pre-factored).
// Nonlinear devices add g(x) that depends on x. The combined
// system is:
//
//   f(x) = J_lin · x + b_lin + g(x) = 0
//   J(x) = J_lin + ∂g/∂x
//
// Newton:
//   J(x_k) · dx = -f(x_k)
//   x_{k+1} = x_k + dx
//
// At each iteration we re-factor J_lin + J_nl(x_k). The factor
// cost is the same as v1's per-step refactor, BUT we still
// amortize the assembly: J_lin is built ONCE per switch state,
// and many simulation steps re-use that base. Only the small
// nonlinear delta changes per iteration.

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/segment.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/sparse/solver.hpp"
#include "pulsim/topology/graph.hpp"

#include <functional>
#include <stdexcept>
#include <string>

namespace pulsim::pwl {

/// Nonlinear stamping callback. Given the current Newton iterate
/// `x`, fills `J_nl` and `f_nl` with the nonlinear contributions
/// and returns the residual norm (`max(|i_nl|)` for diagnostic
/// reporting).
///
/// Implementations should:
///   1. Zero J_nl and f_nl up front.
///   2. Walk the graph for `BranchKind::Nonlinear` branches.
///   3. Stamp the standard 2-terminal pattern using Layer 2's
///      `evaluate_current_and_jacobian<T>`.
using NonlinearRefreshFn = std::function<
    Real(const Vector& x,
         sparse::Matrix& J_nl,
         Vector& f_nl,
         const topology::Graph& graph,
         const DevicePool& pool)>;

/// Newton-iterated solve on top of a cached PwlSegment, with an
/// explicit `b_extra` vector. This overload composes with the
/// trap-companion history contributions from Layer 4 V1: at
/// step n+1 the caller passes `b_extra = history_compute(...)`,
/// and the residual is `J_lin·x + (b_constant + b_extra) + g(x)`.
///
/// For linear-only circuits (`refresh` returns 0 and stamps
/// nothing), the loop exits after 1 iteration with a result
/// identical to `cache.solve(mask, b_extra, x)`.
///
/// Throws `std::runtime_error` on non-convergence.
[[nodiscard]] inline Vector solve_with_newton_b_extra(
    const PwlSegment& seg,
    const NonlinearRefreshFn& refresh,
    const topology::Graph& graph,
    const DevicePool& pool,
    const Vector& x_init,
    const Vector& b_extra,
    Size max_iters = Size{50},
    Real tol_dx  = Real{1e-9},
    Real tol_res = Real{1e-9},
    bool enable_line_search = false,
    bool enable_lm = false) {
    const Size n = seg.state_size;
    Vector x = x_init;
    if (static_cast<Size>(x.size()) != n) {
        x = Vector::Zero(static_cast<Index>(n));
    }

    sparse::Matrix J_nl(static_cast<Index>(n),
                         static_cast<Index>(n));
    Vector f_nl = Vector::Zero(static_cast<Index>(n));
    Vector dx;

    // Levenberg-Marquardt damping state.
    Real lm_lambda = Real{1e-6};
    constexpr Real lm_min    = Real{1e-12};
    constexpr Real lm_max    = Real{1e8};
    constexpr Real lm_shrink = Real{0.5};
    constexpr Real lm_grow   = Real{10};
    constexpr Size lm_max_attempts = Size{30};

    Real last_dx_norm  = std::numeric_limits<Real>::infinity();
    Real last_res_norm = std::numeric_limits<Real>::infinity();

    for (Size iter = 0; iter < max_iters; ++iter) {
        // 1. Refresh nonlinear contributions at current x.
        const Real nl_residual_norm =
            refresh(x, J_nl, f_nl, graph, pool);
        sparse::compress_in_place(J_nl);

        // 2. Build combined J = J_lin + J_nl. We can't add an
        //    Eigen::SparseMatrix to a const reference cheaply
        //    without making a copy, so we copy J_lin once per
        //    iteration. Future perf optimisation: avoid the copy
        //    via pattern-sharing.
        sparse::Matrix J_combined = seg.J;
        sparse::compress_in_place(J_combined);
        if (J_nl.nonZeros() > 0) {
            J_combined += J_nl;
        }
        sparse::compress_in_place(J_combined);

        // 3. Build combined f(x) = J_lin·x + (b_constant +
        //    b_extra) + f_nl.
        Vector f_combined =
            seg.J * x + seg.b_constant + b_extra + f_nl;

        (void)nl_residual_norm;   // diagnostic only
        const Real baseline_norm =
            f_combined.lpNorm<Eigen::Infinity>();

        // 4. Solve for `dx`. The strategy depends on which
        //    globalization is active:
        //      * Plain Newton (default): solve J · dx = -f once,
        //        accept α = 1.
        //      * Line search: solve once with α = 1, backtrack
        //        α if residual worsens.
        //      * LM: build J_lm = J + λ·I, solve, accept-and-
        //        shrink-λ if residual drops, else grow λ and
        //        retry.
        Real alpha = Real{1};
        if (enable_lm) {
            // LM inner loop. Acceptance uses the L2 norm of f
            // (the natural LM objective ||f||₂²) since the
            // infinity norm can plateau when the worst residual
            // row is stable.
            const Real baseline_l2 = f_combined.norm();
            // Early-exit: if the baseline residual is already
            // at convergence, return immediately. Otherwise LM
            // would try to "improve" a near-zero residual and
            // fail by definition.
            if (baseline_norm < tol_res) {
                last_dx_norm  = Real{0};
                last_res_norm = baseline_norm;
                return x;
            }
            sparse::Matrix J_nl_trial(static_cast<Index>(n),
                                       static_cast<Index>(n));
            Vector f_nl_trial = Vector::Zero(static_cast<Index>(n));
            bool accepted = false;
            for (Size attempt = 0;
                 attempt < lm_max_attempts;
                 ++attempt) {
                // Build J_lm = J_combined + λ·I by copying and
                // bumping the diagonal.
                sparse::Matrix J_lm = J_combined;
                for (Index i = 0; i < J_lm.rows(); ++i) {
                    J_lm.coeffRef(i, i) += lm_lambda;
                }
                sparse::compress_in_place(J_lm);
                auto lm_solver = sparse::make_default_solver();
                if (!lm_solver->analyze(J_lm) ||
                    !lm_solver->factorize(J_lm)) {
                    // Singular at this λ — grow and retry.
                    lm_lambda *= lm_grow;
                    if (lm_lambda > lm_max) {
                        throw std::runtime_error(
                            "solve_with_newton (LM): factor "
                            "failed at λ = " +
                            std::to_string(lm_lambda));
                    }
                    continue;
                }
                Vector lm_neg_f = -f_combined;
                lm_solver->solve(lm_neg_f, dx);
                const Vector x_trial = x + dx;
                (void)refresh(x_trial, J_nl_trial, f_nl_trial,
                               graph, pool);
                const Vector f_trial =
                    seg.J * x_trial + seg.b_constant +
                    b_extra + f_nl_trial;
                // Accept when the trial residual is no worse
                // than the baseline (with a tiny tolerance for
                // FP noise — near-converged residuals can
                // differ in the lowest bits between iterations).
                if (f_trial.norm() <= baseline_l2 * Real{1.0001}) {
                    // Accept; shrink λ.
                    accepted = true;
                    lm_lambda = std::max(lm_lambda * lm_shrink,
                                          lm_min);
                    break;
                }
                // Reject; grow λ.
                lm_lambda *= lm_grow;
                if (lm_lambda > lm_max) {
                    throw std::runtime_error(
                        "solve_with_newton (LM): λ exceeded "
                        "limit at iter " +
                        std::to_string(iter));
                }
            }
            if (!accepted) {
                throw std::runtime_error(
                    "solve_with_newton (LM): "
                    "no improving step at iter " +
                    std::to_string(iter));
            }
        } else {
            // Plain Newton + optional line search.
            auto solver = sparse::make_default_solver();
            if (!solver->analyze(J_combined)) {
                throw std::runtime_error(
                    "solve_with_newton: combined matrix is "
                    "structurally singular at iter " +
                    std::to_string(iter));
            }
            if (!solver->factorize(J_combined)) {
                throw std::runtime_error(
                    "solve_with_newton: combined matrix is "
                    "numerically singular at iter " +
                    std::to_string(iter));
            }
            Vector neg_f = -f_combined;
            solver->solve(neg_f, dx);

            if (enable_line_search) {
                sparse::Matrix J_nl_trial(static_cast<Index>(n),
                                           static_cast<Index>(n));
                Vector f_nl_trial =
                    Vector::Zero(static_cast<Index>(n));
                bool accepted = false;
                for (Size bt = 0; bt < Size{8}; ++bt) {
                    const Vector x_trial = x + alpha * dx;
                    (void)refresh(x_trial, J_nl_trial, f_nl_trial,
                                   graph, pool);
                    const Vector f_trial =
                        seg.J * x_trial + seg.b_constant +
                        b_extra + f_nl_trial;
                    if (f_trial.lpNorm<Eigen::Infinity>() <
                            baseline_norm) {
                        accepted = true;
                        break;
                    }
                    alpha *= Real{0.5};
                }
                if (!accepted) {
                    alpha = Real{1};
                }
            }
        }
        x += alpha * dx;

        // 6. Convergence check. We use the MNA residual norm
        //    `||f_combined||_inf` only — `nl_residual_norm`
        //    (returned by the refresh) is the magnitude of the
        //    device CURRENTS, not a residual, and at convergence
        //    the currents are typically O(mA-A), not zero.
        last_dx_norm  = (alpha * dx).lpNorm<Eigen::Infinity>();
        last_res_norm = f_combined.lpNorm<Eigen::Infinity>();
        if (last_dx_norm < tol_dx && last_res_norm < tol_res) {
            return x;
        }
    }

    throw std::runtime_error(
        "solve_with_newton: failed to converge after " +
        std::to_string(max_iters) +
        " iterations (||dx||_inf = " +
        std::to_string(last_dx_norm) +
        ", ||residual||_inf = " +
        std::to_string(last_res_norm) + ")");
}

/// Layer 4 V3 entry point — Newton without trap-companion
/// history (b_extra = 0). Delegates to the b_extra overload.
[[nodiscard]] inline Vector solve_with_newton(
    const PwlSegment& seg,
    const NonlinearRefreshFn& refresh,
    const topology::Graph& graph,
    const DevicePool& pool,
    const Vector& x_init,
    Size max_iters = Size{50},
    Real tol_dx  = Real{1e-9},
    Real tol_res = Real{1e-9}) {
    const Vector zero_b_extra =
        Vector::Zero(static_cast<Index>(seg.state_size));
    return solve_with_newton_b_extra(
        seg, refresh, graph, pool, x_init, zero_b_extra,
        max_iters, tol_dx, tol_res);
}

/// Convenience: no-op refresh function for linear-only systems.
/// Useful for testing the Newton path with a circuit that has
/// no Nonlinear-kind branches (the loop should converge in 1
/// iteration).
[[nodiscard]] inline Real noop_refresh(
    const Vector& /*x*/, sparse::Matrix& J_nl, Vector& f_nl,
    const topology::Graph& /*graph*/,
    const DevicePool& /*pool*/) {
    if (J_nl.rows() > 0) {
        J_nl.setZero();
    }
    if (f_nl.size() > 0) {
        f_nl.setZero();
    }
    return Real{0};
}

}  // namespace pulsim::pwl
