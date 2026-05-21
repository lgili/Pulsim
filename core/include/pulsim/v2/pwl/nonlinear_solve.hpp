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

#include "pulsim/v2/numeric/dense.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/pwl/segment.hpp"
#include "pulsim/v2/sparse/matrix.hpp"
#include "pulsim/v2/sparse/solver.hpp"
#include "pulsim/v2/topology/graph.hpp"

#include <functional>
#include <stdexcept>
#include <string>

namespace pulsim::v2::pwl {

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

/// Newton-iterated solve on top of a cached PwlSegment.
///
/// For linear-only circuits (`refresh` returns 0 and stamps
/// nothing), the loop exits after 1 iteration with a result
/// identical to `cache.solve` (single triangular solve on the
/// pre-factored J_lin).
///
/// Throws `std::runtime_error` on non-convergence with the last
/// `||dx||_∞` and `||residual||_∞` in the message.
[[nodiscard]] inline Vector solve_with_newton(
    const PwlSegment& seg,
    const NonlinearRefreshFn& refresh,
    const topology::Graph& graph,
    const DevicePool& pool,
    const Vector& x_init,
    Size max_iters = Size{50},
    Real tol_dx  = Real{1e-9},
    Real tol_res = Real{1e-9}) {
    const Size n = seg.state_size;
    Vector x = x_init;
    if (static_cast<Size>(x.size()) != n) {
        x = Vector::Zero(static_cast<Index>(n));
    }

    sparse::Matrix J_nl(static_cast<Index>(n),
                         static_cast<Index>(n));
    Vector f_nl = Vector::Zero(static_cast<Index>(n));
    Vector dx;

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

        // 3. Build combined f(x) = J_lin·x + b_constant + f_nl.
        Vector f_combined =
            seg.J * x + seg.b_constant + f_nl;

        // 4. Solve J · dx = -f.
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

        // 5. Update x.
        x += dx;

        // 6. Convergence check. We use the MNA residual norm
        //    `||f_combined||_inf` only — `nl_residual_norm`
        //    (returned by the refresh) is the magnitude of the
        //    device CURRENTS, not a residual, and at convergence
        //    the currents are typically O(mA-A), not zero.
        (void)nl_residual_norm;   // diagnostic only
        last_dx_norm  = dx.lpNorm<Eigen::Infinity>();
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

}  // namespace pulsim::v2::pwl
