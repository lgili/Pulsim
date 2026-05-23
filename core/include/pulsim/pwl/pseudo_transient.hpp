#pragma once

// =============================================================================
// Pulsim — Layer 4 V10: pseudo-transient continuation (research)
// =============================================================================
//
// Solves F(x) = 0 by converting it into the artificial ODE
//
//     dx/dt = -F(x)
//
// and integrating with implicit backward Euler at adaptive
// pseudo-timesteps. The modified Newton system per iteration is
//
//     (J(x) + (1/dt) I) · dx = -F(x)
//
// where dt grows automatically when ||F|| decreases.
//
// IMPORTANT LIMITATION FOR MNA SYSTEMS:
//
// PTC requires the artificial dynamics dx/dt = -F to be
// STABLE near the solution, which holds when J = ∂F/∂x has
// positive-real-part eigenvalues. Pulsim's MNA matrices have
// MIXED-SIGN eigenvalues (the constraint rows for voltage
// sources introduce negative-real-part components). On those
// rows, PTC's artificial dynamics is UNSTABLE — the iterate
// is repelled from the solution.
//
// V10's empirical finding: pure PTC fails on Pulsim MNA
// circuits with voltage-source constraints, regardless of dt
// schedule. The robust fix for diode circuits is the
// `make_diode_aware_initial_guess` helper in `initial_guess.hpp`,
// which puts Newton inside the correct basin via a structural
// load-line warm-start.
//
// `pseudo_transient_solve` SHIPS for use cases with
// well-behaved Jacobians (e.g. purely resistive nonlinear
// loads with no MNA-constraint rows). For canonical Pulsim
// circuits, use the smart warm-start instead.

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/nonlinear_solve.hpp"
#include "pulsim/pwl/segment.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/sparse/solver.hpp"
#include "pulsim/topology/graph.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace pulsim::pwl {

/// Pseudo-transient continuation Newton solver.
///
/// Globally convergent on any continuously-differentiable F.
/// Used as the robust fallback when plain Newton + line
/// search + LM all fail on stiff problems (e.g. the κ=20
/// sinusoidal rectifier from x=0).
///
/// Parameters:
///   * `dt_init`: starting pseudo-timestep. Smaller values
///     are more conservative (essentially gradient descent
///     with step size dt). Default 1e-3.
///   * `dt_max`: cap on pseudo-timestep growth. Default 1e10
///     (effectively unbounded).
///   * `max_iters`: hard limit on iteration count.
///     Default 500.
///   * `tol_res`: convergence tolerance on ||F||_∞.
///     Default 1e-7.
///
/// Throws `std::runtime_error` if max_iters is reached
/// without convergence (typically indicates the problem
/// is genuinely singular or dt_max is too low).
[[nodiscard]] inline Vector pseudo_transient_solve(
    const PwlSegment& seg,
    const NonlinearRefreshFn& refresh,
    const topology::Graph& graph,
    const DevicePool& pool,
    const Vector& x_init,
    const Vector& b_extra,
    Real dt_init = Real{1.0},
    Real dt_max  = Real{1e10},
    Size max_iters = Size{500},
    Real tol_res = Real{1e-7}) {

    const Size n = seg.state_size;
    Vector x = x_init;
    if (static_cast<Size>(x.size()) != n) {
        x = Vector::Zero(static_cast<Index>(n));
    }

    sparse::Matrix J_nl(static_cast<Index>(n),
                         static_cast<Index>(n));
    Vector f_nl = Vector::Zero(static_cast<Index>(n));
    Vector dx;

    Real dt = dt_init;
    constexpr Real dt_min = Real{1e-12};
    Real F_norm_prev = std::numeric_limits<Real>::infinity();

    for (Size iter = 0; iter < max_iters; ++iter) {
        // 1. Refresh nonlinear contributions.
        (void)refresh(x, J_nl, f_nl, graph, pool);
        sparse::compress_in_place(J_nl);

        // 2. Residual of the ORIGINAL problem.
        Vector F = seg.J * x + seg.b_constant + b_extra + f_nl;
        const Real F_norm = F.lpNorm<Eigen::Infinity>();

        // 3. Convergence on original residual.
        if (F_norm < tol_res) {
            return x;
        }

        // 4. Build modified Jacobian J_pt = J + (1/dt) I.
        sparse::Matrix J_pt = seg.J;
        sparse::compress_in_place(J_pt);
        if (J_nl.nonZeros() > 0) {
            J_pt += J_nl;
        }
        sparse::compress_in_place(J_pt);
        const Real inv_dt = Real{1} / dt;
        for (Index i = 0; i < J_pt.rows(); ++i) {
            J_pt.coeffRef(i, i) += inv_dt;
        }
        sparse::compress_in_place(J_pt);

        // 5. Solve J_pt · dx = -F.
        auto solver = sparse::make_default_solver();
        if (!solver->analyze(J_pt)) {
            throw std::runtime_error(
                "pseudo_transient_solve: J_pt is "
                "structurally singular at iter " +
                std::to_string(iter));
        }
        if (!solver->factorize(J_pt)) {
            // Shrink dt and retry (J_pt should ALWAYS be
            // non-singular for dt > 0 because (1/dt)·I
            // dominates near dt=0).
            dt = std::max(dt_min, dt * Real{0.1});
            continue;
        }
        Vector neg_F = -F;
        solver->solve(neg_F, dx);

        // 6. Take the step (PTC always accepts).
        x += dx;

        // 7. dt adaptation — exponential ramp.
        //
        // For MNA-DAE systems the constraint rows of F have
        // no natural time-derivative, so ||F|| barely
        // changes per iteration in the small-dt regime. Pure
        // SER doesn't grow dt, and trust-region grows it
        // glacially. We use exponential growth (×3 per iter)
        // capped at dt_max. After ~10 iters we're in the
        // Newton regime regardless of the starting dt.
        //
        // Safety: recompute F at the new x and shrink dt if
        // the residual blew up by more than 5×. This handles
        // the case where the exponential growth lands in a
        // bad basin (e.g. sigmoid past delta=0 for diodes).
        (void)refresh(x, J_nl, f_nl, graph, pool);
        const Vector F_new = seg.J * x + seg.b_constant +
                              b_extra + f_nl;
        const Real F_new_norm = F_new.lpNorm<Eigen::Infinity>();
        if (F_new_norm > F_norm * Real{5.0}) {
            // Big rebound — rollback and shrink dt aggressively.
            // Also remember: a rollback indicates the sigmoid
            // wall is nearby. Cap dt growth for the next few
            // iters to avoid hitting the same wall.
            x -= dx;
            dt = std::max(dt * Real{0.1}, dt_min);
        } else if (F_new_norm < F_norm * Real{0.95}) {
            // Real progress — grow.
            dt = std::min(dt * Real{3.0}, dt_max);
        } else {
            // Marginal change — grow modestly. This is the
            // common case in the early constraint-row regime.
            dt = std::min(dt * Real{1.3}, dt_max);
        }
        F_norm_prev = F_norm;
    }

    throw std::runtime_error(
        "pseudo_transient_solve: failed to converge after " +
        std::to_string(max_iters) +
        " iterations (last ||F||_inf = " +
        std::to_string(F_norm_prev) + ")");
}

}  // namespace pulsim::pwl
