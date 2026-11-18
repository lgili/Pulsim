#pragma once

// =============================================================================
// Pulsim — Layer 4 V10: pseudo-transient continuation (research)
// =============================================================================
//
// Solves F(x) = 0 by integrating the artificial dynamics
//
//     D · dx/dt = -F(x),   D = 1 on every node (KCL) row, 0 elsewhere
//
// with implicit backward Euler at adaptive pseudo-timesteps. The
// modified Newton system per iteration is
//
//     (J(x) + (1/dt) D) · dx = -F(x)
//
// where dt grows automatically when ||F|| decreases. D is a 1 F
// capacitor from every node to ground: the branch rows (voltage
// sources, inductors at DC, controlled sources) stay algebraic and are
// met exactly at every iterate, and for a passive circuit the
// pseudo-dynamics is a passive RC network, which backward Euler
// integrates stably at any dt.
//
// Until 2026-11 the shift was (1/dt)·I on EVERY row. On a voltage
// source's branch row (no diagonal of its own) that term is not a
// capacitor but a negative resistance in the loop, and the iterate
// was repelled from the solution: a 10 V, 1 kΩ / 5 kΩ divider ended
// at ||F|| = 1.4e84 after 500 iterations instead of 8.3333 V.

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/row_names.hpp"
#include "pulsim/pwl/nonlinear_solve.hpp"
#include "pulsim/pwl/segment.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/sparse/solver.hpp"
#include "pulsim/topology/graph.hpp"

#include <algorithm>
#include <format>
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
    // Where the worst residual sat on the last iteration — reported
    // if we give up, so the user gets a location and not just a
    // magnitude (v2.0 Phase 1 diagnostics parity).
    Index F_worst_row = kInvalidIndex;

    for (Size iter = 0; iter < max_iters; ++iter) {
        // 1. Refresh nonlinear contributions.
        (void)refresh(x, J_nl, f_nl, graph, pool);
        sparse::compress_in_place(J_nl);

        // 2. Residual of the ORIGINAL problem.
        Vector F = seg.J * x + seg.b_constant + b_extra + f_nl;
        Eigen::Index f_at = 0;
        const Real F_norm = F.size() == 0
            ? Real{0}
            : F.cwiseAbs().maxCoeff(&f_at);
        F_worst_row = F.size() == 0 ? kInvalidIndex
                                     : static_cast<Index>(f_at);

        // 3. Convergence on original residual. tol_res is in amperes
        // on a node row, i.e. tol_res / G volts (0.1 mV through 1 kΩ),
        // so finish with one plain Newton step and keep it if it lowers
        // the residual: exact on a linear circuit.
        if (F_norm < tol_res) {
            sparse::Matrix J_full = seg.J;
            if (J_nl.nonZeros() > 0) {
                J_full += J_nl;
            }
            sparse::compress_in_place(J_full);
            auto polish = sparse::make_default_solver();
            if (polish->analyze(J_full) && polish->factorize(J_full)) {
                Vector neg_F = -F;
                polish->solve(neg_F, dx);
                const Vector x_try = x + dx;
                (void)refresh(x_try, J_nl, f_nl, graph, pool);
                const Vector F_try = seg.J * x_try + seg.b_constant +
                                     b_extra + f_nl;
                if (F_try.lpNorm<Eigen::Infinity>() <= F_norm) {
                    return x_try;
                }
            }
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
        const Index n_kcl = std::min<Index>(
            static_cast<Index>(graph.num_nodes()), J_pt.rows());
        for (Index i = 0; i < n_kcl; ++i) {
            J_pt.coeffRef(i, i) += inv_dt;
        }
        sparse::compress_in_place(J_pt);

        // 5. Solve J_pt · dx = -F.
        auto solver = sparse::make_default_solver();
        if (!solver->analyze(J_pt)) {
            throw std::runtime_error(std::format(
                "pseudo_transient_solve: J_pt is "
                "structurally singular at iter {}{}",
                iter,
                explain_singular(graph, pool, J_pt, solver.get())));
        }
        if (!solver->factorize(J_pt)) {
            // Shrink dt and retry: a larger node shunt
            // (1/dt) regularises a floating node.
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

    throw std::runtime_error(std::format(
        "pseudo_transient_solve: failed to converge after {} iterations "
        "(last ||F||_inf = {:.3e}, worst at {})",
        max_iters, F_norm_prev,
        row_equation_label(graph, pool, F_worst_row)));
}

}  // namespace pulsim::pwl
