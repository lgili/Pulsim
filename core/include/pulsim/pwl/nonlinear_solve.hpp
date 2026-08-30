#pragma once

// =============================================================================
// Pulsim — Layer 4 V3: Nonlinear Newton on cached linear factor
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
#include "pulsim/pwl/row_names.hpp"
#include "pulsim/pwl/segment.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/sparse/solver.hpp"
#include "pulsim/topology/graph.hpp"

#include <format>
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
    // Argmax companions to the two norms above (v2.0 Phase 1) —
    // the MNA row where each is worst, for the failure message.
    Index worst_dx_row  = kInvalidIndex;
    Index worst_res_row = kInvalidIndex;

    // ----- Auto-LM detection state (GUI integration findings T1.2) -----
    // Two failure modes promote `enable_lm` to true mid-solve:
    //
    // 1. **Singular factorize.** The combined Jacobian is numerically
    //    rank-deficient. The user described this as the
    //    `PwlStateSpaceCache: numerically singular` symptom on a
    //    multi-stage switched topology (e.g. boost MOSFET at 65 kHz +
    //    a VSI at 20 kHz). LM with λ·I diagonal bump cures it.
    //
    // 2. **Near-miss stall.** Newton hits residual ≈ 0 (well below
    //    `tol_res`) but `||dx||` plateaus above `tol_dx`. Indicates
    //    a flat valley in the solution manifold — Newton's full step
    //    overshoots and walks the valley without progressing. LM's
    //    diagonal damping curls the step back toward the descent
    //    direction.
    //
    // Pre-fix the user had to set `enable_newton_lm=True` manually on
    // these topologies (plus often a physical RC snubber). After this
    // change, default settings recover automatically.
    constexpr Size kNearMissStreak = Size{3};
    Size near_miss_streak = Size{0};

    // Residual of the PREVIOUS iteration, for the line-search
    // auto-promotion above. Infinity on the first pass so it never
    // fires before there is something to compare against.
    Real prev_baseline_norm = std::numeric_limits<Real>::infinity();

    for (Size iter = 0; iter < max_iters; ++iter) {
        // Defense-in-depth (audit 2026-05 critic): a non-finite iterate means
        // a NaN/Inf entered the system — e.g. a bad device parameter that
        // slipped past the builder gate, or an ill-conditioned step. A NaN
        // compares false against every tolerance below, so without this guard
        // the loop would silently burn to max_iters (or grow LM's λ to the
        // limit) and report a misleading error. Fail loudly and diagnosably.
        if (!x.allFinite()) {
            throw std::runtime_error(std::format(
                "solve_with_newton: non-finite state vector at iter {} — a "
                "NaN/Inf propagated into the solution (check device "
                "parameters and matrix conditioning)",
                iter));
        }

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

        // Companion guard: catch a NaN/Inf that entered via the nonlinear
        // refresh (f_nl) or b_extra even when x itself is still finite —
        // otherwise it silently corrupts the convergence/LM/line-search tests.
        if (!f_combined.allFinite()) {
            // Name WHERE the NaN entered — on a big converter the
            // first non-finite row is usually the offending device
            // itself (v2.0 Phase 1 diagnostics parity).
            Index bad = kInvalidIndex;
            for (Index i = 0; i < static_cast<Index>(f_combined.size());
                 ++i) {
                if (!std::isfinite(f_combined[i])) { bad = i; break; }
            }
            throw std::runtime_error(std::format(
                "solve_with_newton: non-finite residual at iter {}, "
                "first at {} — the nonlinear refresh or b_extra "
                "produced a NaN/Inf (check device parameters and "
                "matrix conditioning)",
                iter, row_equation_label(graph, pool, bad)));
        }

        (void)nl_residual_norm;   // diagnostic only
        const Real baseline_norm =
            f_combined.lpNorm<Eigen::Infinity>();

        // v2.0 Phase 2 — auto-promote LINE SEARCH, the way LM is
        // auto-promoted below.
        //
        // The trigger is the one condition backtracking exists for
        // and the one neither LM trigger sees: a full Newton step
        // that made the residual WORSE. LM promotes on a singular
        // factorize or on a near-miss stall (residual already tiny,
        // ||dx|| plateaued), so a plainly DIVERGING Newton falls
        // through both and the run dies.
        //
        // Promoting instead of defaulting line search on is
        // deliberate: measured on a mains rectifier, backtracking
        // from the first iteration costs ~30 % on a run that never
        // needed it, while this comparison costs nothing. It
        // recovered 6 of 11 failing (voltage, sharpness, dt)
        // combinations, and on runs that already converged it moved
        // the answer by 1.2e-12 on a 170 V scale — line search
        // changes the PATH to the root, not the root.
        if (!enable_line_search && !enable_lm && iter > 0 &&
            std::isfinite(prev_baseline_norm) &&
            baseline_norm > prev_baseline_norm) {
            enable_line_search = true;
        }
        prev_baseline_norm = baseline_norm;

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
                        throw std::runtime_error(std::format(
                            "solve_with_newton (LM): factor "
                            "failed at λ = {}",
                            lm_lambda));
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
                    throw std::runtime_error(std::format(
                        "solve_with_newton (LM): λ exceeded "
                        "limit at iter {}",
                        iter));
                }
            }
            if (!accepted) {
                throw std::runtime_error(std::format(
                    "solve_with_newton (LM): no improving step at "
                    "iter {} — worst residuals: {}",
                    iter,
                    top_entries_by_name(graph, pool, f_combined)));
            }
        } else {
            // Plain Newton + optional line search.
            auto solver = sparse::make_default_solver();
            if (!solver->analyze(J_combined)) {
                throw std::runtime_error(std::format(
                    "solve_with_newton: combined matrix is "
                    "structurally singular at iter {}{}",
                    iter,
                    explain_singular(graph, pool, J_combined,
                                      solver.get())));
            }
            if (!solver->factorize(J_combined)) {
                // Auto-LM promotion (GUI integration findings T1.2):
                // numerically singular Jacobian usually means a
                // multi-stage switched topology hit a mask combo
                // where multiple switches commutate together and
                // a regularized solve is needed. Instead of
                // throwing, flip into LM mode and retry this
                // iteration via `continue` — the LM branch above
                // builds J_lm = J + λ·I, which restores full rank.
                enable_lm = true;
                continue;
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
        // v2.0 Phase 1 (audit finding
        // `singular-errors-dont-name-the-node`): keep the ARGMAX, not
        // just the magnitude. `lpNorm<Infinity>()` throws away the
        // one piece of information a stuck user actually needs —
        // WHERE the residual lives. maxCoeff(&idx) costs the same
        // pass and yields an original MNA row index (no permutation
        // is involved: f_combined and dx are in state-vector space),
        // which `pwl::row_label` turns into "node sw3_e".
        //
        // Eigen defines lpNorm<Infinity>() as exactly
        // `cwiseAbs().maxCoeff()` (Core/Dot.h) — with ONE extra
        // guard: it short-circuits a size-0 vector to 0 instead of
        // calling maxCoeff, which asserts on an empty input. Keep
        // that guard so this stays a strictly value-preserving
        // change for every input the old code accepted.
        Eigen::Index dx_at = 0;
        Eigen::Index res_at = 0;
        last_dx_norm  = dx.size() == 0
            ? Real{0}
            : (alpha * dx).cwiseAbs().maxCoeff(&dx_at);
        last_res_norm = f_combined.size() == 0
            ? Real{0}
            : f_combined.cwiseAbs().maxCoeff(&res_at);
        worst_dx_row  = dx.size() == 0
            ? kInvalidIndex : static_cast<Index>(dx_at);
        worst_res_row = f_combined.size() == 0
            ? kInvalidIndex : static_cast<Index>(res_at);
        if (last_dx_norm < tol_dx && last_res_norm < tol_res) {
            return x;
        }
        // SCALE-RELATIVE residual acceptance. `tol_res` is
        // ABSOLUTE, and a residual is a current: on a converter
        // with kV nodes and kA branch currents, machine epsilon
        // alone puts it near 1e-9, so an absolute 1e-9 can be
        // unreachable no matter how good the answer is. Measured:
        // an ideal-Shockley flyback with a kV leakage spike where
        // ||dx|| had converged to 6.8e-14 — the iterate was exact
        // to fourteen digits — while ||residual|| plateaued at
        // 7.5e-9, which is 6e-12 RELATIVE to the equations' own
        // scale. The fixed engine aborted that circuit even split
        // into 64 sub-steps.
        //
        // So: accept when dx has converged AND the residual is at
        // the noise floor of the system's own magnitudes. The
        // scale is the largest |entry| of the iterate and of the
        // assembled RHS terms, which is what the residual is a
        // difference OF. This can only make runs that used to abort succeed;
        // it never loosens the dx criterion, which is the one that
        // says the ANSWER stopped moving.
        if (last_dx_norm < tol_dx) {
            const Real scale = std::max({
                Real{1},
                x.size() == 0 ? Real{0} : x.cwiseAbs().maxCoeff(),
                seg.b_constant.size() == 0
                    ? Real{0}
                    : seg.b_constant.cwiseAbs().maxCoeff(),
                b_extra.size() == 0
                    ? Real{0}
                    : b_extra.cwiseAbs().maxCoeff()});
            if (last_res_norm < tol_res * scale) {
                return x;
            }
        }
        // Auto-LM promotion #2: near-miss stall (GUI T1.2).
        // Residual is essentially zero (we're at a fixed point of f)
        // but `dx` plateaus above `tol_dx` — Newton's full step
        // overshoots a flat valley in the solution manifold and
        // ping-pongs without progressing. Diagnostic the user
        // reported: `||residual||_inf ≈ 1e-13`,
        // `||dx||_inf ≈ 4e-6`. LM's diagonal damping curls the
        // step back toward the descent direction. Promote after a
        // consecutive streak so a single off-iter doesn't trigger.
        if (!enable_lm
                && last_res_norm < tol_res
                && last_dx_norm >= tol_dx) {
            ++near_miss_streak;
            if (near_miss_streak >= kNearMissStreak) {
                enable_lm = true;
            }
        } else {
            near_miss_streak = Size{0};
        }
    }

    throw std::runtime_error(std::format(
        "solve_with_newton: failed to converge after {} iterations "
        "(||dx||_inf = {:.3e} worst at {}, ||residual||_inf = {:.3e} "
        "worst at {})",
        max_iters,
        last_dx_norm, row_label(graph, pool, worst_dx_row),
        last_res_norm,
        row_equation_label(graph, pool, worst_res_row)));
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
