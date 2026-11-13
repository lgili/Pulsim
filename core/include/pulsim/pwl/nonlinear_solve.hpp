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
#include "pulsim/solver/result.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/sparse/solver.hpp"
#include "pulsim/topology/graph.hpp"

#include <algorithm>
#include <cstdint>
#include <format>
#include <memory>
#include <optional>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

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

/// Per-run Newton workspace (update-newton-persistent-solver, stages A1
/// and A2).
///
/// Holds the combined Jacobian `J = J_lin + J_nl` on the union pattern
/// of the two operands, written in place through two slot arrays (the
/// per-entry sum is Eigen's: `a + b`, `a + 0`, `0 + b`), and one direct
/// solver analysed once per pattern and refactorised in place. The
/// pattern is checked after every refresh (index arrays of both
/// operands); a change rebuilds the slots and analyses again.
///
/// `backend` (stage A2): the engines' run workspaces use the in-house
/// Gilbert–Peierls LU (`Pulsim`), whose refactor allocates nothing; a
/// call without a workspace (DC paths, continuation) keeps
/// `Eigen::SparseLU`, so the DC cascade is unchanged. One workspace per
/// run: never shared between threads.
struct NewtonWorkspace {
    sparse::Backend backend = sparse::Backend::Pulsim;

    // Counters, monotonic over the workspace's life.
    std::uint64_t solves         = 0;   // solve_with_newton_b_extra calls
    std::uint64_t iterations     = 0;   // Newton iterations
    std::uint64_t analyses       = 0;   // symbolic analyses (plain path)
    std::uint64_t factorizations = 0;   // numeric factorisations (plain path)
    std::uint64_t lm_attempts    = 0;   // LM factorisation attempts
    std::vector<std::uint16_t> iterations_per_solve;   // per call

    /// Move the counters out, for `SimulationResult::newton`.
    [[nodiscard]] solver::NewtonCounters take_counters() {
        solver::NewtonCounters c;
        c.solves = solves;
        c.iterations = iterations;
        c.analyses = analyses;
        c.factorizations = factorizations;
        c.lm_attempts = lm_attempts;
        c.iterations_per_solve = std::move(iterations_per_solve);
        return c;
    }

    sparse::Matrix J;                    // combined Jacobian
    sparse::Matrix J_nl;                 // refresh target
    // Line-search / LM refresh target. Compressed after every refresh,
    // like J_nl. On Eigen 3.4.x, inserting into an emptied compressed
    // matrix (resize / setZero) hands all its spare capacity to the first
    // column, and an out-of-order insert then reallocates to that
    // capacity plus a few entries: a persistent target that is never
    // squeezed grows on every refill and is copied whole on every
    // insert (120 s CI timeout on Ubuntu's 3.4.0; Eigen 5 is immune).
    // makeCompressed squeezes the storage back to nnz.
    sparse::Matrix J_nl_trial;
    Vector f_nl, f_nl_trial, jx, f, neg_f, dx, x_trial, f_trial;
    Vector ro_f, ro_x;   // roundoff acceptance: f(x_k), x_k of the last iteration
    Vector ro_scale;     // roundoff acceptance: row scale at the call's start state
    Real ro_x0_inf = 0;  // roundoff acceptance: |x_0|_inf of the call
    std::unique_ptr<sparse::DirectSolver> solver;   // analysed for J
    std::vector<Real> ls, ns;            // operands scattered on J

    /// Write `J = L + N` (bit-equal to `L; += N` in Eigen) and return
    /// true when the analysis of the previous call still applies.
    bool combine(const sparse::Matrix& L, const sparse::Matrix& N) {
        bool same = valid_ && same_pattern_(L, l_outer_, l_inner_) &&
                    same_pattern_(N, n_outer_, n_inner_);
        if (!same) {
            record_(L, l_outer_, l_inner_);
            record_(N, n_outer_, n_inner_);
            if (N.nonZeros() > 0) {
                J = L + N;   // union pattern
            } else {
                J = L;
            }
            J.makeCompressed();
            slots_(L, l2j_);
            slots_(N, n2j_);
            valid_ = true;
        }
        const auto nnz = static_cast<std::size_t>(J.nonZeros());
        Real* jv = J.valuePtr();
        if (N.nonZeros() == 0) {
            // Today's path copies J_lin unchanged when J_nl is empty.
            std::copy_n(L.valuePtr(), nnz, jv);
            return same;
        }
        ls.assign(nnz, Real{0});
        ns.assign(nnz, Real{0});
        const Real* lv = L.valuePtr();
        for (std::size_t k = 0; k < l2j_.size(); ++k) ls[l2j_[k]] = lv[k];
        const Real* nv = N.valuePtr();
        for (std::size_t k = 0; k < n2j_.size(); ++k) ns[n2j_[k]] = nv[k];
        for (std::size_t k = 0; k < nnz; ++k) jv[k] = ls[k] + ns[k];
        return same;
    }

private:
    bool valid_ = false;
    std::vector<Index> l_outer_, l_inner_, n_outer_, n_inner_, l2j_, n2j_;

    static bool same_pattern_(const sparse::Matrix& A,
                              const std::vector<Index>& outer,
                              const std::vector<Index>& inner) {
        const auto no = static_cast<std::size_t>(A.outerSize() + 1);
        const auto nnz = static_cast<std::size_t>(A.nonZeros());
        return outer.size() == no && inner.size() == nnz &&
               std::equal(outer.begin(), outer.end(), A.outerIndexPtr()) &&
               std::equal(inner.begin(), inner.end(), A.innerIndexPtr());
    }
    static void record_(const sparse::Matrix& A, std::vector<Index>& outer,
                        std::vector<Index>& inner) {
        outer.assign(A.outerIndexPtr(), A.outerIndexPtr() + A.outerSize() + 1);
        inner.assign(A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros());
    }
    void slots_(const sparse::Matrix& A, std::vector<Index>& out) const {
        out.resize(static_cast<std::size_t>(A.nonZeros()));
        const Index* ap = A.outerIndexPtr();
        const Index* ai = A.innerIndexPtr();
        const Index* up = J.outerIndexPtr();
        const Index* ui = J.innerIndexPtr();
        for (Index j = 0; j < A.outerSize(); ++j) {
            Index q = up[j];
            for (Index p = ap[j]; p < ap[j + 1]; ++p) {
                while (ui[q] != ai[p]) ++q;
                out[static_cast<std::size_t>(p)] = q;
            }
        }
    }
};

/// Transient-only acceptance of an iterate whose residual is at the
/// roundoff of its own row (add-newton-roundoff-acceptance). Checked
/// only where the solve would throw for an exhausted budget; DC
/// callers pass nothing and compile to the old loop. The fields after
/// `dx_rel_max` report the last call.
struct NewtonRoundoffAcceptance {
    bool enabled = false;
    Real omega_max = Real{8} * std::numeric_limits<Real>::epsilon();
    Real dx_rel_max = Real{1e-6};
    /// Run budget: the longest run of accepted solves with no converged
    /// solve between them. Accepts are for isolated solves; a run of
    /// them is a state the solver is not tracking (BUDGET in DECISIONS).
    Size max_consecutive = Size{130};
    bool accepted = false;
    Real omega = 0;
    Index row = kInvalidIndex;
    /// Run counters (the caller keeps one object per run).
    Size consecutive = 0;
    Size longest = 0;
    Size total = 0;
};

/// Thrown when the run's roundoff-acceptance budget is exhausted. A
/// named error, not a Newton failure: no caller may retry it at a smaller
/// step (a retry is how the fixed engine's ladder would swallow it).
struct NewtonRoundoffBudgetExceeded : std::runtime_error {
    using std::runtime_error::runtime_error;
};

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
///
/// In-place form (task 2.4): `x` is the initial iterate on entry and the
/// solution on return; on a throw it holds the last iterate.
inline void solve_with_newton_in_place(
    const PwlSegment& seg,
    const NonlinearRefreshFn& refresh,
    const topology::Graph& graph,
    const DevicePool& pool,
    Vector& x,
    const Vector& b_extra,
    Size max_iters = Size{50},
    Real tol_dx  = Real{1e-9},
    Real tol_res = Real{1e-9},
    bool enable_line_search = false,
    bool enable_lm = false,
    NewtonWorkspace* ws = nullptr,
    NewtonRoundoffAcceptance* roundoff = nullptr) {
    const Size n = seg.state_size;
    const bool ro_on = roundoff != nullptr && roundoff->enabled;
    if (roundoff != nullptr) {
        roundoff->accepted = false;
    }
    // Roundoff acceptance state: whether f(x_k), x_k were recorded this
    // call (in the workspace: no allocation per call) and the last three
    // ||dx||.
    bool ro_set = false;
    bool ro_scaled = false;
    // ||dx|| of the last six iterations, newest first.
    Real ro_dx[6];
    std::fill(std::begin(ro_dx), std::end(ro_dx),
              std::numeric_limits<Real>::infinity());
    if (static_cast<Size>(x.size()) != n) {
        x = Vector::Zero(static_cast<Index>(n));
    }

    // A call without a run workspace gets a call-local one: the same
    // code path, with the analysis reused across this call's iterations,
    // on Eigen's LU (the DC paths keep their factorisation).
    // Built only when needed: an Eigen sparse matrix allocates on
    // default construction.
    std::optional<NewtonWorkspace> local_ws;
    if (ws == nullptr) {
        local_ws.emplace();
        local_ws->backend = sparse::Backend::Eigen;
    }
    NewtonWorkspace& w = ws != nullptr ? *ws : *local_ws;
    ++w.solves;
    // Records this call's iteration count on every exit, throws included.
    struct TraceGuard {
        NewtonWorkspace& w;
        std::uint64_t start;
        ~TraceGuard() {
            const auto k = std::min<std::uint64_t>(w.iterations - start, 65535);
            w.iterations_per_solve.push_back(static_cast<std::uint16_t>(k));
        }
    } trace_guard{w, w.iterations};
    sparse::Matrix& J_nl = w.J_nl;
    // J_nl keeps its pattern between calls of one run (task 2.3): the
    // refreshers zero its values and stamp into existing entries.
    if (J_nl.rows() != static_cast<Index>(n) || J_nl.cols() != static_cast<Index>(n)) {
        J_nl.resize(static_cast<Index>(n), static_cast<Index>(n));
    }
    sparse::zero_keep_pattern(J_nl);
    Vector& f_nl = w.f_nl;
    f_nl.setZero(static_cast<Index>(n));
    Vector& dx = w.dx;

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

        ++w.iterations;
        // 1. Refresh nonlinear contributions at current x.
        const Real nl_residual_norm =
            refresh(x, J_nl, f_nl, graph, pool);
        sparse::compress_in_place(J_nl);

        // 2. Combined J = J_lin + J_nl, in place on the workspace's
        //    union pattern (bit-equal to `J = seg.J; J += J_nl`).
        //    A caller's uncompressed `seg.J` is compressed in a copy,
        //    as the old path did.
        std::optional<sparse::Matrix> L_compressed;   // see local_ws
        const sparse::Matrix* L = &seg.J;
        if (!seg.J.isCompressed()) {
            L_compressed.emplace(seg.J);
            L_compressed->makeCompressed();
            L = &*L_compressed;
        }
        const bool reuse_analysis = w.combine(*L, J_nl);
        const sparse::Matrix& J_combined = w.J;

        // 3. Build combined f(x) = J_lin·x + (b_constant +
        //    b_extra) + f_nl — the product into a hoisted buffer
        //    first, then the additions in the original order.
        w.jx.noalias() = seg.J * x;
        Vector& f_combined = w.f;
        f_combined = w.jx + seg.b_constant + b_extra + f_nl;

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
        // Roundoff acceptance: the row scale is taken HERE, at the start
        // state of the call (iteration 0: the step's start or its warm
        // start) and the step's sources, never at the iterate the check
        // runs on. Read at a diverging iterate it grew with |x| and let a
        // 2.4e13 A state through (ll16_flyback_closed_loop on pwl).
        if (ro_on && !ro_scaled) {
            const Vector ax0 = x.cwiseAbs();
            w.ro_scale = L->cwiseAbs() * ax0;
            w.ro_scale += seg.b_constant.cwiseAbs() + b_extra.cwiseAbs()
                        + f_nl.cwiseAbs();
            if (J_nl.nonZeros() > 0) {
                w.ro_scale += J_nl.cwiseAbs() * ax0;
            }
            w.ro_x0_inf = x.size() == 0 ? Real{0} : x.cwiseAbs().maxCoeff();
            ro_scaled = true;
        }
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
                if (roundoff != nullptr) roundoff->consecutive = 0;
                return;
            }
            sparse::Matrix& J_nl_trial = w.J_nl_trial;
            if (J_nl_trial.rows() != static_cast<Index>(n)) {
                J_nl_trial.resize(static_cast<Index>(n), static_cast<Index>(n));
            }
            sparse::zero_keep_pattern(J_nl_trial);
            Vector& f_nl_trial = w.f_nl_trial;
            f_nl_trial.setZero(static_cast<Index>(n));
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
                ++w.lm_attempts;
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
                Vector& x_trial = w.x_trial;
                x_trial = x + dx;
                (void)refresh(x_trial, J_nl_trial, f_nl_trial,
                               graph, pool);
                sparse::compress_in_place(J_nl_trial);   // see J_nl_trial
                Vector& f_trial = w.f_trial;
                f_trial =
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
            // Plain Newton + optional line search. One analysis per
            // pattern, reused while the pattern holds (stage A1).
            if (!reuse_analysis || !w.solver) {
                w.solver = sparse::make_default_solver(n, w.backend);
                ++w.analyses;
                if (!w.solver->analyze(J_combined)) {
                    auto& solver = w.solver;
                    std::string msg = std::format(
                        "solve_with_newton: combined matrix is "
                        "structurally singular at iter {}{}",
                        iter,
                        explain_singular(graph, pool, J_combined,
                                          solver.get()));
                    w.solver.reset();
                    throw std::runtime_error(msg);
                }
            }
            auto& solver = w.solver;
            ++w.factorizations;
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
            Vector& neg_f = w.neg_f;
            neg_f = -f_combined;
            solver->solve(neg_f, dx);

            if (enable_line_search) {
                sparse::Matrix& J_nl_trial = w.J_nl_trial;
                if (J_nl_trial.rows() != static_cast<Index>(n)) {
                    J_nl_trial.resize(static_cast<Index>(n),
                                      static_cast<Index>(n));
                }
                sparse::zero_keep_pattern(J_nl_trial);
                Vector& f_nl_trial = w.f_nl_trial;
                f_nl_trial.setZero(static_cast<Index>(n));
                bool accepted = false;
                for (Size bt = 0; bt < Size{8}; ++bt) {
                    Vector& x_trial = w.x_trial;
                    x_trial = x + alpha * dx;
                    (void)refresh(x_trial, J_nl_trial, f_nl_trial,
                                   graph, pool);
                    sparse::compress_in_place(J_nl_trial);   // see J_nl_trial
                    Vector& f_trial = w.f_trial;
                    f_trial =
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
        if (ro_on) {
            w.ro_f = f_combined;
            w.ro_x = x;
            ro_set = true;
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
        for (int k = 5; k > 0; --k) ro_dx[k] = ro_dx[k - 1];
        ro_dx[0] = last_dx_norm;
        if (last_dx_norm < tol_dx && last_res_norm < tol_res) {
            if (roundoff != nullptr) roundoff->consecutive = 0;
            return;
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
                if (roundoff != nullptr) roundoff->consecutive = 0;
                return;
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

    // Budget exhausted. Transient callers may accept x_k (the iterate
    // whose residual is checked, not x_k + dx_k) when ||dx|| neither
    // contracted by 4 over the last two steps nor grew over the last
    // three (a two-sided stagnation test), ||dx_k|| <= 1e-6 of the START
    // state, and every row's residual at x_k is within omega_max of the
    // magnitudes that row sums at the START state (Oettli-Prager backward
    // error against the call's own scale). Nothing taken from the iterate
    // can make the test easier as the iterate diverges. The absolute
    // tol_dx stays the test everywhere else.
    std::string ro_note;
    if (ro_on && ro_set && ro_scaled) {
        const Vector& ro_f = w.ro_f;
        // No NET contraction over two steps. "Each step at least half
        // the previous" read a roundoff limit cycle (||dx|| 3.2e-7,
        // 5.0e-7, 1.9e-7) as converging. Quadratic convergence, or
        // linear at a rate below 1/2, cuts ||dx|| by more than 4 in two
        // steps; a slower rate counts as stagnated and the tests below
        // decide.
        const bool no_contraction = ro_dx[0] >= Real{0.25} * ro_dx[2];
        // ...and no growth: the largest of the last three is at most
        // twice the largest of the three before (a limit cycle repeats
        // its maxima; a diverging iterate does not).
        const Real m_new = std::max({ro_dx[0], ro_dx[1], ro_dx[2]});
        const Real m_old = std::max({ro_dx[3], ro_dx[4], ro_dx[5]});
        const bool growing = !(m_new <= Real{2} * m_old);
        const bool stagnated = no_contraction && !growing;
        const Real dx_cap = roundoff->dx_rel_max
            * std::max(Real{1}, w.ro_x0_inf);
        const Vector& scale = w.ro_scale;
        Real omega = Real{0};
        Index row = kInvalidIndex;
        for (Index i = 0; i < static_cast<Index>(n); ++i) {
            const Real fi = std::abs(ro_f[i]);
            const Real be = scale[i] > Real{0}
                ? fi / scale[i]
                : (fi == Real{0} ? Real{0}
                                 : std::numeric_limits<Real>::infinity());
            if (be > omega) {
                omega = be;
                row = i;
            }
        }
        if (stagnated && ro_dx[0] <= dx_cap
            && omega <= roundoff->omega_max) {
            if (roundoff->consecutive + 1 > roundoff->max_consecutive) {
                throw NewtonRoundoffBudgetExceeded(std::format(
                    "solve_with_newton: {} consecutive Newton solves ended "
                    "at the iteration budget and would each have been "
                    "accepted at round-off level (worst row {} at {}, "
                    "backward error {:.2g} eps; {} accepted in this run). "
                    "Round-off acceptance is for isolated solves; a run of "
                    "them means the solver is not tracking this state. "
                    "Check the initial state and the device named, and "
                    "cross-check on the other transient engine.",
                    roundoff->consecutive + 1,
                    row_equation_label(graph, pool, row), row,
                    omega / std::numeric_limits<Real>::epsilon(),
                    roundoff->total));
            }
            x = w.ro_x;
            roundoff->accepted = true;
            roundoff->omega = omega;
            roundoff->row = row;
            ++roundoff->consecutive;
            ++roundoff->total;
            roundoff->longest = std::max(roundoff->longest,
                                         roundoff->consecutive);
            return;
        }
        ro_note = std::format(
            " Roundoff acceptance declined: row backward error {:.2g} eps "
            "at {} (limit {:.2g} eps), ||dx|| {} 1e-6·|x_0|, last steps "
            "{} (||dx|| {:.3e}, {:.3e}, {:.3e}).",
            omega / std::numeric_limits<Real>::epsilon(),
            row_equation_label(graph, pool, row),
            roundoff->omega_max / std::numeric_limits<Real>::epsilon(),
            ro_dx[0] <= dx_cap ? "within" : "above",
            growing ? "growing" : (no_contraction ? "stagnated"
                                                  : "still contracting"),
            ro_dx[2], ro_dx[1], ro_dx[0]);
    }

    // Report the norms RELATIVE to the circuit's own scale as well as
    // absolutely. `tol_dx` and `tol_res` are absolute, deliberately —
    // an absolute step test is what keeps Newton off a spurious branch
    // of a multi-root system, and making it relative was measured to
    // let the DC cascade stop 10.67 V from the homotopy's answer. But
    // an absolute 1e-9 on a 300 V converter switching at kilohertz is
    // 3e-12 relative, which no Jacobian delivers, and the user staring
    // at "1.0e-06" has no way to tell whether that is a real
    // non-convergence or a tolerance that does not fit their circuit.
    // So the message gives both — and it must NOT tell them to loosen
    // the tolerances when the relative figures look converged. It
    // used to, and following it was measured to be a 4.3x wrong
    // answer: the snubbed flyback of test_saturable_transformer.py
    // aborts at h_min on TR-BDF2, and with tol_newton_dx = 1e-5 /
    // tol_newton_res = 1e-4 it "completes" at V_out = 3.53 V against
    // 15.31 V on engine='pwl', dt = 1e-8. Point at the device and at
    // the OTHER engine as a cross-check instead. Do not prescribe a
    // dt direction here: on the fixed engine the documented Newton
    // failure (diode-mask chatter) gets WORSE with a smaller dt, and
    // run_transient adds its own engine-specific note.
    const Real x_scale = std::max(
        Real{1}, x.size() == 0 ? Real{0} : x.cwiseAbs().maxCoeff());
    const Real rhs_scale = std::max({
        Real{1},
        seg.b_constant.size() == 0 ? Real{0}
                                   : seg.b_constant.cwiseAbs().maxCoeff(),
        b_extra.size() == 0 ? Real{0} : b_extra.cwiseAbs().maxCoeff()});
    throw std::runtime_error(std::format(
        "solve_with_newton: failed to converge after {} iterations "
        "(||dx||_inf = {:.3e} worst at {}, ||residual||_inf = {:.3e} "
        "worst at {}). Relative to this circuit's own scale that is "
        "dx/|x| = {:.2e} (|x| up to {:.3g}) and res/|b| = {:.2e} "
        "(|b| up to {:.3g}), against absolute tolerances of {:.1e} and "
        "{:.1e}. Do not loosen tol_newton_dx / tol_newton_res to get "
        "past this, even if the relative figures look converged: the "
        "absolute step test is what keeps Newton off a spurious root, "
        "and loosening it has produced silently wrong operating points "
        "(a flyback that aborted this way, re-run with loosened "
        "tolerances, reported 3.5 V instead of 15.3 V). Instead, look "
        "at the device named above, and cross-check on the other "
        "transient engine (engine='pwl' with an explicit dt, if this "
        "came from TR-BDF2 or 'auto').{}",
        max_iters,
        last_dx_norm, row_label(graph, pool, worst_dx_row),
        last_res_norm,
        row_equation_label(graph, pool, worst_res_row),
        last_dx_norm / x_scale, x_scale,
        last_res_norm / rhs_scale, rhs_scale,
        tol_dx, tol_res, ro_note));
}

/// By-value form of `solve_with_newton_in_place`: `x_init` is left
/// untouched when the solve throws.
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
    bool enable_lm = false,
    NewtonWorkspace* ws = nullptr,
    NewtonRoundoffAcceptance* roundoff = nullptr) {
    Vector x = x_init;
    solve_with_newton_in_place(seg, refresh, graph, pool, x, b_extra,
                               max_iters, tol_dx, tol_res,
                               enable_line_search, enable_lm, ws, roundoff);
    return x;
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
