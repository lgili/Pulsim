#pragma once

// =============================================================================
// Pulsim — the whole Newton shooting for a periodic orbit, in the kernel
// =============================================================================
//
// add-design-optimization-workflow, DECISIONS 16/18. `period_map_from_run`
// (period_integrals.hpp) moved one period of the exact stepper and its
// integrals out of Python; the Newton loop around it stayed there and held
// the GIL for most of a buck evaluation. This header is that loop,
// statement for statement the one of pulsim/periodic_steady_state.py
// (`periodic_steady_state`, the kernel-map branch):
//
//   G(x0) = P(x0) − x0,   (I − M)·δ = G,   x0 ← x0 + λ·δ (Armijo, λ = 2^-k)
//
// with the scaled norm max_i |G_i| / (rtol·(scale_i + |x0_i|)), the stopping
// rule on the Newton STEP (or one polish step from a converged residual),
// three rounds of `n_relax` relaxation periods when no λ is accepted, the
// period-run budget, the singular (I − M) and non-isolated (|μ| = 1) checks,
// the Floquet multipliers of M, and a verification period carrying the
// caller's observables whose return to x0 is checked. Every outcome that is
// not a converged orbit is a status with the data the caller formats its
// typed error from; nothing is accepted on a weaker test than the Python
// loop's.
//
// `run(x, with_observables)` is one period from x (a PeriodMapResult with
// M); it throws PeriodRunFailed for a run that did not complete.

#include "pulsim/dsed/period_integrals.hpp"
#include "pulsim/numeric/types.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pulsim::dsed {

/// A period run that did not complete (the engine's own non-convergence).
struct PeriodRunFailed : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct OrbitOptions {
    Real rtol = Real{1e-9};
    int max_period_runs = 60;
    int n_relax = 5;
    Real armijo = Real{1e-4};
    int n_lambda = 9;               // λ ∈ {1, 1/2, …, 2^-8}
    Real marginal = Real{1e-10};    // | |μ| − 1 | below this: not isolated
    int max_relax_rounds = 3;
};

enum class OrbitStatus {
    Converged,
    Budget,           // max_period_runs reached (verify_budget: at the check)
    NoProgress,       // three relaxation rounds without an accepted step
    Singular,         // I − M singular during the iteration
    Marginal,         // a multiplier of the converged M on the unit circle
    Grazing,          // a period without M (an event tangent to its surface)
    RunFailed,        // a period run threw PeriodRunFailed
    Irreproducible,   // the verification period did not return to x0
};

inline const char* to_string(OrbitStatus s) {
    switch (s) {
        case OrbitStatus::Converged: return "converged";
        case OrbitStatus::Budget: return "budget";
        case OrbitStatus::NoProgress: return "no_progress";
        case OrbitStatus::Singular: return "singular";
        case OrbitStatus::Marginal: return "marginal";
        case OrbitStatus::Grazing: return "grazing";
        case OrbitStatus::RunFailed: return "run_failed";
        case OrbitStatus::Irreproducible: return "irreproducible";
    }
    return "?";
}

template <class MaskT>
struct OrbitResult {
    OrbitStatus status = OrbitStatus::Converged;
    std::string message;              // run failure / grazing text
    bool verify_budget = false;       // Budget hit before the verification
    Vector x0;                        // the orbit (or the last iterate)
    DenseMatrix monodromy;            // M at x0
    Eigen::VectorXcd multipliers;     // eigenvalues of M
    Vector scale;                     // max |x_i(t)| over the normed runs
    std::vector<Real> history;        // max|P(x) − x| per iteration
    std::vector<std::vector<std::string>> patterns;   // event names
    std::vector<std::pair<std::string, Real>> last_pattern;   // (name, t/T)
    Eigen::VectorXcd last_mu;         // multipliers of the last M
    Vector marginal_direction;        // |v| of the singular / marginal mode
    std::complex<Real> marginal_mu{0, 0};
    Real residual = Real{0};          // max|P(x0) − x0| of the verification
    int n_runs = 0;
    int n_iterations = 0;
    int n_relax_rounds = 0;
    std::size_t n_steps = 0;
    PeriodMapResult<MaskT> period;    // the verification period
    bool has_period = false;
};

namespace detail {
/// max |v_i|, NaN if any entry is NaN (numpy's max semantics).
inline Real max_abs(const Vector& v) {
    Real m = Real{0};
    for (Index i = 0; i < v.size(); ++i) {
        if (std::isnan(v[i])) {
            return v[i];
        }
        m = std::max(m, std::abs(v[i]));
    }
    return m;
}
}  // namespace detail

/// Newton shooting to the periodic orbit from `x` (see the header note).
template <class MaskT, class RunFn>
[[nodiscard]] OrbitResult<MaskT> periodic_orbit_newton(
    RunFn&& run, Vector x, Real T, const OrbitOptions& opt) {
    using detail::max_abs;
    OrbitResult<MaskT> out;
    const Index n = x.size();
    out.scale = Vector::Zero(n);
    const Real rtol = opt.rtol;

    // max_i |v_i| / (rtol (scale_i + |x0_i|)), 0 where v_i = 0; NaN
    // propagates (numpy's max does, and a NaN must never pass a test)
    auto scaled = [&](const Vector& v, const Vector& x0) {
        Real q = Real{0};
        for (Index i = 0; i < n; ++i) {
            if (v[i] == Real{0}) {
                continue;
            }
            const Real den = rtol * (out.scale[i] + std::abs(x0[i]));
            const Real qi = std::abs(v[i]) / (den > Real{0} ? den : Real{1e-300});
            if (std::isnan(qi)) {
                return qi;
            }
            q = std::max(q, qi);
        }
        return q;
    };
    // the scaled norm; it folds the run's scale in first (Python `norm`)
    auto norm = [&](const PeriodMapResult<MaskT>& tr, const Vector& x0) {
        out.scale = out.scale.cwiseMax(tr.scale);
        return scaled(tr.x_end - x0, x0);
    };
    // one period; false (status set) when the budget is spent or it failed
    auto run_one = [&](const Vector& x0, bool obs,
                       PeriodMapResult<MaskT>& tr) -> bool {
        if (out.n_runs >= opt.max_period_runs) {
            out.status = OrbitStatus::Budget;
            return false;
        }
        ++out.n_runs;
        try {
            tr = run(x0, obs);
        } catch (const PeriodRunFailed& e) {
            out.status = OrbitStatus::RunFailed;
            out.message = e.what();
            return false;
        }
        out.n_steps += tr.n_steps;
        return true;
    };
    auto fail = [&](const Vector& x0) {
        out.x0 = x0;
        return out;
    };

    PeriodMapResult<MaskT> traj;
    Vector tx = x;    // traj's x0
    {
        // the first run is outside the budget check (Python: s.run(x))
        ++out.n_runs;
        try {
            traj = run(tx, false);
        } catch (const PeriodRunFailed& e) {
            out.status = OrbitStatus::RunFailed;
            out.message = e.what();
            return fail(tx);
        }
        out.n_steps += traj.n_steps;
    }
    const DenseMatrix I = DenseMatrix::Identity(n, n);
    int relax_rounds = 0;
    bool polished = false;
    for (;;) {
        ++out.n_iterations;
        const Real r = norm(traj, tx);
        const Vector G = traj.x_end - tx;
        out.history.push_back(max_abs(G));
        std::vector<std::string> names;
        out.last_pattern.clear();
        for (const auto& e : traj.events) {
            names.push_back(e.name);
            out.last_pattern.emplace_back(e.name, e.t / T);
        }
        out.patterns.push_back(std::move(names));
        if (traj.monodromy.size() == 0) {
            out.status = OrbitStatus::Grazing;
            out.message = traj.grazing.empty()
                ? std::string("the period map returned no M") : traj.grazing;
            return fail(tx);
        }
        const DenseMatrix& M = traj.monodromy;
        out.last_mu = Eigen::EigenSolver<DenseMatrix>(M, false).eigenvalues();
        const DenseMatrix IM = I - M;
        Eigen::JacobiSVD<DenseMatrix> svd(IM, Eigen::ComputeFullV);
        const auto& sig = svd.singularValues();
        const bool singular =
            sig[n - 1] <= Real{1e-10} * std::max(Real{1}, sig[0]);
        const Vector delta = IM.partialPivLu().solve(G);
        if (r <= Real{1}) {
            // a small residual is not a small state error with a slow mode
            // (|μ| → 1): stop on the Newton step, or after one polish step
            if (singular || polished || scaled(delta, tx) <= Real{1}) {
                break;
            }
        }
        if (singular) {
            out.status = OrbitStatus::Singular;
            out.marginal_direction = svd.matrixV().col(n - 1).cwiseAbs();
            return fail(tx);
        }
        bool accepted = false;
        Real lam = Real{1};
        for (int k = 0; k < opt.n_lambda; ++k) {
            PeriodMapResult<MaskT> tt;
            const Vector xl = tx + lam * delta;
            if (!run_one(xl, false, tt)) {
                return fail(tx);
            }
            const Real r_new = norm(tt, xl);
            if (r_new <= (Real{1} - opt.armijo * lam) * norm(traj, tx)
                || (r <= Real{1} && r_new <= Real{1})) {
                polished = r <= Real{1};
                traj = std::move(tt);
                tx = xl;
                accepted = true;
                break;
            }
            lam *= Real{0.5};
        }
        if (accepted) {
            continue;
        }
        ++relax_rounds;
        out.n_relax_rounds = relax_rounds;
        if (relax_rounds > opt.max_relax_rounds) {
            out.status = OrbitStatus::NoProgress;
            return fail(tx);
        }
        for (int k = 0; k < opt.n_relax; ++k) {
            PeriodMapResult<MaskT> tt;
            const Vector xr = traj.x_end;
            if (!run_one(xr, false, tt)) {
                return fail(tx);
            }
            traj = std::move(tt);
            tx = xr;
        }
    }

    // ---- the orbit ----
    out.x0 = tx;
    out.monodromy = traj.monodromy;
    Eigen::EigenSolver<DenseMatrix> es(out.monodromy, true);
    out.multipliers = es.eigenvalues();
    for (Index i = 0; i < n; ++i) {
        const auto m = out.multipliers[i];
        if (std::abs(std::abs(m) - Real{1}) < opt.marginal) {
            out.status = OrbitStatus::Marginal;
            out.marginal_mu = m;
            out.marginal_direction = es.eigenvectors().col(i).cwiseAbs();
            return out;
        }
    }
    if (out.n_runs >= opt.max_period_runs) {
        out.status = OrbitStatus::Budget;
        out.verify_budget = true;
        return out;
    }
    ++out.n_runs;
    try {
        out.period = run(tx, true);
    } catch (const PeriodRunFailed& e) {
        out.status = OrbitStatus::RunFailed;
        out.message = e.what();
        return out;
    }
    out.has_period = true;
    out.n_steps += out.period.n_steps;
    out.residual = max_abs(out.period.x_end - tx);
    const Real floor = rtol * (out.scale + tx.cwiseAbs()).maxCoeff();
    if (!(out.residual <= Real{1e3} * std::max(out.history.back(), floor))) {
        out.status = OrbitStatus::Irreproducible;
    }
    return out;
}

}  // namespace pulsim::dsed
