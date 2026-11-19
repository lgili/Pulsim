#pragma once

// add-high-order-stiff-integrator: a body for the step skeleton on small
// dense MNA systems  C·x' + G·x + g(x) = b(t)  (q = C·x, f = G·x + g − b).
// It plays the engine's part for Esdirk43 and TrBdf2 (one factor per
// (h, Jacobian) key, simplified Newton per stage, stage predictors,
// filtered error), so the two integrators are compared on the same
// services. Test-only; the engine-switch step writes the real body.

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <vector>

#include "pulsim/solver/step/esdirk43.hpp"
#include "pulsim/solver/step/stepper.hpp"

namespace mna {

using pulsim::Real;
using pulsim::Size;
namespace step = pulsim::step;
using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

struct Circuit {
    Mat G, C;
    std::function<Vec(Real)> b;
    std::function<Vec(const Vec&)> g;    // nonlinear currents (optional)
    std::function<Mat(const Vec&)> dg;   // their Jacobian
    std::vector<int> norm_vars;          // differential unknowns: the error norm
    [[nodiscard]] Vec f(const Vec& x, Real t) const {
        Vec r = G * x - b(t);
        if (g) r += g(x);
        return r;
    }
    [[nodiscard]] Mat jac(const Vec& x) const { return dg ? Mat(G + dg(x)) : G; }
    [[nodiscard]] bool linear() const { return !g; }
    [[nodiscard]] bool alg_row(Eigen::Index i) const { return C.row(i).cwiseAbs().maxCoeff() == 0; }
    [[nodiscard]] bool alg_col(Eigen::Index j) const { return C.col(j).cwiseAbs().maxCoeff() == 0; }
    /// The engine's consistent solve: x + Z·y with C·Z = 0 (q unchanged)
    /// and the algebraic combinations N^T·f = 0 (N^T·C = 0) at t; index 1
    /// makes N^T·J·Z nonsingular. Newton for a nonlinear g.
    [[nodiscard]] Vec consistent(const Vec& x, Real t) const {
        const Eigen::FullPivLU<Mat> lc(C);
        if (lc.rank() == C.rows()) return x;
        const Mat Z = lc.kernel();
        const Mat N = Eigen::FullPivLU<Mat>(C.transpose()).kernel();
        Vec y = Vec::Zero(Z.cols());
        for (int k = 0; k < 30; ++k) {
            const Vec xc = x + Z * y;
            const Vec dy = (N.transpose() * jac(xc) * Z).fullPivLu().solve(N.transpose() * f(xc, t));
            y -= dy;
            if (dy.norm() <= 1e-15 * (1 + y.norm())) break;
        }
        return x + Z * y;
    }
};

template <class I>
struct Body {
    static constexpr bool is_esdirk = std::is_same_v<I, step::Esdirk43>;
    const Circuit& m;
    Vec x, x0, x_save;
    Real t = 0;
    Real rtol = 1e-6, atol = 1e-9;
    Real newton_tol = 0.03;   // weighted RMS of a Newton update, in tolerance units
    int newton_max = 8;
    bool filter = true;
    bool restart = true;      // no previous step to extrapolate from
    std::array<Vec, 5> Y, d;  // stage values; increments q(Y) − ψ
    Vec hqd0, qd_g;
    // counters
    Size n_factor = 0, n_solve = 0, n_newton = 0, n_stage = 0, max_newton = 0;
    Size n_refresh = 0, n_newton_fail = 0;
    // factor cache: one entry keyed by (factor_dt, linearisation point)
    Eigen::PartialPivLU<Mat> lu;
    Real lu_dt = -1;
    Real lu_t = -1;
    // the accepted step's points for the stage predictor
    std::vector<std::pair<Real, Vec>> prev_pts, pts;
    std::function<void(const Body&, const step::StepFrame&)> hook;

    Body(const Circuit& c, Vec x_init) : m(c), x(std::move(x_init)) {}

    bool poll() { return true; }
    void begin(const step::StepFrame&) {}
    Real clamp(const step::StepFrame&, Real h) { return h; }
    void save() { x_save = x; }
    void restore() { x = x_save; }
    bool after_solve(step::StepFrame&) { return true; }

    [[nodiscard]] Real wnorm(const Vec& e, const Vec& a, const Vec& b2,
                             const std::vector<int>* vars) const {
        Real s = 0;
        int n = 0;
        const auto add = [&](Eigen::Index k) {
            const Real sc = atol + rtol * std::max(std::abs(a[k]), std::abs(b2[k]));
            s += (e[k] / sc) * (e[k] / sc);
            ++n;
        };
        if (vars) {
            for (const int k : *vars) add(k);
        } else {
            for (Eigen::Index k = 0; k < e.size(); ++k) add(k);
        }
        return std::sqrt(s / std::max(n, 1));
    }
    void factor(Real factor_dt, Real t0) {
        // linear: the dt-keyed cache (solve_at); nonlinear: one Jacobian per step
        if (factor_dt == lu_dt && (m.linear() || t0 == lu_t)) return;
        lu.compute(m.jac(x0) + (Real{2} / factor_dt) * m.C);
        lu_dt = factor_dt;
        lu_t = t0;
        ++n_factor;
    }
    [[nodiscard]] Vec predict(Real tau) const {
        const auto n = pts.size();
        if (n == 1) return pts[0].second;
        if (n == 2) {
            const auto& [ta, xa] = pts[0];
            const auto& [tb, xb] = pts[1];
            return xb + (tau - tb) / (tb - ta) * (xb - xa);
        }
        const auto& [ta, xa] = pts[n - 3];
        const auto& [tb, xb] = pts[n - 2];
        const auto& [tc, xc] = pts[n - 1];
        return ((tau - tb) * (tau - tc) / ((ta - tb) * (ta - tc))) * xa
               + ((tau - ta) * (tau - tc) / ((tb - ta) * (tb - tc))) * xb
               + ((tau - ta) * (tau - tb) / ((tc - ta) * (tc - tb))) * xc;
    }
    // Simplified Newton on r(Y) = 0 with the cached factor; r's Jacobian is
    // the factor's matrix (up to the linearisation point).
    // A stage that does not converge on the step's Jacobian gets one
    // refresh at its last finite iterate (counted), then fails the step.
    template <class R>
    bool newton(Vec& y, Real factor_dt, R&& residual) {
        ++n_stage;
        const Vec y_pred = y;
        for (int pass = 0; pass < 2; ++pass) {
            Vec y_ok = y;
            for (int k = 1; k <= newton_max; ++k) {
                const Vec dy = lu.solve(residual(y));
                ++n_solve;
                ++n_newton;
                y -= dy;
                if (!y.allFinite()) break;
                y_ok = y;
                if (m.linear() || wnorm(dy, y, y, nullptr) <= newton_tol) {
                    max_newton = std::max<Size>(max_newton, static_cast<Size>(k));
                    return true;
                }
            }
            if (pass == 1 || m.linear()) break;
            ++n_refresh;
            lu.compute(m.jac(y_ok) + (Real{2} / factor_dt) * m.C);
            ++n_factor;
            y = y_ok.allFinite() ? y_ok : y_pred;
        }
        ++n_newton_fail;
        return false;
    }

    Real t_start = 0, h_step = 0;
    bool take(const step::StepFrame& f) {
        x0 = x;
        t_start = f.t0;
        h_step = f.h;
        const Real h = f.h;
        pts.clear();
        if (!restart && prev_pts.size() >= 3) {
            pts.push_back(prev_pts[prev_pts.size() - 3]);
            pts.push_back(prev_pts[prev_pts.size() - 2]);
        }
        pts.emplace_back(f.t0, x0);
        hqd0 = -h * m.f(x0, f.t0);
        const Vec q0 = m.C * x0;
        if constexpr (is_esdirk) {
            for (int s = 0; s < I::n_stages; ++s) {
                const step::StageSpec sp = I::stage(h, s);
                factor(sp.factor_dt, f.t0);
                const Vec psi = I::psi(s, q0, hqd0, d);
                const Real ts = f.t0 + sp.t_off;
                const Real gh = sp.ctx_h;
                Vec y = predict(ts);
                if (!newton(y, sp.factor_dt, [&](const Vec& v) { return Vec((m.C * v - psi) / gh + m.f(v, ts)); }))
                    return false;
                Y[s] = y;
                d[s] = m.C * y - psi;
                pts.emplace_back(ts, y);
            }
        } else {
            using H = pulsim::pwl::HistoryState;
            const step::StageSpec s0 = I::stage(h, 0);
            factor(s0.factor_dt, f.t0);
            const Real tg = f.t0 + s0.t_off, gh = s0.ctx_h;
            const Vec qd0 = hqd0 / h;
            Vec yg = predict(tg);
            if (!newton(yg, s0.factor_dt, [&](const Vec& v) {
                    return Vec(Real{2} * (m.C * v - q0) / gh - qd0 + m.f(v, tg));
                }))
                return false;
            Y[0] = yg;
            pts.emplace_back(tg, yg);
            if constexpr (I::n_stages == 1) return true;   // Trapezoidal
            Vec y1 = predict(f.t1);
            const Vec hist = H::trbdf2_c2() * (m.C * yg) + H::trbdf2_c3() * q0;
            if (!newton(y1, s0.factor_dt, [&](const Vec& v) {
                    return Vec((H::trbdf2_c1() * (m.C * v) + hist) / h + m.f(v, f.t1));
                }))
                return false;
            Y[1] = y1;
            pts.emplace_back(f.t1, y1);
        }
        return true;
    }
    [[nodiscard]] const Vec& result() const { return Y[I::n_stages - 1]; }

    /// The error vector in x units (filtered unless `filter` is off).
    [[nodiscard]] Vec error_vector(const step::StepFrame& f) {
        const Real h = f.h;
        Vec eq;
        Real scale;   // the stage matrix's C coefficient
        if constexpr (is_esdirk) {
            eq = I::error_increment(hqd0, d);
            scale = Real{1} / (I::gamma * h);
        } else {
            const Real gh = I::stage(h, 0).ctx_h;
            const Vec qn = hqd0 / h;
            const Vec qg = -m.f(Y[0], f.t0 + gh);
            const Vec q1 = -m.f(Y[1], f.t1);
            eq.resize(qn.size());
            for (Eigen::Index i = 0; i < qn.size(); ++i) eq[i] = I::lte(h, qn[i], qg[i], q1[i]);
            scale = Real{2} / gh;
        }
        if (filter) {
            ++n_solve;
            return lu.solve(scale * eq);
        }
        Vec e = Vec::Zero(eq.size());   // raw: C must be diagonal on the norm
        for (const int k : m.norm_vars) e[k] = eq[k] / m.C(k, k);
        return e;
    }
    Real error(const step::StepFrame& f) {
        return wnorm(error_vector(f), x0, result(), &m.norm_vars);
    }
    void commit(const step::StepFrame&) { x = result(); }
    void on_step(const step::StepFrame& f) {
        if (hook) hook(*this, f);
        t = f.t1;
        prev_pts = pts;
        restart = false;
    }
    /// Dense output inside the step just taken (Esdirk43 only).
    [[nodiscard]] Vec dense(Real theta) const { return I::dense(theta, x0, Y); }
    /// The same with the algebraic unknowns re-solved at t0 + θh (q kept).
    [[nodiscard]] Vec dense_consistent(Real theta) const {
        return m.consistent(dense(theta), t_start + theta * h_step);
    }
};

}  // namespace mna
