#pragma once

// add-high-order-stiff-integrator: a body for the step skeleton on small
// dense MNA systems  q(x)' + G·x + g(x) = b(t)  (q = C·x + q_n(x),
// f = G·x + g − b). A nonlinear q (saturable flux, junction charge) is
// integrated in q: the stage unknowns are q, x is recovered by Newton
// with the device's own dq/dx.
// It plays the engine's part for Esdirk43 and TrBdf2 (one factor per
// (h, Jacobian) key, simplified Newton per stage, stage predictors,
// filtered error), so the two integrators are compared on the same
// services. Test-only; the engine-switch step writes the real body.

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
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
    std::function<Vec(const Vec&)> qn;   // nonlinear charge / flux (optional), added to C·x
    std::function<Mat(const Vec&)> dqn;  // its Jacobian
    std::vector<int> norm_vars;          // differential unknowns: the error norm
    [[nodiscard]] Vec f(const Vec& x, Real t) const {
        Vec r = G * x - b(t);
        if (g) r += g(x);
        return r;
    }
    [[nodiscard]] Mat jac(const Vec& x) const { return dg ? Mat(G + dg(x)) : G; }
    [[nodiscard]] Vec q(const Vec& x) const { return qn ? Vec(C * x + qn(x)) : Vec(C * x); }
    [[nodiscard]] Mat dq(const Vec& x) const { return dqn ? Mat(C + dqn(x)) : C; }
    [[nodiscard]] bool linear() const { return !g && !qn; }
    /// The engine's consistent solve: x + Z·y with dq·Z = 0 (q unchanged)
    /// and the algebraic combinations N^T·f = 0 (N^T·dq = 0) at t; index 1
    /// makes N^T·J·Z nonsingular. Newton for a nonlinear g.
    /// shortcut: Z, N from dq at x (structural zeros of a nonlinear q).
    [[nodiscard]] Vec consistent(const Vec& x, Real t) const {
        const Mat D = dq(x);
        const Eigen::FullPivLU<Mat> lc(D);
        if (lc.rank() == D.rows()) return x;
        const Mat Z = lc.kernel();
        const Mat N = Eigen::FullPivLU<Mat>(D.transpose()).kernel();
        Vec y = Vec::Zero(Z.cols());
        for (int k = 0; k < 30; ++k) {
            const Vec xc = x + Z * y;
            const Vec dy = (N.transpose() * jac(xc) * Z).fullPivLu().solve(N.transpose() * f(xc, t));
            y -= dy;
            if (dy.norm() <= 1e-15 * (1 + y.norm())) break;
        }
        return x + Z * y;
    }
    /// x from a target charge: R^T·(q(x) − q_t) = 0 on the range of dq and
    /// N^T·f(x, t) = 0 on its left null space (the consistent solve with q
    /// given). Newton with the device's own dq/dx from the guess x, damped
    /// by backtracking on |F| (tanh-type laws 2-cycle undamped). Returns
    /// false when it did not converge (x is then the last iterate).
    [[nodiscard]] bool from_q(const Vec& qt, Vec& x, Real t, Size* iters = nullptr) const {
        const Mat D0 = dq(x);
        const Mat R = Eigen::FullPivLU<Mat>(D0).image(D0);
        const Mat N = Eigen::FullPivLU<Mat>(D0.transpose()).kernel();
        const Eigen::Index n = x.size(), r = R.cols();
        const auto F = [&](const Vec& z) {
            Vec v(n);
            v.head(r) = R.transpose() * (q(z) - qt);
            if (r < n) v.tail(n - r) = N.transpose() * f(z, t);
            return v;
        };
        for (int k = 0; k < 50; ++k) {
            Mat J(n, n);
            J.topRows(r) = R.transpose() * dq(x);
            if (r < n) J.bottomRows(n - r) = N.transpose() * jac(x);
            const Vec F0 = F(x);
            Vec dx = J.fullPivLu().solve(F0);
            if (iters) ++*iters;
            const bool small = dx.norm() <= 1e-12 * (1 + x.norm());
            for (int j = 0; j < 30 && !small && !(F(Vec(x - dx)).norm() < F0.norm()); ++j) dx *= Real{0.5};
            x -= dx;
            if (!x.allFinite()) return false;
            if (small) return true;
        }
        return false;
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
    // Jacobian refresh: when an iteration contracts by less than
    // `theta_refresh` (|dy_k| > θ·|dy_{k−1}|, k ≥ 2) or a pass runs out of
    // iterations, refactor at the last finite iterate; at most `max_refresh`
    // per stage, then the stage fails the step.
    Real theta_refresh = 0.5;
    int max_refresh = 10;
    bool filter = true;
    bool restart = true;      // no previous step to extrapolate from
    // End the step on the quadrature q1 = q0 + h·Σ b_i·q̇(Y_i) (q̇ = −f at the
    // stage values) and recover x1 from it (from_q), instead of x1 = Y_6:
    // the flux / charge balance then holds to round-off whatever the
    // stages' Newton residuals (ESDIRK only).
    bool carry_q = false;
    Vec x_end;
    Size n_recover = 0;   // from_q iterations (each one a solve)
    std::array<Vec, 5> Y, d;  // stage values; increments q(Y) − ψ
    Vec hqd0, qd_g;
    // counters
    Size n_factor = 0, n_solve = 0, n_newton = 0, n_stage = 0, max_newton = 0;
    Size n_refresh = 0, n_newton_fail = 0;
    Size n_resid = 0;   // residual evaluations of the backtracking (no solve)
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
        lu.compute(m.jac(x0) + (Real{2} / factor_dt) * m.dq(x0));
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
    // the factor's matrix (up to the linearisation point). The factor is the
    // step's (one Jacobian per attempt) until a stage contracts slowly (a
    // saturating flux: λ' falls by 1e3 across the knee) or stalls; then it
    // is refreshed at the stage's last finite iterate (counted) and kept for
    // the rest of the step.
    template <class R>
    bool newton(Vec& y, Real factor_dt, R&& residual) {
        ++n_stage;
        const Vec y_pred = y;
        int its = 0;
        for (int pass = 0; pass <= max_refresh; ++pass) {
            Vec y_ok = y;
            Real w_prev = std::numeric_limits<Real>::infinity();
            for (int k = 1; k <= newton_max; ++k) {
                const Vec r = residual(y);
                Vec dy = lu.solve(r);
                ++n_solve;
                ++n_newton;
                ++its;
                // convergence is judged on the FULL step, before any damping
                const Real w = m.linear() ? 0 : wnorm(dy, Vec(y - dy), Vec(y - dy), nullptr);
                if (pass > 0 && w > newton_tol) {
                    // after a refresh: backtrack on the residual norm (a
                    // sigmoidal law such as tanh makes plain Newton 2-cycle);
                    // no descent on this Jacobian is a stall (refresh / fail)
                    const Real r0 = r.norm();
                    bool descent = false;
                    for (int j = 0; j < 30 && !descent; ++j) {
                        ++n_resid;
                        descent = residual(Vec(y - dy)).norm() < r0;
                        if (!descent) dy *= Real{0.5};
                    }
                    if (!descent) break;
                }
                y -= dy;
                if (!y.allFinite()) break;
                y_ok = y;
                if (w <= newton_tol) {
                    max_newton = std::max<Size>(max_newton, static_cast<Size>(its));
                    return true;
                }
                if (k >= 2 && w > theta_refresh * w_prev) break;   // slow: refresh now
                w_prev = w;
            }
            if (pass == max_refresh || m.linear()) break;
            ++n_refresh;
            lu.compute(m.jac(y_ok) + (Real{2} / factor_dt) * m.dq(y_ok));
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
        const Vec q0 = m.q(x0);
        if constexpr (is_esdirk) {
            for (int s = 0; s < I::n_stages; ++s) {
                const step::StageSpec sp = I::stage(h, s);
                factor(sp.factor_dt, f.t0);
                const Vec psi = I::psi(s, q0, hqd0, d);
                const Real ts = f.t0 + sp.t_off;
                const Real gh = sp.ctx_h;
                Vec y = predict(ts);
                if (!newton(y, sp.factor_dt, [&](const Vec& v) { return Vec((m.q(v) - psi) / gh + m.f(v, ts)); }))
                    return false;
                Y[s] = y;
                d[s] = m.q(y) - psi;
                pts.emplace_back(ts, y);
            }
            if (carry_q) {
                Vec q1 = q0 + I::b(0) * hqd0;
                for (int s = 0; s < I::n_stages; ++s)
                    q1 -= (h * I::b(s + 1)) * m.f(Y[static_cast<std::size_t>(s)], f.t0 + I::c(s + 1) * h);
                x_end = Y[I::n_stages - 1];
                if (!m.from_q(q1, x_end, f.t1, &n_recover)) {
                    ++n_newton_fail;
                    return false;
                }
            }
        } else {
            using H = pulsim::pwl::HistoryState;
            const step::StageSpec s0 = I::stage(h, 0);
            factor(s0.factor_dt, f.t0);
            const Real tg = f.t0 + s0.t_off, gh = s0.ctx_h;
            const Vec qd0 = hqd0 / h;
            Vec yg = predict(tg);
            if (!newton(yg, s0.factor_dt, [&](const Vec& v) {
                    return Vec(Real{2} * (m.q(v) - q0) / gh - qd0 + m.f(v, tg));
                }))
                return false;
            Y[0] = yg;
            pts.emplace_back(tg, yg);
            if constexpr (I::n_stages == 1) return true;   // Trapezoidal
            Vec y1 = predict(f.t1);
            const Vec hist = H::trbdf2_c2() * m.q(yg) + H::trbdf2_c3() * q0;
            if (!newton(y1, s0.factor_dt, [&](const Vec& v) {
                    return Vec((H::trbdf2_c1() * m.q(v) + hist) / h + m.f(v, f.t1));
                }))
                return false;
            Y[1] = y1;
            pts.emplace_back(f.t1, y1);
        }
        return true;
    }
    [[nodiscard]] const Vec& result() const {
        if constexpr (is_esdirk) {
            if (carry_q) return x_end;
        }
        return Y[I::n_stages - 1];
    }

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
        Vec e = Vec::Zero(eq.size());   // raw: dq must be diagonal on the norm
        const Mat D = m.dq(x0);
        for (const int k : m.norm_vars) e[k] = eq[k] / D(k, k);
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
    /// With carry_q the committed end state stands for the last stage value.
    [[nodiscard]] Vec dense(Real theta) const {
        std::array<Vec, 5> y = Y;
        y[4] = result();
        return I::dense(theta, x0, y);
    }
    /// Dense output of q (the conserved variable), stage-value form.
    [[nodiscard]] Vec dense_q(Real theta) const {
        std::array<Vec, 5> qy;
        for (std::size_t s = 0; s < qy.size(); ++s) qy[s] = m.q(s + 1 == qy.size() ? result() : Y[s]);
        return I::dense(theta, Vec(m.q(x0)), qy);
    }
    /// The same with the algebraic unknowns re-solved at t0 + θh (q kept);
    /// a nonlinear q is combined in q, then x is recovered from it.
    [[nodiscard]] Vec dense_consistent(Real theta) const {
        const Real tt = t_start + theta * h_step;
        if (!m.qn) return m.consistent(dense(theta), tt);
        Vec x1 = dense(theta);
        if (!m.from_q(dense_q(theta), x1, tt)) x1.setConstant(std::numeric_limits<Real>::quiet_NaN());
        return x1;
    }
};

}  // namespace mna
