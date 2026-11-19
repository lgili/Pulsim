#pragma once

// =============================================================================
// Pulsim — saturating, cross-coupled dq flux map as a co-energy spline
// =============================================================================
//
// add-mna-native-machines group 4. The PMSM's dq constitutive law
//
//     λ_d(i_d, i_q) = ∂W'/∂i_d,    λ_q(i_d, i_q) = ∂W'/∂i_q
//
// is the gradient of ONE co-energy W'(i_d, i_q), a C² bicubic spline on
// the rectangular knot grid (kd × kq). So the incremental inductance
// matrix is the Hessian of W' — symmetric by construction (reciprocity,
// ∂λ_d/∂i_q = ∂λ_q/∂i_d: a map without it creates or destroys energy over
// a current cycle) and continuous, and λ is C¹: what Newton needs.
//
// Storage: per cell (a, b) the 16 power-basis coefficients of
//     W' = Σ_{m,n=0..3} c_mn · u^m · v^n,  u = i_d − kd[a], v = i_q − kq[b]
// at coef[((a·(nq−1)) + b)·16 + 4m + n]. The fit (least squares to both
// tables, Python `FluxMapDQ.from_tables`) produces them from a tensor
// cubic B-spline; `validate` checks the C² joins, so a caller that
// hands in arbitrary cells is refused instead of stamping a Jacobian
// that jumps.
//
// Outside the grid W' continues as its second-order Taylor expansion
// across the boundary (per axis, so also at the corners): W' stays C²
// and reciprocal everywhere, so a Newton ITERATE that wanders out is on
// a smooth law consistent with its Jacobian. It is NOT positive definite
// arbitrarily far out (the mixed terms grow with the overshoot: one
// saturating map went indefinite 10 A past a 32 A grid), so the law is
// DEFINED on the grid plus a band of kBand × the axis span on each side:
// `validate` checks positive definiteness on a sample of that whole
// region, and an ACCEPTED state outside it is refused by name
// (`PmsmMnaHistory::update_from_state`). Extend the table instead.
// (Continuing the boundary cubic made λ_d fall 14 A past the grid.)

#include "pulsim/numeric/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulsim::models {

struct FluxMapDQ {
    std::vector<Real> kd;     //!< i_d knots [A], strictly increasing
    std::vector<Real> kq;     //!< i_q knots [A], strictly increasing
    std::vector<Real> coef;   //!< 16 per cell, see the header
    std::string source;       //!< provenance label (FEA file, "synthetic", ...)
    Real fit_residual = 0;    //!< max |fit − table| / flux range of the fit

    static constexpr Real kBand = Real{0.1};   //!< validity band, fraction of the axis span

    /// Inside the grid plus the validity band.
    [[nodiscard]] bool in_band(Real id, Real iq) const noexcept {
        const Real bd = kBand * (kd.back() - kd.front());
        const Real bq = kBand * (kq.back() - kq.front());
        return id >= kd.front() - bd && id <= kd.back() + bd &&
               iq >= kq.front() - bq && iq <= kq.back() + bq;
    }

    struct Eval {
        Real W = 0;                     //!< co-energy [J per (3/2)]
        Real gd = 0, gq = 0;            //!< λ_d, λ_q [Wb]
        Real hdd = 0, hdq = 0, hqq = 0; //!< incremental inductances [H]
    };

    [[nodiscard]] static Size cell_of(const std::vector<Real>& k, Real x) noexcept {
        // Index a with k[a] <= x < k[a+1], clamped to [0, n−2].
        const auto it = std::upper_bound(k.begin() + 1, k.end() - 1, x);
        return static_cast<Size>(it - k.begin()) - 1;
    }

    /// Coefficients of P(x) = Σ q_k x^k re-expanded about x0, truncated to
    /// the quadratic (the Taylor extension past the grid).
    static void taylor2_(Real* q, Size stride, Real x0) noexcept {
        const Real q0 = q[0], q1 = q[stride], q2 = q[2 * stride], q3 = q[3 * stride];
        q[0] = ((q3 * x0 + q2) * x0 + q1) * x0 + q0;
        q[stride] = (Real{3} * q3 * x0 + Real{2} * q2) * x0 + q1;
        q[2 * stride] = Real{3} * q3 * x0 + q2;
        q[3 * stride] = Real{0};
    }

    [[nodiscard]] Eval eval_in(Size a, Size b, Real id, Real iq) const noexcept {
        std::array<Real, 16> cc;
        const Real* src = coef.data() + (a * (kq.size() - 1) + b) * 16;
        std::copy(src, src + 16, cc.begin());
        const Real* c = cc.data();
        Real u = id - kd[a], v = iq - kq[b];
        const Real wd = kd[a + 1] - kd[a], wq = kq[b + 1] - kq[b];
        const Real ud = (a == 0 && u < Real{0}) ? Real{0}
                      : (a + 2 == kd.size() && u > wd) ? wd : Real{-1};
        const Real vq = (b == 0 && v < Real{0}) ? Real{0}
                      : (b + 2 == kq.size() && v > wq) ? wq : Real{-1};
        if (ud >= Real{0}) {   // past the grid along i_d: index m, stride 4
            for (Size n = 0; n < 4; ++n) taylor2_(cc.data() + n, 4, ud);
            u -= ud;
        }
        if (vq >= Real{0}) {   // along i_q: index n, stride 1
            for (Size m = 0; m < 4; ++m) taylor2_(cc.data() + 4 * m, 1, vq);
            v -= vq;
        }
        // P_m(v) = Σ_n c_mn v^n and its first two v-derivatives.
        std::array<Real, 4> P{}, Pv{}, Pvv{};
        for (Size m = 0; m < 4; ++m) {
            const Real* r = c + 4 * m;
            P[m]   = ((r[3] * v + r[2]) * v + r[1]) * v + r[0];
            Pv[m]  = (Real{3} * r[3] * v + Real{2} * r[2]) * v + r[1];
            Pvv[m] = Real{6} * r[3] * v + Real{2} * r[2];
        }
        auto horner = [u](const std::array<Real, 4>& q) {
            return ((q[3] * u + q[2]) * u + q[1]) * u + q[0];
        };
        auto d1 = [u](const std::array<Real, 4>& q) {
            return (Real{3} * q[3] * u + Real{2} * q[2]) * u + q[1];
        };
        Eval e;
        e.W = horner(P);
        e.gd = d1(P);
        e.gq = horner(Pv);
        e.hdd = Real{6} * P[3] * u + Real{2} * P[2];
        e.hdq = d1(Pv);
        e.hqq = horner(Pvv);
        return e;
    }

    [[nodiscard]] Eval eval(Real id, Real iq) const noexcept {
        return eval_in(cell_of(kd, id), cell_of(kq, iq), id, iq);
    }

    /// λ(x0 + dx) − λ(x0) without cancellation: ∫₀¹ H(x0 + s·dx)·dx ds,
    /// split where the segment crosses a knot line (the grid edges too).
    /// Within a cell, and past the grid, H is a polynomial of degree ≤ 4
    /// along the line, so 3-point
    /// Gauss–Legendre is exact: the result IS the difference of the two
    /// gradients, but its round-off scales with |dx|, not with |λ|
    /// (the flux row multiplies it by 2/h).
    [[nodiscard]] std::array<Real, 2> grad_increment(Real id0, Real iq0,
                                                     Real dd, Real dq) const {
        if (dd == Real{0} && dq == Real{0}) return {Real{0}, Real{0}};
        std::array<Real, 64> cuts{};
        Size nc = 0;
        cuts[nc++] = Real{0};
        auto add = [&](const std::vector<Real>& k, Real x0, Real dx) {
            if (dx == Real{0}) return;
            for (Size j = 0; j < k.size(); ++j) {   // the grid edges too
                const Real s = (k[j] - x0) / dx;
                if (s > Real{0} && s < Real{1}) {
                    if (nc + 1 >= cuts.size()) return;   // more crossings: see below
                    cuts[nc++] = s;
                }
            }
        };
        add(kd, id0, dd);
        add(kq, iq0, dq);
        cuts[nc++] = Real{1};
        std::sort(cuts.begin(), cuts.begin() + static_cast<std::ptrdiff_t>(nc));
        // A step that crosses more than 62 knot lines is not an increment;
        // fall back to the plain difference (exact in exact arithmetic).
        if (nc >= cuts.size()) {
            const auto e1 = eval(id0 + dd, iq0 + dq), e0 = eval(id0, iq0);
            return {e1.gd - e0.gd, e1.gq - e0.gq};
        }
        constexpr Real g = Real{0.7745966692414834};   // √(3/5)
        constexpr std::array<Real, 3> xs{-g, Real{0}, g};
        constexpr std::array<Real, 3> ws{Real{5} / 9, Real{8} / 9, Real{5} / 9};
        Real sd = 0, sq = 0;
        for (Size p = 0; p + 1 < nc; ++p) {
            const Real s0 = cuts[p], s1 = cuts[p + 1];
            if (!(s1 > s0)) continue;
            const Real half = Real{0.5} * (s1 - s0), mid = Real{0.5} * (s1 + s0);
            const Size a = cell_of(kd, id0 + mid * dd), b = cell_of(kq, iq0 + mid * dq);
            for (Size k = 0; k < 3; ++k) {
                const Real s = mid + half * xs[k];
                const auto e = eval_in(a, b, id0 + s * dd, iq0 + s * dq);
                const Real w = ws[k] * half;
                sd += w * (e.hdd * dd + e.hdq * dq);
                sq += w * (e.hdq * dd + e.hqq * dq);
            }
        }
        return {sd, sq};
    }

    /// Shape, finiteness, C² joins between cells, and positive-definite
    /// Hessian on a 5×5 sample of every cell plus half a cell past the
    /// grid. Throws std::invalid_argument naming the map's source.
    void validate() const {
        const std::string who = std::format("flux map '{}'", source);
        auto knots = [&](const std::vector<Real>& k, const char* axis) {
            if (k.size() < 2) {
                throw std::invalid_argument(std::format(
                    "{}: needs at least 2 {} knots", who, axis));
            }
            for (Size j = 0; j < k.size(); ++j) {
                if (!std::isfinite(k[j]) || (j > 0 && !(k[j] > k[j - 1]))) {
                    throw std::invalid_argument(std::format(
                        "{}: {} knots must be finite and strictly increasing",
                        who, axis));
                }
            }
        };
        knots(kd, "i_d");
        knots(kq, "i_q");
        const Size nd = kd.size() - 1, nq = kq.size() - 1;
        if (coef.size() != nd * nq * 16) {
            throw std::invalid_argument(std::format(
                "{}: expected {} coefficients (16 per cell) but got {}", who,
                nd * nq * 16, coef.size()));
        }
        for (const Real c : coef) {
            if (!std::isfinite(c)) {
                throw std::invalid_argument(std::format("{}: non-finite coefficient", who));
            }
        }
        // Scales for the join tolerance.
        Real gscale = 0, hscale = 0;
        for (Size a = 0; a < nd; ++a)
            for (Size b = 0; b < nq; ++b) {
                const auto e = eval_in(a, b, kd[a], kq[b]);
                gscale = std::max({gscale, std::abs(e.gd), std::abs(e.gq)});
                hscale = std::max({hscale, std::abs(e.hdd), std::abs(e.hdq), std::abs(e.hqq)});
            }
        const Real span = std::max(kd.back() - kd.front(), kq.back() - kq.front());
        const Real tol_g = Real{1e-11} * (gscale + hscale * span);
        const Real tol_h = Real{1e-11} * (hscale + gscale / span);
        auto join = [&](Size a0, Size b0, Size a1, Size b1, Real id, Real iq) {
            const auto l = eval_in(a0, b0, id, iq), r = eval_in(a1, b1, id, iq);
            if (std::abs(l.gd - r.gd) > tol_g || std::abs(l.gq - r.gq) > tol_g ||
                std::abs(l.hdd - r.hdd) > tol_h || std::abs(l.hdq - r.hdq) > tol_h ||
                std::abs(l.hqq - r.hqq) > tol_h) {
                throw std::invalid_argument(std::format(
                    "{}: cells do not join C² at (i_d, i_q) = ({}, {}); the "
                    "Newton Jacobian would jump there", who, id, iq));
            }
        };
        constexpr std::array<Real, 3> fr{Real{0}, Real{0.5}, Real{1}};
        for (Size a = 0; a < nd; ++a)
            for (Size b = 0; b < nq; ++b)
                for (const Real f : fr) {
                    const Real iq = kq[b] + f * (kq[b + 1] - kq[b]);
                    const Real id = kd[a] + f * (kd[a + 1] - kd[a]);
                    if (a + 1 < nd) join(a, b, a + 1, b, kd[a + 1], iq);
                    if (b + 1 < nq) join(a, b, a, b + 1, id, kq[b + 1]);
                }
        // Positive definite on the grid and the validity band: 4 samples
        // per cell, 8 across each band.
        auto pd = [&](Real id, Real iq) {
            const auto e = eval(id, iq);
            if (!(e.hdd > Real{0}) || !(e.hdd * e.hqq - e.hdq * e.hdq > Real{0})) {
                throw std::invalid_argument(std::format(
                    "{}: incremental inductance not positive definite at "
                    "(i_d, i_q) = ({}, {}): L_dd = {}, L_dq = {}, L_qq = {} "
                    "(checked on the grid and a band of {} x the span past it)",
                    who, id, iq, e.hdd, e.hdq, e.hqq, kBand));
            }
        };
        auto samples = [](const std::vector<Real>& k) {
            const Real band = kBand * (k.back() - k.front());
            std::vector<Real> s;
            for (Size f = 0; f < 8; ++f) s.push_back(k.front() - band + Real(f) / 8 * band);
            for (Size j = 0; j + 1 < k.size(); ++j)
                for (Size f = 0; f < 4; ++f) s.push_back(k[j] + Real(f) / 4 * (k[j + 1] - k[j]));
            for (Size f = 0; f <= 8; ++f) s.push_back(k.back() + Real(f) / 8 * band);
            return s;
        };
        const auto sd = samples(kd), sq = samples(kq);
        for (const Real id : sd)
            for (const Real iq : sq) pd(id, iq);
    }
};

}  // namespace pulsim::models
