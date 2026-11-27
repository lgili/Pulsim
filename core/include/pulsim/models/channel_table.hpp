#pragma once

// =============================================================================
// Pulsim — tabulated MOSFET channel I_D(V_GS, V_DS) (add-power-mosfet-
// datasheet-device task 5.1)
// =============================================================================
//
// A rectilinear grid V_GS[i] × V_DS[j] (V_DS[0] = 0) of drain currents
// I[i][j] ≥ 0, non-decreasing in both arguments, with the V_DS = 0 column
// identically zero (no current without a voltage). Interpolation: bicubic
// Hermite, per-axis tangents from the monotone (Fritsch–Butland / PCHIP)
// rule along each grid line, zero twist, so the law is C¹ everywhere.
// Outside the grid it continues to first order from the nearest grid point
// (value + gradient + the mixed term in a corner), which keeps it C¹.
//
// Third quadrant by terminal swap: with V_DS < 0 the drain is the source,
//     i(V_GS, V_DS) = −I(V_GS − V_DS, −V_DS),
// C¹ at V_DS = 0 because the V_DS = 0 column is zero (so is its V_GS
// tangent): both sides give i = 0, ∂i/∂V_GS = 0, ∂i/∂V_DS = I_d(V_GS, 0).
//
// Refusals (pulsim::NetlistError, Python `pulsim.NetlistError`) name the
// offending axis point.

#include "pulsim/numeric/types.hpp"
#include "pulsim/sim/error.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <string>
#include <vector>

namespace pulsim::models {

class ChannelTable {
public:
    /// The current and its two partials at one bias point.
    struct Eval {
        Real i = 0, di_dvgs = 0, di_dvds = 0;
    };

    /// `I[i][j]` = I_D at (v_gs[i], v_ds[j]) [A].
    ChannelTable(std::vector<Real> v_gs, std::vector<Real> v_ds,
                 std::vector<std::vector<Real>> I)
        : x_(std::move(v_gs)), y_(std::move(v_ds)), f_(std::move(I)) {
        validate_();
        tangents_();
    }

    [[nodiscard]] const std::vector<Real>& v_gs() const noexcept { return x_; }
    [[nodiscard]] const std::vector<Real>& v_ds() const noexcept { return y_; }
    [[nodiscard]] const std::vector<std::vector<Real>>& current_grid() const noexcept { return f_; }

    /// The first-quadrant law I(V_GS, V_DS) for V_DS ≥ 0 (also evaluated,
    /// by its continuation, below 0 — `current` never does that).
    [[nodiscard]] Eval forward(Real vgs, Real vds) const noexcept {
        const Real xc = std::clamp(vgs, x_.front(), x_.back());
        const Real yc = std::clamp(vds, y_.front(), y_.back());
        const Patch p = patch_(xc, yc);
        const Real dx = vgs - xc, dy = vds - yc;
        return {p.f + dx * p.fx + dy * p.fy + dx * dy * p.fxy,
                p.fx + dy * p.fxy, p.fy + dx * p.fxy};
    }
    /// The terminal-swap branch: −I(V_GS − V_DS, −V_DS) and its partials.
    [[nodiscard]] Eval reverse(Real vgs, Real vds) const noexcept {
        const Eval e = forward(vgs - vds, -vds);
        return {-e.i, -e.di_dvgs, e.di_dvgs + e.di_dvds};
    }
    /// The drain current at (V_GS, V_DS), either sign of V_DS.
    [[nodiscard]] Eval current(Real vgs, Real vds) const noexcept {
        return vds >= Real{0} ? forward(vgs, vds) : reverse(vgs, vds);
    }

private:
    std::vector<Real> x_, y_;
    std::vector<std::vector<Real>> f_, fx_, fy_;

    struct Patch { Real f, fx, fy, fxy; };

    [[noreturn]] static void refuse_(const std::string& why) {
        throw NetlistError("ChannelTable", "", "ChannelTable: " + why);
    }

    void validate_() const {
        auto axis = [](const std::vector<Real>& a, const char* nm) {
            if (a.size() < 2) {
                refuse_(std::format("the {} axis needs at least 2 points (got {})", nm, a.size()));
            }
            for (std::size_t k = 0; k < a.size(); ++k) {
                if (!std::isfinite(a[k]) || (k > 0 && !(a[k] > a[k - 1]))) {
                    refuse_(std::format("the {} axis must be finite and strictly increasing; "
                                        "point {} = {}{}", nm, k, a[k],
                                        k > 0 ? std::format(" follows {}", a[k - 1]) : ""));
                }
            }
        };
        axis(x_, "V_GS");
        axis(y_, "V_DS");
        if (y_.front() != Real{0}) {
            refuse_(std::format("the V_DS axis must start at 0 (got {}): the third "
                                "quadrant is the terminal swap of the first", y_.front()));
        }
        if (f_.size() != x_.size()) {
            refuse_(std::format("{} rows of current for {} V_GS points", f_.size(), x_.size()));
        }
        for (std::size_t i = 0; i < x_.size(); ++i) {
            if (f_[i].size() != y_.size()) {
                refuse_(std::format("row V_GS = {} has {} currents for {} V_DS points",
                                    x_[i], f_[i].size(), y_.size()));
            }
            for (std::size_t j = 0; j < y_.size(); ++j) {
                const Real v = f_[i][j];
                if (!std::isfinite(v)) {
                    refuse_(std::format("I_D(V_GS = {}, V_DS = {}) = {} is not finite", x_[i], y_[j], v));
                }
                if (j == 0 && v != Real{0}) {
                    refuse_(std::format("I_D(V_GS = {}, V_DS = 0) = {} must be 0 (no current "
                                        "without a drain voltage)", x_[i], v));
                }
                if (j > 0 && v < f_[i][j - 1]) {
                    refuse_(std::format("not monotone in V_DS at V_GS = {}: I_D(V_DS = {}) = {} is "
                                        "below I_D(V_DS = {}) = {} (point [{}][{}])",
                                        x_[i], y_[j], v, y_[j - 1], f_[i][j - 1], i, j));
                }
                if (i > 0 && v < f_[i - 1][j]) {
                    refuse_(std::format("not monotone in V_GS at V_DS = {}: I_D(V_GS = {}) = {} is "
                                        "below I_D(V_GS = {}) = {} (point [{}][{}])",
                                        y_[j], x_[i], v, x_[i - 1], f_[i - 1][j], i, j));
                }
            }
        }
    }

    /// Monotone (Fritsch–Butland) tangents of samples `v` on knots `t`.
    static std::vector<Real> pchip_(const std::vector<Real>& t, const std::vector<Real>& v) {
        const std::size_t n = t.size();
        std::vector<Real> d(n, Real{0}), s(n - 1);
        for (std::size_t k = 0; k + 1 < n; ++k) s[k] = (v[k + 1] - v[k]) / (t[k + 1] - t[k]);
        if (n == 2) { d[0] = d[1] = s[0]; return d; }
        for (std::size_t k = 1; k + 1 < n; ++k) {
            if (s[k - 1] * s[k] > Real{0}) {
                const Real h0 = t[k] - t[k - 1], h1 = t[k + 1] - t[k];
                const Real w1 = Real{2} * h1 + h0, w2 = h1 + Real{2} * h0;
                d[k] = (w1 + w2) / (w1 / s[k - 1] + w2 / s[k]);
            }
        }
        auto end = [](Real h0, Real h1, Real s0, Real s1) {
            Real e = ((Real{2} * h0 + h1) * s0 - h0 * s1) / (h0 + h1);
            if (e * s0 <= Real{0}) e = Real{0};
            else if (s0 * s1 <= Real{0} && std::abs(e) > std::abs(Real{3} * s0)) e = Real{3} * s0;
            return e;
        };
        d[0] = end(t[1] - t[0], t[2] - t[1], s[0], s[1]);
        d[n - 1] = end(t[n - 1] - t[n - 2], t[n - 2] - t[n - 3], s[n - 2], s[n - 3]);
        return d;
    }

    void tangents_() {
        const std::size_t nx = x_.size(), ny = y_.size();
        fx_.assign(nx, std::vector<Real>(ny));
        fy_.assign(nx, std::vector<Real>(ny));
        for (std::size_t i = 0; i < nx; ++i) fy_[i] = pchip_(y_, f_[i]);
        std::vector<Real> col(nx);
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) col[i] = f_[i][j];
            const auto d = pchip_(x_, col);
            for (std::size_t i = 0; i < nx; ++i) fx_[i][j] = d[i];
        }
    }

    static std::size_t cell_(const std::vector<Real>& a, Real v) noexcept {
        const auto it = std::upper_bound(a.begin(), a.end(), v);
        const auto j = static_cast<std::size_t>(it - a.begin());
        return std::min(j == 0 ? std::size_t{0} : j - 1, a.size() - 2);
    }

    /// Bicubic Hermite (zero twist) value and partials at a point of the grid box.
    [[nodiscard]] Patch patch_(Real x, Real y) const noexcept {
        const std::size_t i = cell_(x_, x), j = cell_(y_, y);
        const Real hx = x_[i + 1] - x_[i], hy = y_[j + 1] - y_[j];
        const Real t = (x - x_[i]) / hx, u = (y - y_[j]) / hy;
        // Hermite basis: H0, H1 (values at the ends), G0, G1 (tangents), and d/ds
        auto basis = [](Real s, Real b[4], Real db[4]) {
            const Real s2 = s * s, s3 = s2 * s;
            b[0] = Real{2} * s3 - Real{3} * s2 + Real{1};
            b[1] = Real{3} * s2 - Real{2} * s3;
            b[2] = s3 - Real{2} * s2 + s;
            b[3] = s3 - s2;
            db[0] = Real{6} * s2 - Real{6} * s;
            db[1] = Real{6} * s - Real{6} * s2;
            db[2] = Real{3} * s2 - Real{4} * s + Real{1};
            db[3] = Real{3} * s2 - Real{2} * s;
        };
        Real bt[4], dbt[4], bu[4], dbu[4];
        basis(t, bt, dbt);
        basis(u, bu, dbu);
        Patch p{0, 0, 0, 0};
        for (int a = 0; a < 2; ++a) {
            for (int c = 0; c < 2; ++c) {
                const Real F = f_[i + a][j + c];
                const Real Fx = fx_[i + a][j + c] * hx;   // tangents in cell units
                const Real Fy = fy_[i + a][j + c] * hy;
                const Real Ht = bt[a], Gt = bt[2 + a], dHt = dbt[a], dGt = dbt[2 + a];
                const Real Hu = bu[c], Gu = bu[2 + c], dHu = dbu[c], dGu = dbu[2 + c];
                p.f   += Ht * Hu * F + Gt * Hu * Fx + Ht * Gu * Fy;
                p.fx  += dHt * Hu * F + dGt * Hu * Fx + dHt * Gu * Fy;
                p.fy  += Ht * dHu * F + Gt * dHu * Fx + Ht * dGu * Fy;
                p.fxy += dHt * dHu * F + dGt * dHu * Fx + dHt * dGu * Fy;
            }
        }
        p.fx /= hx;
        p.fy /= hy;
        p.fxy /= hx * hy;
        return p;
    }
};

}  // namespace pulsim::models
