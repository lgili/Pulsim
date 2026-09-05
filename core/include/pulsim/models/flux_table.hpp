#pragma once

// =============================================================================
// Pulsim — Phase 4 C.4: monotone λ(i) lookup table (a flux law)
// =============================================================================
//
// A flux device is v = dλ/dt with λ = λ(i). The saturable inductor
// carries that law in closed form (an arctangent). A CORE — geometry,
// gap, a B(H) curve from a datasheet or from the Jiles-Atherton
// anhysteretic — does not have one, so it is tabulated:
//
//     knots (i_k, λ_k), k = 0..K-1,  i_0 = 0, λ_0 = 0, strictly
//     increasing in both,  extended as an ODD function to i < 0.
//
// Between knots: monotone piecewise-cubic Hermite (Fritsch–Carlson
// PCHIP). Its interpolant is guaranteed monotone on monotone data,
// which is what makes L = dλ/di > 0 everywhere — an interpolating
// spline can overshoot and hand Newton a negative inductance. The
// derivative is continuous (C¹ λ, C⁰ L); the Jacobian entry is
// exact for the interpolant, not a finite difference.
//
// Beyond the last knot: linear at the last slope, i.e. a constant
// residual inductance — a saturated core is an air-core inductor, and
// extrapolating the cubic instead would eventually turn λ around.
//
// Accuracy, measured against the analytic atan law it must be able to
// replace (L0 = 1 mH, I_sat = 5 A, L_res = 50 µH, 0 ≤ i ≤ 10 I_sat):
//
//     knots   spacing        max rel err λ     max rel err L
//       32    uniform            4.4e-2            4.4e-2
//       32    sqrt-spaced        1.8e-4            6.6e-3
//       64    uniform            1.4e-2            1.4e-2
//       64    sqrt-spaced        2.1e-5            1.6e-3
//
// so knots must be DENSE NEAR ZERO where L is largest and λ curves
// most; uniform spacing wastes most of the table in the flat tail.
// Callers that generate a table from a core law get that for free by
// sweeping λ uniformly (i then lands densely below the knee); callers
// tabulating an i-domain law should use sqrt spacing.
//
// Refuses by name: fewer than 3 knots, a non-zero origin, a knot pair
// that is not strictly increasing in both i and λ (a non-monotone
// table is a negative inductance, and the stamp would converge on it).

#include "pulsim/numeric/types.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulsim::models {

class FluxTable {
public:
    FluxTable() = default;

    /// Build from knots for i ≥ 0. `what` names the device in
    /// refusals.
    FluxTable(std::vector<Real> i_knots, std::vector<Real> lambda_knots,
              const std::string& what = "FluxTable")
        : i_(std::move(i_knots)), lam_(std::move(lambda_knots)) {
        validate_(what);
        compute_slopes_();
    }

    [[nodiscard]] bool empty() const noexcept { return i_.empty(); }
    [[nodiscard]] Size size() const noexcept { return i_.size(); }
    [[nodiscard]] const std::vector<Real>& i_knots() const noexcept { return i_; }
    [[nodiscard]] const std::vector<Real>& lambda_knots() const noexcept { return lam_; }
    /// dλ/di at the origin — the unsaturated inductance.
    [[nodiscard]] Real L_0() const noexcept { return m_.empty() ? Real{0} : m_[0]; }
    /// Slope beyond the last knot — the residual (air) inductance.
    [[nodiscard]] Real L_residual() const noexcept { return m_.empty() ? Real{0} : m_.back(); }
    [[nodiscard]] Real i_max() const noexcept { return i_.empty() ? Real{0} : i_.back(); }

    /// λ(i), odd in i.
    [[nodiscard]] Real flux(Real i) const noexcept {
        const Real a = std::abs(i);
        const Real s = i < Real{0} ? Real{-1} : Real{1};
        if (a >= i_.back()) {
            return s * (lam_.back() + m_.back() * (a - i_.back()));
        }
        const Size j = segment_(a);
        const Real h = i_[j + 1] - i_[j];
        const Real t = (a - i_[j]) / h;
        const Real t2 = t * t, t3 = t2 * t;
        const Real h00 = 2 * t3 - 3 * t2 + 1;
        const Real h10 = t3 - 2 * t2 + t;
        const Real h01 = -2 * t3 + 3 * t2;
        const Real h11 = t3 - t2;
        return s * (h00 * lam_[j] + h10 * h * m_[j]
                    + h01 * lam_[j + 1] + h11 * h * m_[j + 1]);
    }

    /// L(i) = dλ/di, even in i, > 0 everywhere.
    [[nodiscard]] Real inductance(Real i) const noexcept {
        const Real a = std::abs(i);
        if (a >= i_.back()) return m_.back();
        const Size j = segment_(a);
        const Real h = i_[j + 1] - i_[j];
        const Real t = (a - i_[j]) / h;
        const Real t2 = t * t;
        return (6 * t2 - 6 * t) * lam_[j] / h
               + (3 * t2 - 4 * t + 1) * m_[j]
               + (-6 * t2 + 6 * t) * lam_[j + 1] / h
               + (3 * t2 - 2 * t) * m_[j + 1];
    }

private:
    std::vector<Real> i_, lam_, m_;

    [[nodiscard]] Size segment_(Real a) const noexcept {
        // First knot strictly greater than a, minus one. a < i_.back().
        const auto it = std::upper_bound(i_.begin(), i_.end(), a);
        const auto j = static_cast<Size>(it - i_.begin());
        return j == 0 ? Size{0} : j - 1;
    }

    void validate_(const std::string& what) const {
        if (i_.size() != lam_.size()) {
            throw std::invalid_argument(std::format(
                "{}: {} current knots but {} flux knots.",
                what, i_.size(), lam_.size()));
        }
        if (i_.size() < 3) {
            throw std::invalid_argument(std::format(
                "{}: a flux table needs at least 3 knots (got {}) — two "
                "define a line, which is a linear inductor; use "
                "add_inductor for that.", what, i_.size()));
        }
        if (i_[0] != Real{0} || lam_[0] != Real{0}) {
            throw std::invalid_argument(std::format(
                "{}: the first knot must be the origin (0, 0) — the table "
                "is extended as an odd function, λ(−i) = −λ(i), and a "
                "non-zero λ(0) would be a flux with no current to hold "
                "it (a permanent magnet, which this is not). Got ({}, {}).",
                what, i_[0], lam_[0]));
        }
        for (Size k = 1; k < i_.size(); ++k) {
            if (!(i_[k] > i_[k - 1]) || !std::isfinite(i_[k])) {
                throw std::invalid_argument(std::format(
                    "{}: current knots must be strictly increasing and "
                    "finite; knot {} = {} follows {}.",
                    what, k, i_[k], i_[k - 1]));
            }
            if (!(lam_[k] > lam_[k - 1]) || !std::isfinite(lam_[k])) {
                throw std::invalid_argument(std::format(
                    "{}: flux knots must be strictly increasing — λ that "
                    "falls or holds as i rises is a zero or NEGATIVE "
                    "inductance, and Newton would converge on it. Knot "
                    "{}: λ = {} follows λ = {} (i = {} → {}).",
                    what, k, lam_[k], lam_[k - 1], i_[k - 1], i_[k]));
            }
        }
    }

    /// Fritsch–Carlson shape-preserving slopes.
    void compute_slopes_() {
        const Size n = i_.size();
        std::vector<Real> h(n - 1), d(n - 1);
        for (Size k = 0; k + 1 < n; ++k) {
            h[k] = i_[k + 1] - i_[k];
            d[k] = (lam_[k + 1] - lam_[k]) / h[k];
        }
        m_.assign(n, Real{0});
        for (Size k = 1; k + 1 < n; ++k) {
            // Secants are all positive here (validated), so the
            // weighted harmonic mean applies at every interior knot.
            const Real w1 = 2 * h[k] + h[k - 1];
            const Real w2 = h[k] + 2 * h[k - 1];
            m_[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k]);
        }
        auto end_slope = [](Real h0, Real h1, Real d0, Real d1) {
            Real s = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
            if (s <= Real{0}) s = Real{0};                 // sign of d0
            else if (s > 3 * d0) s = 3 * d0;               // shape guard
            return s;
        };
        m_[0]     = end_slope(h[0], h[1], d[0], d[1]);
        m_[n - 1] = end_slope(h[n - 2], h[n - 3], d[n - 2], d[n - 3]);
        // The origin slope is L_0 and must be positive; a zero from
        // the guard above would be a shorted inductor at rest.
        if (!(m_[0] > Real{0})) m_[0] = d[0];
        if (!(m_[n - 1] > Real{0})) m_[n - 1] = d[n - 2];
    }
};

}  // namespace pulsim::models
