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
//     knots (i_k, λ_k, L_k), k = 0..K-1,  i_0 = λ_0 = 0, i and λ
//     strictly increasing, L_k = dλ/di at the knot, > 0;
//     extended as an ODD function to i < 0.
//
// TRIPLES, NOT PAIRS. Between knots the interpolant is the cubic
// Hermite with the stored L_k as tangents, so L(i) = dλ/di is the
// EXACT derivative of the same expression λ(i) evaluates — which is
// what the Newton stamp needs (residual from λ, Jacobian from L,
// one function). Every generator knows the slope in closed form: the
// analytic law has L(i), and a core has L = N²Ae/(le·dH_c/dB + lg/μ₀)
// at each swept B. Measured against the analytic atan law at 64
// knots: PCHIP-ESTIMATED slopes give 1.6e-3 relative error in L
// (O(h²), and 7.5e-3 on a sharp gapped core); EXACT slopes give
// 2e-5 (1.2e-4 on the same core) — fifty to a hundred times better
// for the same table. That is why the pair-only constructor below
// is the fallback for raw digitised data, and says so.
//
// Monotonicity: with exact slopes on a concave law the Fritsch–
// Carlson condition α² + β² ≤ 9 (α = L_k/d_k, β = L_{k+1}/d_k, d_k
// the secant) holds on every segment (measured max 2.44), which
// guarantees λ monotone and L > 0 between knots. It is CHECKED at
// construction and a violating tangent is clamped radially onto the
// circle — a user table digitised with a kink can otherwise hand
// Newton a negative inductance. For pairs, Fritsch–Carlson slope
// estimation is monotone by construction.
//
// Origin: L_0 is stored exactly (a generator always knows it); it is
// NOT estimated from the first two knots, where a 3-point end-slope
// formula is 4e-4 to 1.4e-3 high.
//
// Tail: linear beyond the last knot at `L_tail`, the exact residual
// (air) inductance the generator supplies — the saturated core is an
// air-core inductor. The generator's job is to sweep far enough that
// the last knot's own L is already within a percent of L_tail, so
// the Jacobian has no jump there; extrapolating at an ESTIMATED end
// slope left the saturated inductance 27-33 % high forever.
//
// Refuses by name: fewer than 3 knots, a non-zero origin, a knot
// pair that is not strictly increasing in both i and λ, a non-
// positive slope (a non-monotone table is a negative inductance, and
// the stamp would converge on it).

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

    /// From TRIPLES: knots for i ≥ 0 with the exact slope at each,
    /// and the exact residual slope for the tail. The preferred
    /// constructor — every generator has the slopes.
    FluxTable(std::vector<Real> i_knots, std::vector<Real> lambda_knots,
              std::vector<Real> slope_knots, Real L_tail,
              const std::string& what = "FluxTable")
        : i_(std::move(i_knots)), lam_(std::move(lambda_knots)),
          m_(std::move(slope_knots)), L_tail_(L_tail) {
        validate_(what, /*with_slopes=*/true);
        clamp_to_monotone_region_(what);
    }

    /// From PAIRS: slopes ESTIMATED by Fritsch–Carlson (monotone by
    /// construction, O(h²) in L — about 1.6e-3 at 64 well-placed
    /// knots). For raw digitised data only; pass `L_0` and `L_tail`
    /// when you know them, since the end-slope estimates are the
    /// worst part of the fit.
    FluxTable(std::vector<Real> i_knots, std::vector<Real> lambda_knots,
              const std::string& what = "FluxTable",
              Real L_0 = Real{0}, Real L_tail = Real{0})
        : i_(std::move(i_knots)), lam_(std::move(lambda_knots)) {
        validate_(what, /*with_slopes=*/false);
        estimate_slopes_();
        if (L_0 > Real{0}) m_.front() = L_0;
        if (L_tail > Real{0}) { m_.back() = L_tail; L_tail_ = L_tail; }
        else L_tail_ = m_.back();
        clamp_to_monotone_region_(what);
    }

    [[nodiscard]] bool empty() const noexcept { return i_.empty(); }
    [[nodiscard]] Size size() const noexcept { return i_.size(); }
    [[nodiscard]] const std::vector<Real>& i_knots() const noexcept { return i_; }
    [[nodiscard]] const std::vector<Real>& lambda_knots() const noexcept { return lam_; }
    [[nodiscard]] const std::vector<Real>& slope_knots() const noexcept { return m_; }
    /// dλ/di at the origin — the unsaturated inductance.
    [[nodiscard]] Real L_0() const noexcept { return m_.empty() ? Real{0} : m_.front(); }
    /// Slope beyond the last knot — the residual (air) inductance.
    [[nodiscard]] Real L_residual() const noexcept { return L_tail_; }
    [[nodiscard]] Real i_max() const noexcept { return i_.empty() ? Real{0} : i_.back(); }

    /// λ(i), odd in i.
    [[nodiscard]] Real flux(Real i) const noexcept {
        const Real a = std::abs(i);
        const Real s = i < Real{0} ? Real{-1} : Real{1};
        if (a >= i_.back()) {
            return s * (lam_.back() + L_tail_ * (a - i_.back()));
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

    /// L(i) = dλ/di, even in i, > 0 everywhere — the exact derivative
    /// of `flux()`.
    [[nodiscard]] Real inductance(Real i) const noexcept {
        const Real a = std::abs(i);
        if (a >= i_.back()) return L_tail_;
        const Size j = segment_(a);
        const Real h = i_[j + 1] - i_[j];
        const Real t = (a - i_[j]) / h;
        const Real t2 = t * t;
        return (6 * t2 - 6 * t) * lam_[j] / h
               + (3 * t2 - 4 * t + 1) * m_[j]
               + (-6 * t2 + 6 * t) * lam_[j + 1] / h
               + (3 * t2 - 2 * t) * m_[j + 1];
    }

    /// Largest α² + β² seen at construction (≤ 9 after clamping);
    /// > 9 before clamping means a tangent was pulled in.
    [[nodiscard]] Real max_fritsch_carlson_radius2() const noexcept {
        return fc_max_;
    }
    [[nodiscard]] Size clamped_tangents() const noexcept { return n_clamped_; }

private:
    std::vector<Real> i_, lam_, m_;
    Real L_tail_ = Real{0};
    Real fc_max_ = Real{0};
    Size n_clamped_ = 0;

    [[nodiscard]] Size segment_(Real a) const noexcept {
        const auto it = std::upper_bound(i_.begin(), i_.end(), a);
        const auto j = static_cast<Size>(it - i_.begin());
        return j == 0 ? Size{0} : j - 1;
    }

    void validate_(const std::string& what, bool with_slopes) const {
        if (i_.size() != lam_.size() || (with_slopes && m_.size() != i_.size())) {
            throw std::invalid_argument(std::format(
                "{}: {} current knots but {} flux knots{}.",
                what, i_.size(), lam_.size(),
                with_slopes ? std::format(" and {} slopes", m_.size()) : ""));
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
        if (with_slopes) {
            for (Size k = 0; k < m_.size(); ++k) {
                if (!(m_[k] > Real{0}) || !std::isfinite(m_[k])) {
                    throw std::invalid_argument(std::format(
                        "{}: slope (inductance) at knot {} is {} — it must "
                        "be positive and finite.", what, k, m_[k]));
                }
            }
            if (!(L_tail_ > Real{0}) || !std::isfinite(L_tail_)) {
                throw std::invalid_argument(std::format(
                    "{}: the residual (tail) inductance is {} — it must be "
                    "positive and finite; a saturated core is an air-core "
                    "inductor, not a short.", what, L_tail_));
            }
        }
    }

    /// Fritsch–Carlson shape-preserving slope estimates (pairs only).
    void estimate_slopes_() {
        const Size n = i_.size();
        std::vector<Real> h(n - 1), d(n - 1);
        for (Size k = 0; k + 1 < n; ++k) {
            h[k] = i_[k + 1] - i_[k];
            d[k] = (lam_[k + 1] - lam_[k]) / h[k];
        }
        m_.assign(n, Real{0});
        for (Size k = 1; k + 1 < n; ++k) {
            const Real w1 = 2 * h[k] + h[k - 1];
            const Real w2 = h[k] + 2 * h[k - 1];
            m_[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k]);
        }
        auto end_slope = [](Real h0, Real h1, Real d0, Real d1) {
            Real s = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
            if (s <= Real{0}) s = d0;
            else if (s > 3 * d0) s = 3 * d0;
            return s;
        };
        m_[0]     = end_slope(h[0], h[1], d[0], d[1]);
        m_[n - 1] = end_slope(h[n - 2], h[n - 3], d[n - 2], d[n - 3]);
    }

    /// Pull any tangent pair outside the Fritsch–Carlson circle
    /// (α² + β² ≤ 9) radially onto it. Keeps λ monotone and L > 0
    /// between knots whatever the caller supplied.
    void clamp_to_monotone_region_(const std::string& /*what*/) {
        const Size n = i_.size();
        fc_max_ = Real{0};
        n_clamped_ = 0;
        for (Size k = 0; k + 1 < n; ++k) {
            const Real d = (lam_[k + 1] - lam_[k]) / (i_[k + 1] - i_[k]);
            const Real alpha = m_[k] / d, beta = m_[k + 1] / d;
            const Real r2 = alpha * alpha + beta * beta;
            fc_max_ = std::max(fc_max_, r2);
            if (r2 > Real{9}) {
                const Real tau = Real{3} / std::sqrt(r2);
                m_[k]     = tau * alpha * d;
                m_[k + 1] = tau * beta * d;
                ++n_clamped_;
            }
        }
    }
};

}  // namespace pulsim::models
