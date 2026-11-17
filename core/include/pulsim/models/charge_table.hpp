#pragma once

// =============================================================================
// Pulsim — monotone Q(v) lookup table (a charge law)
// =============================================================================
//
// Sibling of `FluxTable` (models/flux_table.hpp) for a capacitor: a
// datasheet C_rss(V_DS) or C_oss(V_DS) curve falls two to three decades
// and has no closed form, so it is tabulated as
//
//     knots (v_k, Q_k, C_k), v strictly increasing, one knot at v = 0
//     with Q = 0, C_k = dQ/dv at the knot, > 0.
//
// Between knots Q is the cubic Hermite with the stored C_k as tangents,
// and C(v) is the EXACT derivative of that same cubic — the Newton stamp
// takes its residual from Q and its Jacobian from C, so the two must be
// one function (owner note: a Newton law must be smooth to ~1e-12 and its
// Jacobian the derivative of what the residual evaluates).
//
// Differences from FluxTable, all deliberate:
//   * NOT odd: a junction is not symmetric, so the table covers negative
//     v explicitly (the v = 0 knot can sit anywhere in the list).
//   * Tails continue linearly at the END-KNOT slopes, so C(v) is
//     continuous everywhere (Q is C¹). A separate tail slope would put a
//     jump in the Jacobian at the last knot.
//   * Q is evaluated as Q_j + Δ(v − v_j), never as a sum of four
//     O(Q) Hermite terms: at 400 V a 1 pV step moves Q by ~9 ulps, and
//     the four-term form carries several ulps of noise — the picovolt
//     monotonicity test fails on it.
//
// Monotonicity: the Fritsch–Carlson condition α² + β² ≤ 9 is checked on
// every segment and a violating tangent pair is pulled radially onto the
// circle, as in FluxTable, so Q is monotone and C ≥ 0 between knots
// whatever the caller supplied. `clamped_tangents()` reports how many.
//
// From PAIRS (v, C) — a digitised curve: Q_k is the exact integral of
// the Fritsch–Carlson monotone cubic interpolant of C through the pairs,
// and the C_k are the data. Between knots the table's C(v) is then the
// quadratic that matches both end values and the segment's charge;
// `pulsim.datasheet.fit_charge_table` adds knots until it reproduces
// every input point within its tolerance and reports the residual.

#include "pulsim/numeric/types.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulsim::models {

class ChargeTable {
public:
    ChargeTable() = default;

    /// From TRIPLES: knots with the exact slope (capacitance) at each.
    ChargeTable(std::vector<Real> v_knots, std::vector<Real> q_knots,
                std::vector<Real> c_knots,
                const std::string& what = "ChargeTable")
        : v_(std::move(v_knots)), q_(std::move(q_knots)),
          c_(std::move(c_knots)) {
        validate_(what);
        clamp_to_monotone_region_();
    }

    /// From PAIRS (v, C): Q_k by exact integration of the monotone
    /// cubic interpolant of C. For digitised capacitance curves.
    [[nodiscard]] static ChargeTable from_capacitance(
        const std::vector<Real>& v, const std::vector<Real>& c,
        const std::string& what = "ChargeTable") {
        if (v.size() != c.size()) {
            throw std::invalid_argument(std::format(
                "{}: {} voltage knots but {} capacitance values.",
                what, v.size(), c.size()));
        }
        // Validate the pairs before integrating (same messages as the
        // triple form); a dummy Q keeps validate_ happy.
        ChargeTable probe;
        probe.v_ = v;
        probe.c_ = c;
        probe.q_.assign(v.size(), Real{0});
        probe.validate_(what, /*check_q=*/false);

        const Size n = v.size();
        // Fritsch–Carlson slopes of C(v) (shape-preserving).
        std::vector<Real> h(n - 1), d(n - 1), s(n, Real{0});
        for (Size k = 0; k + 1 < n; ++k) {
            h[k] = v[k + 1] - v[k];
            d[k] = (c[k + 1] - c[k]) / h[k];
        }
        for (Size k = 1; k + 1 < n; ++k) {
            if (d[k - 1] * d[k] > Real{0}) {
                const Real w1 = 2 * h[k] + h[k - 1];
                const Real w2 = h[k] + 2 * h[k - 1];
                s[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k]);
            }
        }
        s[0] = d[0];
        s[n - 1] = d[n - 2];
        // Integral of the Hermite cubic of C over each segment:
        //   h (C_k + C_{k+1}) / 2 + h² (s_k − s_{k+1}) / 12.
        std::vector<Real> seg(n - 1);
        for (Size k = 0; k + 1 < n; ++k) {
            seg[k] = h[k] * (c[k] + c[k + 1]) / 2
                     + h[k] * h[k] * (s[k] - s[k + 1]) / 12;
        }
        std::vector<Real> q(n, Real{0});
        const Size z = probe.zero_index_();
        for (Size k = z; k + 1 < n; ++k) q[k + 1] = q[k] + seg[k];
        for (Size k = z; k > 0; --k) q[k - 1] = q[k] - seg[k - 1];
        return ChargeTable(v, std::move(q), c, what);
    }

    [[nodiscard]] Size size() const noexcept { return v_.size(); }
    [[nodiscard]] const std::vector<Real>& v_knots() const noexcept { return v_; }
    [[nodiscard]] const std::vector<Real>& q_knots() const noexcept { return q_; }
    [[nodiscard]] const std::vector<Real>& c_knots() const noexcept { return c_; }
    [[nodiscard]] Size clamped_tangents() const noexcept { return n_clamped_; }
    [[nodiscard]] Real max_fritsch_carlson_radius2() const noexcept { return fc_max_; }

    /// Q(v) [C].
    [[nodiscard]] Real charge(Real v) const noexcept {
        if (v <= v_.front()) return q_.front() + c_.front() * (v - v_.front());
        if (v >= v_.back())  return q_.back() + c_.back() * (v - v_.back());
        const Size j = segment_(v);
        const Real h = v_[j + 1] - v_[j];
        const Real x = v - v_[j];
        const Real t = x / h;
        const Real d = (q_[j + 1] - q_[j]) / h;
        const Real c2 = 3 * d - 2 * c_[j] - c_[j + 1];
        const Real c3 = c_[j] + c_[j + 1] - 2 * d;
        return q_[j] + x * (c_[j] + t * (c2 + t * c3));
    }

    /// C(v) = dQ/dv [F] — the exact derivative of `charge()`.
    [[nodiscard]] Real capacitance(Real v) const noexcept {
        if (v <= v_.front()) return c_.front();
        if (v >= v_.back())  return c_.back();
        const Size j = segment_(v);
        const Real h = v_[j + 1] - v_[j];
        const Real t = (v - v_[j]) / h;
        const Real d = (q_[j + 1] - q_[j]) / h;
        const Real c2 = 3 * d - 2 * c_[j] - c_[j + 1];
        const Real c3 = c_[j] + c_[j + 1] - 2 * d;
        return c_[j] + t * (2 * c2 + 3 * t * c3);
    }

private:
    std::vector<Real> v_, q_, c_;
    Real fc_max_ = Real{0};
    Size n_clamped_ = 0;

    [[nodiscard]] Size segment_(Real v) const noexcept {
        const auto it = std::upper_bound(v_.begin(), v_.end(), v);
        const auto j = static_cast<Size>(it - v_.begin());
        return j == 0 ? Size{0} : j - 1;
    }

    [[nodiscard]] Size zero_index_() const noexcept {
        for (Size k = 0; k < v_.size(); ++k) if (v_[k] == Real{0}) return k;
        return 0;
    }

    void validate_(const std::string& what, bool check_q = true) const {
        if (v_.size() != q_.size() || v_.size() != c_.size()) {
            throw std::invalid_argument(std::format(
                "{}: {} voltage knots, {} charge knots and {} capacitance "
                "knots — they must be the same length.",
                what, v_.size(), q_.size(), c_.size()));
        }
        if (v_.size() < 3) {
            throw std::invalid_argument(std::format(
                "{}: a charge table needs at least 3 knots (got {}) — two "
                "define a line, which is a linear capacitor; use "
                "add_capacitor for that.", what, v_.size()));
        }
        for (Size k = 0; k < v_.size(); ++k) {
            if (!std::isfinite(v_[k]) || (k > 0 && !(v_[k] > v_[k - 1]))) {
                throw std::invalid_argument(std::format(
                    "{}: voltage knots must be strictly increasing and "
                    "finite; knot {} = {}{}.", what, k, v_[k],
                    k > 0 ? std::format(" follows {}", v_[k - 1]) : ""));
            }
            if (!(c_[k] > Real{0}) || !std::isfinite(c_[k])) {
                throw std::invalid_argument(std::format(
                    "{}: capacitance at knot {} (v = {}) is {} — it must be "
                    "positive and finite; a zero or negative C is a charge "
                    "that does not rise with voltage, and Newton would "
                    "converge on it.", what, k, v_[k], c_[k]));
            }
        }
        bool has_zero = false;
        for (Size k = 0; k < v_.size(); ++k) {
            if (v_[k] == Real{0}) {
                has_zero = true;
                if (check_q && q_[k] != Real{0}) {
                    throw std::invalid_argument(std::format(
                        "{}: the knot at v = 0 must carry Q = 0 (got {}) — "
                        "charge is measured from zero bias.", what, q_[k]));
                }
            }
        }
        if (!has_zero) {
            throw std::invalid_argument(std::format(
                "{}: no knot at v = 0 — the table's charge is referenced "
                "to zero bias (Q(0) = 0), so one knot must sit exactly "
                "there.", what));
        }
        if (check_q) {
            for (Size k = 1; k < q_.size(); ++k) {
                if (!(q_[k] > q_[k - 1]) || !std::isfinite(q_[k])) {
                    throw std::invalid_argument(std::format(
                        "{}: charge knots must be strictly increasing — Q "
                        "that falls or holds as v rises is a zero or "
                        "NEGATIVE capacitance. Knot {}: Q = {} follows "
                        "Q = {} (v = {} → {}).",
                        what, k, q_[k], q_[k - 1], v_[k - 1], v_[k]));
                }
            }
        }
    }

    /// Pull any tangent pair outside the Fritsch–Carlson circle
    /// (α² + β² ≤ 9) radially onto it — Q monotone, C ≥ 0 between knots.
    void clamp_to_monotone_region_() {
        fc_max_ = Real{0};
        n_clamped_ = 0;
        for (Size k = 0; k + 1 < v_.size(); ++k) {
            const Real d = (q_[k + 1] - q_[k]) / (v_[k + 1] - v_[k]);
            const Real alpha = c_[k] / d, beta = c_[k + 1] / d;
            const Real r2 = alpha * alpha + beta * beta;
            fc_max_ = std::max(fc_max_, r2);
            if (r2 > Real{9}) {
                const Real tau = Real{3} / std::sqrt(r2);
                c_[k]     = tau * alpha * d;
                c_[k + 1] = tau * beta * d;
                ++n_clamped_;
            }
        }
    }
};

}  // namespace pulsim::models
