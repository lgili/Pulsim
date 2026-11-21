#pragma once

// Piecewise-linear table waveform (add-controlled-source-set): the value,
// the next knot for breakpoint landing, and table validation that names
// the source. Used by `PwlCurrentSource` (models/controlled_sources.hpp).

#include "pulsim/numeric/types.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace pulsim::models {

/// Piecewise-linear table i(t): linear between knots (t_k, i_k), the
/// first value before t_0, then the last value — or, when `periodic`,
/// the table repeated with period t_last − t_0.
struct PwlWaveform {
    std::vector<Real> t;
    std::vector<Real> v;
    bool periodic = false;

    /// Refuses fewer than two knots, unequal lengths, non-increasing
    /// times, non-finite entries, and `periodic` with unequal end
    /// values (a jump at the wrap). `who` names the source.
    static void validate(const PwlWaveform& w, std::string_view who) {
        if (w.t.size() != w.v.size()) {
            throw std::invalid_argument(std::format(
                "{}: the time and value lists have different lengths "
                "({} and {})", who, w.t.size(), w.v.size()));
        }
        if (w.t.size() < 2) {
            throw std::invalid_argument(std::format(
                "{}: a table needs at least two knots, got {}", who,
                w.t.size()));
        }
        for (std::size_t k = 0; k < w.t.size(); ++k) {
            if (!std::isfinite(w.t[k]) || !std::isfinite(w.v[k])) {
                throw std::invalid_argument(std::format(
                    "{}: knot {} is not finite (t = {}, value = {})",
                    who, k, w.t[k], w.v[k]));
            }
            if (k > 0 && !(w.t[k] > w.t[k - 1])) {
                throw std::invalid_argument(std::format(
                    "{}: times must increase strictly; knot {} "
                    "(t = {}) does not follow knot {} (t = {})",
                    who, k, w.t[k], k - 1, w.t[k - 1]));
            }
        }
        if (w.periodic && w.v.front() != w.v.back()) {
            throw std::invalid_argument(std::format(
                "{}: a periodic table must end where it starts "
                "(first value {}, last value {}), or it jumps at "
                "every wrap", who, w.v.front(), w.v.back()));
        }
    }

    /// Map `t` into the table's own time base.
    [[nodiscard]] Real local_time(Real tt) const noexcept {
        if (periodic && tt > t.back()) {
            const Real T = t.back() - t.front();
            return t.front() + std::fmod(tt - t.front(), T);
        }
        return tt;
    }

    [[nodiscard]] Real value_at(Real tt) const noexcept {
        const Real u = local_time(tt);
        if (u <= t.front()) return v.front();
        if (u >= t.back()) return v.back();
        const auto it = std::upper_bound(t.begin(), t.end(), u);
        const auto k = static_cast<std::size_t>(it - t.begin());
        const Real a = (u - t[k - 1]) / (t[k] - t[k - 1]);
        return v[k - 1] + a * (v[k] - v[k - 1]);
    }

    /// First knot strictly after `t` (+inf when none).
    [[nodiscard]] Real next_breakpoint_after(Real tt) const noexcept {
        if (tt < t.front()) return t.front();
        if (!periodic) {
            const auto it = std::upper_bound(t.begin(), t.end(), tt);
            return it == t.end() ? std::numeric_limits<Real>::infinity()
                                 : *it;
        }
        const Real T = t.back() - t.front();
        const Real base = t.front() + std::floor((tt - t.front()) / T) * T;
        const Real u = tt - base + t.front();
        const auto it = std::upper_bound(t.begin(), t.end(), u);
        return base + ((it == t.end()) ? T : (*it - t.front()));
    }
};

}  // namespace pulsim::models
