// SPDX-License-Identifier: MIT
//
// Pulsim — add-in-kernel-event-losses, group 1: datasheet tables in the
// kernel.
//
//   LossTable3D      E(v, i, Tj)  switching energy [J]
//   ConductionTable  V_on(i, Tj)  on-state voltage [V]
//
// Both reproduce `pulsim.LossTable.__call__` (python/pulsim/loss_tables.py)
// operation for operation: per axis an index and a weight, the weight
// allowed outside [0, 1] (linear extrapolation past both ends), a
// single-point axis has weight 0, the multilinear sum taken in the same
// order (first axis outermost), and the result clamped at 0. Clamping
// the axes instead would report a converter run past its datasheet as if
// it sat on the last tabulated point.
//
// Construction refuses, naming the table and the axis: an empty or
// non-strictly-ascending axis, a non-finite axis value, a value array of
// the wrong size, a negative or non-finite entry. A query with a
// non-finite coordinate is refused too (it would otherwise come back NaN).
//
// Immutable after construction; reads are const and allocation-free, so
// one table may be shared by threads.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pulsim::thermal {

namespace detail {

inline std::vector<double> checked_axis(const char* table, const char* name,
                                        std::vector<double> a) {
    if (a.empty()) {
        throw std::invalid_argument(std::string(table) + ": " + name +
                                    " axis is empty");
    }
    for (double x : a) {
        if (!std::isfinite(x)) {
            throw std::invalid_argument(std::string(table) + ": " + name +
                                        " axis has a non-finite value");
        }
    }
    for (std::size_t k = 1; k < a.size(); ++k) {
        if (!(a[k] > a[k - 1])) {
            throw std::invalid_argument(
                std::string(table) + ": " + name +
                " axis must be strictly ascending (interpolation and the "
                "extrapolation slopes read it in order)");
        }
    }
    return a;
}

inline void checked_values(const char* table, const char* what,
                           const std::vector<double>& e, std::size_t want) {
    if (e.size() != want) {
        throw std::invalid_argument(
            std::string(table) + ": " + what + " has " +
            std::to_string(e.size()) + " entries, the axes say " +
            std::to_string(want));
    }
    for (double x : e) {
        if (!std::isfinite(x)) {
            throw std::invalid_argument(std::string(table) + ": " + what +
                                        " has non-finite entries");
        }
        if (x < 0.0) {
            throw std::invalid_argument(std::string(table) + ": " + what +
                                        " must be >= 0");
        }
    }
}

struct AxisWeight {
    std::size_t lo = 0;
    std::size_t hi = 0;
    double w = 0.0;
};

/// `_interp_axis` of loss_tables.py: searchsorted(side="right") - 1,
/// clipped to [0, n-2]; w unbounded (that is the extrapolation).
inline AxisWeight axis_weight(const char* table, const char* name,
                              const std::vector<double>& axis, double q) {
    if (!std::isfinite(q)) {
        throw std::invalid_argument(std::string(table) + ": " + name +
                                    " query is not finite");
    }
    const std::size_t n = axis.size();
    if (n == 1) return {0, 0, 0.0};
    const auto it = std::upper_bound(axis.begin(), axis.end(), q);
    std::ptrdiff_t lo = (it - axis.begin()) - 1;
    lo = std::clamp<std::ptrdiff_t>(lo, 0, static_cast<std::ptrdiff_t>(n) - 2);
    const auto l = static_cast<std::size_t>(lo);
    return {l, l + 1, (q - axis[l]) / (axis[l + 1] - axis[l])};
}

}  // namespace detail

/// Switching energy on a rectilinear (v, i, Tj) grid; `energy` is
/// row-major with shape (n_v, n_i, n_tj), joules.
class LossTable3D {
public:
    LossTable3D(std::vector<double> v_axis, std::vector<double> i_axis,
                std::vector<double> tj_axis, std::vector<double> energy)
        : v_(detail::checked_axis("LossTable3D", "v", std::move(v_axis))),
          i_(detail::checked_axis("LossTable3D", "i", std::move(i_axis))),
          t_(detail::checked_axis("LossTable3D", "tj", std::move(tj_axis))),
          e_(std::move(energy)) {
        detail::checked_values("LossTable3D", "energy", e_,
                               v_.size() * i_.size() * t_.size());
    }

    [[nodiscard]] double operator()(double v, double i, double tj) const {
        const auto av = detail::axis_weight("LossTable3D", "v", v_, v);
        const auto ai = detail::axis_weight("LossTable3D", "i", i_, i);
        const auto at = detail::axis_weight("LossTable3D", "tj", t_, tj);
        const std::array<std::pair<std::size_t, double>, 2> wv{
            {{av.lo, 1.0 - av.w}, {av.hi, av.w}}};
        const std::array<std::pair<std::size_t, double>, 2> wi{
            {{ai.lo, 1.0 - ai.w}, {ai.hi, ai.w}}};
        const std::array<std::pair<std::size_t, double>, 2> wt{
            {{at.lo, 1.0 - at.w}, {at.hi, at.w}}};
        const std::size_t ni = i_.size(), nt = t_.size();
        double out = 0.0;
        for (const auto& [kv, xv] : wv)
            for (const auto& [ki, xi] : wi)
                for (const auto& [kt, xt] : wt)
                    out += e_[(kv * ni + ki) * nt + kt] * xv * xi * xt;
        return std::max(out, 0.0);
    }

    [[nodiscard]] const std::vector<double>& v_axis() const { return v_; }
    [[nodiscard]] const std::vector<double>& i_axis() const { return i_; }
    [[nodiscard]] const std::vector<double>& tj_axis() const { return t_; }
    [[nodiscard]] const std::vector<double>& energy() const { return e_; }

private:
    std::vector<double> v_, i_, t_, e_;
};

/// On-state voltage V_on(i, Tj) on a rectilinear grid; `v_on` is
/// row-major with shape (n_i, n_tj), volts.
class ConductionTable {
public:
    ConductionTable(std::vector<double> i_axis, std::vector<double> tj_axis,
                    std::vector<double> v_on)
        : i_(detail::checked_axis("ConductionTable", "i", std::move(i_axis))),
          t_(detail::checked_axis("ConductionTable", "tj",
                                  std::move(tj_axis))),
          v_(std::move(v_on)) {
        detail::checked_values("ConductionTable", "v_on", v_,
                               i_.size() * t_.size());
    }

    [[nodiscard]] double operator()(double i, double tj) const {
        const auto ai = detail::axis_weight("ConductionTable", "i", i_, i);
        const auto at = detail::axis_weight("ConductionTable", "tj", t_, tj);
        const std::array<std::pair<std::size_t, double>, 2> wi{
            {{ai.lo, 1.0 - ai.w}, {ai.hi, ai.w}}};
        const std::array<std::pair<std::size_t, double>, 2> wt{
            {{at.lo, 1.0 - at.w}, {at.hi, at.w}}};
        const std::size_t nt = t_.size();
        double out = 0.0;
        for (const auto& [ki, xi] : wi)
            for (const auto& [kt, xt] : wt)
                out += v_[ki * nt + kt] * xi * xt;
        return std::max(out, 0.0);
    }

    [[nodiscard]] const std::vector<double>& i_axis() const { return i_; }
    [[nodiscard]] const std::vector<double>& tj_axis() const { return t_; }
    [[nodiscard]] const std::vector<double>& v_on() const { return v_; }

private:
    std::vector<double> i_, t_, v_;
};

}  // namespace pulsim::thermal
