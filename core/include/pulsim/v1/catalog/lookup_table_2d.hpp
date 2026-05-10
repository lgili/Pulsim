#pragma once

#include "pulsim/v1/numeric_types.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pulsim::v1::catalog {

// =============================================================================
// add-catalog-device-models — LookupTable2D (Phase 1)
// =============================================================================
//
// Bilinear-interpolating 2D table: `z = f(x, y)`. Used by catalog
// devices for parameters that depend on two operating-point variables
// — typically `(I_c, V_ds)` for switching energies, or `(I_c, T_j)`
// for Vce_sat and forward-voltage tables.
//
// Storage is row-major: `z[i_y][i_x]`, with `x` varying along rows.
// Both axes must be strictly increasing.
//
// Out-of-range queries clamp to the nearest edge (saturation), not
// extrapolate — datasheets simply don't characterize behavior past the
// stated limits, so clamping is more honest than blind extrapolation.

class LookupTable2D {
public:
    LookupTable2D() = default;

    /// Build from 1D x and y arrays plus a flat (x.size() × y.size())
    /// row-major value array. `values[i_y * x.size() + i_x]` is the
    /// table cell at `(x[i_x], y[i_y])`.
    LookupTable2D(std::vector<Real> x,
                  std::vector<Real> y,
                  std::vector<Real> values)
        : x_(std::move(x)), y_(std::move(y)), values_(std::move(values)) {
        if (x_.size() < 2 || y_.size() < 2) {
            throw std::invalid_argument(
                "LookupTable2D: need ≥ 2 points on each axis");
        }
        if (values_.size() != x_.size() * y_.size()) {
            throw std::invalid_argument(
                "LookupTable2D: values size must equal x.size() * y.size()");
        }
        for (std::size_t i = 1; i < x_.size(); ++i) {
            if (!(x_[i] > x_[i - 1])) {
                throw std::invalid_argument(
                    "LookupTable2D: x must be strictly increasing");
            }
        }
        for (std::size_t i = 1; i < y_.size(); ++i) {
            if (!(y_[i] > y_[i - 1])) {
                throw std::invalid_argument(
                    "LookupTable2D: y must be strictly increasing");
            }
        }
    }

    /// Bilinear interpolation at `(x, y)`. Out-of-range clamps.
    [[nodiscard]] Real operator()(Real x, Real y) const {
        if (x_.empty() || y_.empty()) return Real{0};

        // Locate intervals; clamp to edges.
        const auto [ix0, ix1, tx] = locate_(x_, x);
        const auto [iy0, iy1, ty] = locate_(y_, y);

        const Real z00 = at_(ix0, iy0);
        const Real z10 = at_(ix1, iy0);
        const Real z01 = at_(ix0, iy1);
        const Real z11 = at_(ix1, iy1);

        const Real z_lo = z00 + tx * (z10 - z00);
        const Real z_hi = z01 + tx * (z11 - z01);
        return z_lo + ty * (z_hi - z_lo);
    }

    [[nodiscard]] std::size_t size_x() const noexcept { return x_.size(); }
    [[nodiscard]] std::size_t size_y() const noexcept { return y_.size(); }
    [[nodiscard]] const std::vector<Real>& x_axis() const noexcept { return x_; }
    [[nodiscard]] const std::vector<Real>& y_axis() const noexcept { return y_; }
    [[nodiscard]] const std::vector<Real>& values() const noexcept { return values_; }

private:
    std::vector<Real> x_;
    std::vector<Real> y_;
    std::vector<Real> values_;

    [[nodiscard]] Real at_(std::size_t ix, std::size_t iy) const {
        return values_[iy * x_.size() + ix];
    }

    [[nodiscard]] static std::tuple<std::size_t, std::size_t, Real>
    locate_(const std::vector<Real>& axis, Real q) {
        if (q <= axis.front()) return {0, 1, Real{0}};
        if (q >= axis.back()) {
            const std::size_t n = axis.size();
            return {n - 2, n - 1, Real{1}};
        }
        const auto it = std::lower_bound(axis.begin(), axis.end(), q);
        const std::size_t i1 = static_cast<std::size_t>(
            std::distance(axis.begin(), it));
        const std::size_t i0 = i1 - 1;
        const Real t = (q - axis[i0]) / (axis[i1] - axis[i0]);
        return {i0, i1, t};
    }
};

}  // namespace pulsim::v1::catalog
