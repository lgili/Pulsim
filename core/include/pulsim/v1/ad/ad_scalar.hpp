#pragma once

// =============================================================================
// PulsimCore v1 - Automatic Differentiation Scalar Bridge
// =============================================================================
// Phase 1 of the `add-automatic-differentiation` change. Provides forward-mode
// AD machinery built on top of `Eigen::AutoDiffScalar<Eigen::VectorXd>` so
// nonlinear device residuals can be authored once as
//
//     template <typename Scalar>
//     Scalar residual_at(span<const Scalar> x_terminals, ...);
//
// and the kernel derives `J = ∂residual/∂x_terminals` automatically when the
// device is stamped under SwitchingMode::Behavioral. Linear devices and
// PWL-mode (Ideal) stamps bypass this path and continue to stamp constants
// directly.
//
// This header intentionally keeps a small surface: a scalar type alias plus
// two helpers. The device migration (Phase 2 of the change) consumes them in
// the per-device `stamp_jacobian_via_ad()` adapter; the validation layer
// (Phase 4) compares AD-derived Jacobians against finite differences.
// =============================================================================

#include "pulsim/v1/numeric_types.hpp"

#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>

#include <cstddef>
#include <span>
#include <vector>

namespace pulsim::v1::ad {

// Forward-mode AD scalar with dynamic derivative-vector size. The derivative
// vector dimension matches the number of independent variables seeded into
// the computation (typically a device's terminal count: 2 for a diode,
// 3 for a 3-terminal device, etc.).
using ADReal = Eigen::AutoDiffScalar<Eigen::VectorXd>;

/// Seed an array of `ADReal` scalars from concrete terminal values. Each
/// returned scalar carries `value = values[i]` and `derivatives = e_i` (the
/// `i`-th unit vector of dimension `values.size()`), so subsequent
/// arithmetic propagates partial derivatives w.r.t. each input automatically.
[[nodiscard]] inline std::vector<ADReal> seed_from_values(
    std::span<const Real> values) {
    const std::size_t n = values.size();
    std::vector<ADReal> seeded;
    seeded.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        ADReal x;
        x.value() = values[i];
        x.derivatives() = Eigen::VectorXd::Unit(static_cast<Eigen::Index>(n),
                                                 static_cast<Eigen::Index>(i));
        seeded.push_back(std::move(x));
    }
    return seeded;
}

/// Convenience overload that takes a contiguous initializer list — useful
/// in unit tests and small-arity device residuals.
[[nodiscard]] inline std::vector<ADReal> seed_from_values(
    std::initializer_list<Real> values) {
    std::vector<Real> tmp(values);
    return seed_from_values(std::span<const Real>(tmp.data(), tmp.size()));
}

/// Extract the per-input partial derivatives of a residual scalar.
/// `residual.derivatives()[i] = ∂residual / ∂x_i`. The returned vector has
/// the same length as the seeded vector that flowed into the computation.
[[nodiscard]] inline Eigen::VectorXd jacobian_row(const ADReal& residual) {
    return residual.derivatives();
}

/// Stack residual rows into a Jacobian matrix `J` of shape `(rows, cols)`,
/// where `cols` is the number of independent variables (uniform across rows).
/// Each row of `J` is the gradient of the corresponding residual scalar.
[[nodiscard]] inline Eigen::MatrixXd stack_jacobian(
    std::span<const ADReal> residuals) {
    if (residuals.empty()) {
        return Eigen::MatrixXd();
    }
    const Eigen::Index cols = residuals.front().derivatives().size();
    Eigen::MatrixXd J(static_cast<Eigen::Index>(residuals.size()), cols);
    for (std::size_t i = 0; i < residuals.size(); ++i) {
        // Defensive: AD scalars produced by ops with a constant operand can
        // emerge with empty derivative vectors. Treat those as a row of zeros.
        const auto& d = residuals[i].derivatives();
        if (d.size() == cols) {
            J.row(static_cast<Eigen::Index>(i)) = d;
        } else {
            J.row(static_cast<Eigen::Index>(i)).setZero();
        }
    }
    return J;
}

}  // namespace pulsim::v1::ad
