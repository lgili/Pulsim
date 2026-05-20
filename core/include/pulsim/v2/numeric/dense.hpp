#pragma once

// =============================================================================
// Pulsim v2 — Layer 0: dense vector + matrix aliases
// =============================================================================
//
// `bootstrap-pulsim-v2-kernel` Phase 1. Thin aliases over Eigen.
//
// We deliberately do NOT wrap Eigen in our own classes:
//   * Eigen's expression templates are mature and well-optimised.
//   * Wrapping forces re-implementation of every common operation.
//   * Eigen is already a Pulsim dependency for v1.
//
// The aliases exist for type-name clarity: a Layer-2 device that says
// it operates on a `pulsim::v2::Vector` is unambiguous to the reader,
// vs `Eigen::Matrix<double, Eigen::Dynamic, 1>` which is verbose and
// hides the precision (Real, not necessarily double).
//
// If we ever need to swap Eigen for a different backend (GPU offload
// via custom kernels, for example), the swap surface is the few
// aliases below — not 50+ files using `Eigen::VectorXd` directly.

#include "pulsim/v2/numeric/types.hpp"

#include <Eigen/Dense>

namespace pulsim::v2 {

// -----------------------------------------------------------------------------
// Dense vector — N × 1 column vector of Real.
//
// Layer 4 + Layer 5 use Vector for the system state x(t), RHS b, etc.
// -----------------------------------------------------------------------------
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

// -----------------------------------------------------------------------------
// Dense matrix — R × C matrix of Real.
//
// Used for small dense blocks: state-space matrices (A, B, C, D) per
// switch combination in Layer 4, per-device transfer matrices in
// Layer 3, etc. Typical size 2×2 to 20×20 — well within the "dense is
// faster than sparse" regime.
// -----------------------------------------------------------------------------
using DenseMatrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

// -----------------------------------------------------------------------------
// Fixed-size dense aliases for the very common 2-pin / 3-pin cases.
// Stack-allocated, no heap traffic. Used by Layer 2 device models.
// -----------------------------------------------------------------------------
using Vector2 = Eigen::Matrix<Real, 2, 1>;
using Vector3 = Eigen::Matrix<Real, 3, 1>;
using Vector4 = Eigen::Matrix<Real, 4, 1>;
using Matrix2 = Eigen::Matrix<Real, 2, 2>;
using Matrix3 = Eigen::Matrix<Real, 3, 3>;
using Matrix4 = Eigen::Matrix<Real, 4, 4>;

}  // namespace pulsim::v2
