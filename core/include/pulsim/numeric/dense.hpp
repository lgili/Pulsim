#pragma once

// =============================================================================
// Pulsim — Layer 0: dense vector + matrix aliases
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
// it operates on a `pulsim::Vector` is unambiguous to the reader,
// vs `Eigen::Matrix<double, Eigen::Dynamic, 1>` which is verbose and
// hides the precision (Real, not necessarily double).
//
// If we ever need to swap Eigen for a different backend (GPU offload
// via custom kernels, for example), the swap surface is the few
// aliases below — not 50+ files using `Eigen::VectorXd` directly.

#include "pulsim/numeric/types.hpp"

#include <Eigen/Dense>

namespace pulsim {

// -----------------------------------------------------------------------------
// Dense vector — N × 1 column vector. Templatized on Scalar (v1.4.0+) so
// the same type expresses both real-valued PWL state vectors and
// complex-valued AC-sweep vectors.
//
// `Vector` (no template arg) remains == `VectorT<Real>` for backward
// compatibility with every Layer 1-9 consumer.
//
// Layer 4 + Layer 5 use `Vector` for the system state x(t), RHS b, etc.
// Layer 5 (`analysis/mna_sweep.hpp`) uses `VectorT<std::complex<Real>>`
// for the per-frequency AC sweep.
// -----------------------------------------------------------------------------
template <typename Scalar>
using VectorT = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

using Vector = VectorT<Real>;

// -----------------------------------------------------------------------------
// Dense matrix — R × C matrix. Same templating pattern as VectorT.
//
// Used for small dense blocks: state-space matrices (A, B, C, D) per
// switch combination in Layer 4, per-device transfer matrices in
// Layer 3, etc. Typical size 2×2 to 20×20 — well within the "dense is
// faster than sparse" regime.
// -----------------------------------------------------------------------------
template <typename Scalar>
using DenseMatrixT = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

using DenseMatrix = DenseMatrixT<Real>;

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

}  // namespace pulsim
