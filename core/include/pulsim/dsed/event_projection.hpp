#pragma once

// =============================================================================
// Pulsim — consistent reinitialization at mode changes
// =============================================================================
//
// v2.0 Phase 3, item 2 (the audit's "obra №3").
//
// THE PROBLEM. A mask change reconfigures the circuit INSTANTLY, and
// the state the integrator carries across the event is, in general,
// not consistent with the new mode:
//
//   * DCM: the diode turns off at i ≈ 0, but the root is located in
//     TIME to 1e-10 s, and at di/dt ~ 1e5 A/s that leaves ~µA in the
//     inductor. The new mode drains it through g_off with
//     τ = L·g_off ≈ 1e-13 s — a time constant the explicit
//     integrator must resolve step by step. Measured: a DCM buck
//     ground 10M steps without covering 5 ms of simulated time.
//   * Paralleled capacitors: a switch closing across two caps at
//     different voltages creates a fast equalization through R_on.
//     Physics says the voltages snap to the charge-conserving value
//     and the difference-energy dissipates; the integrator says
//     "resolve the R_on·C transient".
//
// THE PROJECTION. Both are the same phenomenon: the event deposits
// the state OFF the new mode's slow manifold, exciting decay modes
// far below any step the integrator will take. So project onto that
// manifold: eigendecompose the new mode's A, keep every slow
// component exactly as it was, and move every fast STABLE component
// to its quasi-static value −(wᵢ·b)/λᵢ. In the DAE limit
// (g_off → 0, R_on → 0) this converges to the audit's formulation —
// x⁺ = argmin‖x⁺ − x⁻‖_M subject to the new mode's constraints,
// M = diag(C, L) — because the slow eigenvectors become the
// conserved charge/flux combinations: for two paralleled caps the
// preserved coordinate is (C₁v₁ + C₂v₂)/(C₁+C₂), which IS charge
// conservation, and the projected jump IS the physical equalization.
//
// WHAT IS DELIBERATELY NOT PROJECTED. A mode is only projected when
// Re(λ) is strongly negative — decayed within a small fraction of
// one dt_max, i.e. invisible to the integrator anyway. Resonant
// modes (Re ≈ 0, however large |Im|) are physical ringing and pass
// through untouched; so does anything unstable. When no fast stable
// mode exists — every CCM converter — the projection is exactly the
// identity, and a test pins that.

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"

#include <Eigen/Eigenvalues>

#include <complex>

namespace pulsim::dsed {

/// Project `x` onto the slow manifold of `dx/dt = A·x + b`.
///
/// Components along eigenvectors with `Re(λ) < -fast_threshold` are
/// moved to their quasi-static values; every other component is
/// preserved exactly. Returns `x` unchanged when there is nothing to
/// project — or when the eigenbasis is unusable (defective or
/// near-singular V), because a wrong projection is worse than the
/// grind it would have saved.
[[nodiscard]] inline Vector project_onto_slow_manifold(
    const DenseMatrix& A,
    const Vector& b,
    const Vector& x,
    Real fast_threshold) {
    const auto n = A.rows();
    if (n == 0 || !(fast_threshold > Real{0})) {
        return x;
    }

    Eigen::EigenSolver<DenseMatrix> es(A);
    if (es.info() != Eigen::Success) {
        return x;
    }
    const auto& lambda = es.eigenvalues();

    // Fast, STABLE modes only. Conjugate pairs share Re(λ), so the
    // fast set is conjugate-closed and the projected state is real.
    bool any_fast = false;
    for (Eigen::Index i = 0; i < n; ++i) {
        if (lambda[i].real() < -fast_threshold) {
            any_fast = true;
            break;
        }
    }
    if (!any_fast) {
        return x;             // the common case: exact identity
    }

    const Eigen::MatrixXcd V = es.eigenvectors();
    Eigen::FullPivLU<Eigen::MatrixXcd> lu(V);
    if (!lu.isInvertible()) {
        // Defective or numerically rank-deficient eigenbasis: the
        // split is meaningless. Decline rather than guess.
        return x;
    }

    // Modal coordinates of the state and the drive.
    const Eigen::VectorXcd z  = lu.solve(x.cast<std::complex<Real>>());
    const Eigen::VectorXcd wb = lu.solve(b.cast<std::complex<Real>>());

    Eigen::VectorXcd z_new = z;
    for (Eigen::Index i = 0; i < n; ++i) {
        if (lambda[i].real() < -fast_threshold) {
            // Quasi-static: 0 = λ·z* + (w·b)  ⇒  z* = −(w·b)/λ.
            z_new[i] = -wb[i] / lambda[i];
        }
    }

    Vector out = (V * z_new).real();
    if (!out.allFinite()) {
        return x;             // never trade a grind for a NaN
    }
    return out;
}

}  // namespace pulsim::dsed
