#pragma once

// =============================================================================
// Pulsim — exact stepping of an LTI mode
// =============================================================================
//
// v2.0 Phase 3, item 2's other half.
//
// WHY THIS EXISTS. The consistent-reinitialization projection puts
// the post-event state exactly on the new mode's slow manifold — and
// the DCM buck STILL ground to a halt at h ≈ 2e-10 s, with the state
// frozen at its equilibrium. That measurement killed the premise the
// projection was built on: an explicit integrator is STABILITY-
// limited, not accuracy-limited. DOPRI5's region ends near
// |hλ| ≈ 3.3, so a mode with λ ≈ −5e9 pins h below a nanosecond
// forever, no matter how quiescent the state is. No projection, and
// no error controller, changes that.
//
// THE WAY OUT is that between events a PWL circuit is not a hard ODE
// at all — it is LTI with (for DC sources) constant b, and its
// trajectory has a closed form:
//
//   z = V⁻¹x,  w = V⁻¹b
//   z(t+h) = e^{Λh}·z(t) + h·φ₁(Λh)·w,     φ₁(s) = (eˢ − 1)/s
//   x(t+h) = V·z(t+h)                       (real part)
//
// exact for ANY h — a nanosecond or a millisecond — with no
// stability region, no local truncation error, and no step-size
// controller. The φ₁ form keeps λ = 0 (pure integrator modes) exact
// too. This is what the established event-driven simulators actually
// do, and it is why they have no concept of "the idle mode is too
// stiff".
//
// Event location gets SHARPER as well: predicates are evaluated on
// the analytic trajectory instead of a Hermite interpolant, so the
// Illinois root is exact to the eigendecomposition's precision.
//
// WHEN IT DOES NOT APPLY, say so and fall back: a defective or
// ill-conditioned eigenbasis (the factory returns disabled, the
// scheduler keeps using RK45), or time-varying sources (b(t) is no
// longer constant within the mode; RK45 remains correct there).

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"

#include <Eigen/Eigenvalues>

#include <cmath>
#include <complex>

namespace pulsim::dsed {

/// The cached eigendecomposition of one mode's A, ready to advance
/// a state by an arbitrary h in O(n²).
struct ExactLTI {
    bool valid = false;
    Eigen::MatrixXcd V;        // right eigenvectors
    Eigen::MatrixXcd V_inv;
    Eigen::VectorXcd lambda;
};

/// Build the exact stepper for `A`. `valid == false` — with the
/// reason left to the caller's fallback — when the eigenproblem
/// fails or the eigenbasis is numerically rank-deficient (defective
/// A: a critically-damped pair, a Jordan block). Exactness is the
/// whole point, so a shaky basis disqualifies rather than degrades.
[[nodiscard]] inline ExactLTI make_exact_lti(const DenseMatrix& A) {
    ExactLTI out;
    if (A.rows() == 0) {
        return out;
    }
    Eigen::EigenSolver<DenseMatrix> es(A);
    if (es.info() != Eigen::Success) {
        return out;
    }
    Eigen::FullPivLU<Eigen::MatrixXcd> lu(es.eigenvectors());
    if (!lu.isInvertible()) {
        return out;
    }
    out.V = es.eigenvectors();
    out.V_inv = lu.inverse();
    out.lambda = es.eigenvalues();
    out.valid = true;
    return out;
}

/// φ₁(s) = (eˢ − 1)/s, the exact-integrator kernel. The series
/// branch keeps small |s| (including exactly 0) at full precision —
/// the direct form loses every digit to cancellation there.
[[nodiscard]] inline std::complex<Real> phi1(std::complex<Real> s) {
    if (std::abs(s) < Real{1e-6}) {
        // 1 + s/2 + s²/6 + s³/24 — error O(|s|⁴) < 1e-24.
        return Real{1} + s * (Real{0.5} + s * (Real{1} / Real{6}
               + s * (Real{1} / Real{24})));
    }
    return (std::exp(s) - Real{1}) / s;
}

/// Advance `x` by `h` under dx/dt = A·x + b, exactly.
[[nodiscard]] inline Vector exact_advance(const ExactLTI& e,
                                            const Vector& x,
                                            const Vector& b,
                                            Real h) {
    const Eigen::VectorXcd z = e.V_inv * x.cast<std::complex<Real>>();
    const Eigen::VectorXcd w = e.V_inv * b.cast<std::complex<Real>>();
    Eigen::VectorXcd z_new(z.size());
    for (Eigen::Index i = 0; i < z.size(); ++i) {
        const std::complex<Real> s = e.lambda[i] * h;
        // exp(s) for strongly-decayed modes underflows to 0 —
        // exactly the right answer, no special case needed.
        z_new[i] = std::exp(s) * z[i] + h * phi1(s) * w[i];
    }
    return (e.V * z_new).real();
}

}  // namespace pulsim::dsed
