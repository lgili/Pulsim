#pragma once

#include "pulsim/v1/numeric_types.hpp"

#include <cmath>
#include <numbers>
#include <tuple>

namespace pulsim::v1::motors {

// =============================================================================
// add-motor-models — frame transformations (Phase 2)
// =============================================================================
//
// Standard Clarke (3φ → αβ stationary) and Park (αβ → dq rotating)
// transforms used by every dq-frame motor model. All in
// "amplitude-invariant" form (the 2/3 prefactor on Clarke), which is
// the convention motor-control textbooks (Krause, Mohan) and TI / ST
// motor-control libraries use.
//
// Conventions:
//   - Three-phase quantities are (a, b, c).
//   - Stationary two-phase is (α, β).
//   - Rotating two-phase is (d, q), aligned with the rotor angle θ_e
//     (electrical angle, not mechanical — for a 4-pole machine,
//     θ_e = 2 · θ_mech).

/// Clarke transform: 3φ (a, b, c) → 2φ stationary (α, β).
[[nodiscard]] inline std::pair<Real, Real> clarke(Real a, Real b, Real c) noexcept {
    constexpr Real two_thirds = Real{2} / Real{3};
    constexpr Real one_third  = Real{1} / Real{3};
    constexpr Real one_over_sqrt3 = Real{1} / Real{1.7320508075688772};
    const Real alpha = two_thirds * a - one_third * (b + c);
    const Real beta  = (b - c) * one_over_sqrt3;
    return {alpha, beta};
}

/// Inverse Clarke: αβ → abc.
[[nodiscard]] inline std::tuple<Real, Real, Real> inverse_clarke(
    Real alpha, Real beta) noexcept {
    constexpr Real sqrt3_over_2 = Real{0.8660254037844386};
    const Real a = alpha;
    const Real b = -Real{0.5} * alpha + sqrt3_over_2 * beta;
    const Real c = -Real{0.5} * alpha - sqrt3_over_2 * beta;
    return {a, b, c};
}

/// Park transform: αβ stationary → dq rotating at electrical angle θ_e.
///   d =  α cos θ + β sin θ
///   q = -α sin θ + β cos θ
[[nodiscard]] inline std::pair<Real, Real> park(Real alpha, Real beta,
                                                  Real theta_e) noexcept {
    const Real c = std::cos(theta_e);
    const Real s = std::sin(theta_e);
    const Real d =  alpha * c + beta * s;
    const Real q = -alpha * s + beta * c;
    return {d, q};
}

/// Inverse Park: dq → αβ.
///   α = d cos θ - q sin θ
///   β = d sin θ + q cos θ
[[nodiscard]] inline std::pair<Real, Real> inverse_park(
    Real d, Real q, Real theta_e) noexcept {
    const Real c = std::cos(theta_e);
    const Real s = std::sin(theta_e);
    const Real alpha = d * c - q * s;
    const Real beta  = d * s + q * c;
    return {alpha, beta};
}

/// Composite abc → dq (Clarke + Park).
[[nodiscard]] inline std::pair<Real, Real> abc_to_dq(
    Real a, Real b, Real c, Real theta_e) noexcept {
    const auto [alpha, beta] = clarke(a, b, c);
    return park(alpha, beta, theta_e);
}

/// Composite dq → abc (inverse Park + inverse Clarke).
[[nodiscard]] inline std::tuple<Real, Real, Real> dq_to_abc(
    Real d, Real q, Real theta_e) noexcept {
    const auto [alpha, beta] = inverse_park(d, q, theta_e);
    return inverse_clarke(alpha, beta);
}

}  // namespace pulsim::v1::motors
