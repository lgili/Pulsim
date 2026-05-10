#pragma once

#include "pulsim/v1/numeric_types.hpp"

#include <array>
#include <complex>
#include <numbers>
#include <utility>

namespace pulsim::v1::grid {

// =============================================================================
// add-three-phase-grid-library — Phase 4: symmetrical components
// =============================================================================
//
// Fortescue decomposition of an unbalanced 3φ phasor set into
// positive (+), negative (-), and zero (0) sequence components.
//
//   [V0]   1   [1  1  1] [Va]
//   [V+] = - · [1  α  α²] [Vb]
//   [V-]   3   [1  α² α ] [Vc]
//
// where α = e^{j·2π/3}. The implementation operates on **complex
// phasors** (one frequency at a time); time-domain Fortescue with a
// 1/4-period delay is the alternative for non-phasor inputs.

struct PhasorSet {
    std::complex<Real> a{0.0, 0.0};
    std::complex<Real> b{0.0, 0.0};
    std::complex<Real> c{0.0, 0.0};
};

struct SequenceComponents {
    std::complex<Real> zero{0.0, 0.0};
    std::complex<Real> positive{0.0, 0.0};
    std::complex<Real> negative{0.0, 0.0};
};

/// Decompose a phasor 3-tuple into its sequence components.
/// Convenience: `pure_positive_sequence` test produces zeros on the
/// other two components.
[[nodiscard]] inline SequenceComponents fortescue(const PhasorSet& v) {
    constexpr Real two_pi_3 = Real{2} * std::numbers::pi_v<Real> / Real{3};
    const std::complex<Real> alpha{std::cos( two_pi_3), std::sin( two_pi_3)};
    const std::complex<Real> alpha2{std::cos(-two_pi_3), std::sin(-two_pi_3)};

    SequenceComponents s;
    s.zero     = (v.a + v.b + v.c) / Real{3};
    s.positive = (v.a + alpha  * v.b + alpha2 * v.c) / Real{3};
    s.negative = (v.a + alpha2 * v.b + alpha  * v.c) / Real{3};
    return s;
}

/// Reconstruct the (a, b, c) phasors from sequence components.
/// Identity round-trip with `fortescue`.
[[nodiscard]] inline PhasorSet inverse_fortescue(const SequenceComponents& s) {
    constexpr Real two_pi_3 = Real{2} * std::numbers::pi_v<Real> / Real{3};
    const std::complex<Real> alpha{std::cos( two_pi_3), std::sin( two_pi_3)};
    const std::complex<Real> alpha2{std::cos(-two_pi_3), std::sin(-two_pi_3)};

    PhasorSet v;
    v.a = s.zero + s.positive + s.negative;
    v.b = s.zero + alpha2 * s.positive + alpha  * s.negative;
    v.c = s.zero + alpha  * s.positive + alpha2 * s.negative;
    return v;
}

/// Convenience: degree of unbalance, defined as |V_neg / V_pos|. Above
/// ~ 2 % is typically the IEEE / IEC threshold for an "unbalanced"
/// supply.
[[nodiscard]] inline Real unbalance_factor(const SequenceComponents& s) {
    const Real pos_mag = std::abs(s.positive);
    if (pos_mag < Real{1e-12}) return Real{0};
    return std::abs(s.negative) / pos_mag;
}

}  // namespace pulsim::v1::grid
