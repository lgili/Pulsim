#pragma once

#include "pulsim/v1/numeric_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <tuple>
#include <utility>
#include <vector>

namespace pulsim::v1::grid {

// =============================================================================
// add-three-phase-grid-library — Phase 1: 3φ sources
// =============================================================================
//
// Time-domain 3φ voltage / current generators. They're math objects:
// `evaluate(t)` returns the instantaneous (a, b, c) triple at time t.
// The downstream simulator integration (registering them as Circuit
// devices that stamp three branch equations) lands once the
// Circuit-variant integration follow-up arrives.

enum class PhaseSequence : std::uint8_t {
    Positive,   ///< a → b → c, +120° ordering (default for utility grids)
    Negative,   ///< a → c → b, -120° ordering
};

/// Balanced sinusoidal 3φ voltage source.
///
///   v_a(t) = V_pk · cos(ω·t + φ)
///   v_b(t) = V_pk · cos(ω·t + φ ∓ 2π/3)        (∓ depending on sequence)
///   v_c(t) = V_pk · cos(ω·t + φ ± 2π/3)
struct ThreePhaseSource {
    Real v_rms       = 230.0;                 ///< per-phase RMS (V)
    Real frequency   = 50.0;                  ///< Hz
    Real phase_rad   = 0.0;                   ///< initial phase (rad)
    PhaseSequence sequence = PhaseSequence::Positive;

    [[nodiscard]] std::tuple<Real, Real, Real> evaluate(Real t) const noexcept {
        const Real V_pk = v_rms * std::numbers::sqrt2_v<Real>;
        const Real omega = Real{2} * std::numbers::pi_v<Real> * frequency;
        const Real theta = omega * t + phase_rad;
        const Real shift = Real{2} * std::numbers::pi_v<Real> / Real{3};
        const Real shift_b = (sequence == PhaseSequence::Positive) ? -shift :  shift;
        const Real shift_c = (sequence == PhaseSequence::Positive) ?  shift : -shift;
        return {
            V_pk * std::cos(theta),
            V_pk * std::cos(theta + shift_b),
            V_pk * std::cos(theta + shift_c),
        };
    }
};

/// Programmable 3φ source. Wraps a base `ThreePhaseSource` and applies
/// a per-phase scale envelope `(g_a, g_b, g_c)` that the user updates
/// at run-time. Useful for sag / swell test fixtures (drop g_a to 0.5
/// at t = t_sag for a 50 % single-phase sag) and for slow envelope
/// modulation (line-frequency hum injection on a converter).
struct ThreePhaseSourceProgrammable {
    ThreePhaseSource base;
    Real g_a = 1.0;
    Real g_b = 1.0;
    Real g_c = 1.0;

    [[nodiscard]] std::tuple<Real, Real, Real> evaluate(Real t) const noexcept {
        const auto [a, b, c] = base.evaluate(t);
        return {g_a * a, g_b * b, g_c * c};
    }

    /// Helper: trigger a step-change sag on phase A at `t_sag`.
    /// `(*this).g_a` becomes `g_after` for `t ≥ t_sag` if the user
    /// queries via `evaluate_with_sag`. Stateless on purpose — the
    /// integration layer is the simulator's outer loop.
    [[nodiscard]] std::tuple<Real, Real, Real> evaluate_with_sag(
        Real t,
        Real t_sag,
        Real g_a_after) const noexcept {
        if (t >= t_sag) {
            const auto [a, b, c] = base.evaluate(t);
            return {g_a_after * a, g_b * b, g_c * c};
        }
        return evaluate(t);
    }
};

/// Harmonic-injected 3φ source. Fundamental + arbitrary list of
/// (harmonic order, fraction, phase) triples. Each harmonic respects
/// the base sequence (positive harmonics rotate same direction).
struct HarmonicComponent {
    int  order      = 5;           ///< 5 = 5th harmonic (250 Hz on 50 Hz grid)
    Real magnitude_pct = 0.05;     ///< amplitude as fraction of fundamental
    Real phase_rad  = 0.0;         ///< extra phase offset (rad)
};

struct ThreePhaseHarmonicSource {
    ThreePhaseSource fundamental;
    std::vector<HarmonicComponent> harmonics;

    [[nodiscard]] std::tuple<Real, Real, Real> evaluate(Real t) const noexcept {
        const Real V_pk = fundamental.v_rms * std::numbers::sqrt2_v<Real>;
        const Real omega_1 = Real{2} * std::numbers::pi_v<Real> *
                             fundamental.frequency;
        const Real shift = Real{2} * std::numbers::pi_v<Real> / Real{3};
        const Real shift_b = (fundamental.sequence == PhaseSequence::Positive)
                              ? -shift : shift;
        const Real shift_c = -shift_b;

        Real a = Real{0}, b = Real{0}, c = Real{0};
        // Fundamental contribution.
        const Real th = omega_1 * t + fundamental.phase_rad;
        a += V_pk * std::cos(th);
        b += V_pk * std::cos(th + shift_b);
        c += V_pk * std::cos(th + shift_c);

        // Harmonic contributions. Each harmonic phasor rotates at
        // `n · ω_1` and respects the same sequence as the fundamental
        // (so triplen harmonics — orders 3, 9, 15... — fold into the
        // zero-sequence component).
        for (const auto& h : harmonics) {
            const Real th_n = static_cast<Real>(h.order) * omega_1 * t +
                              fundamental.phase_rad + h.phase_rad;
            const Real V_n = V_pk * h.magnitude_pct;
            a += V_n * std::cos(th_n);
            b += V_n * std::cos(th_n + static_cast<Real>(h.order) * shift_b);
            c += V_n * std::cos(th_n + static_cast<Real>(h.order) * shift_c);
        }
        return {a, b, c};
    }
};

}  // namespace pulsim::v1::grid
