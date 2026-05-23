#pragma once

// =============================================================================
// Pulsim v2 — Layer 2 V11: SineVoltageSource device model
// =============================================================================
//
// First-class sinusoidal AC voltage source: produces
// `v_dc + v_amplitude · sin(2π · frequency · t + phase)`
// at each instant. Eliminates the common workaround of
// writing a custom `b_extra_fn(t)` lambda for AC mains /
// grid analysis / audio amp testing / power-line harmonic
// studies.
//
// Structurally identical to PWMVoltageSource (V4): a
// Source-kind branch with a branch-current unknown and an
// MNA constraint row. The constraint row is stamped at V=0
// baseline; the time-varying sine value is overlaid via
// `b_extra` at runtime by `run_transient`'s built-in AC
// pass.
//
// Schema:
//   v_dc          DC offset                       [V]
//   v_amplitude   peak amplitude of sine          [V]
//   frequency     fundamental frequency           [Hz]
//   phase         phase angle (radians, NOT seconds)
//
// Note: `phase` is RADIANS to match the SPWM helpers (V7/V8/
// V9) which all use radian phase offsets. The PWMVoltageSource
// (V4) uses SECONDS because its phase is a cycle offset on a
// square wave — the conventions differ but each is the
// natural unit for its domain.
//
// Use cases:
//   * 50/60 Hz AC mains (v_dc=0, v_amplitude=220√2, f=50).
//   * Half-wave / full-wave / 3-φ diode rectifiers.
//   * Audio amp small-signal characterisation.
//   * Grid-tied inverter "stiff grid" reference.
//   * Power-line harmonic injection (add several sine
//     sources in series at different frequencies).

#include "pulsim/numeric/concepts.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/graph.hpp"

#include <cmath>
#include <numbers>

namespace pulsim::models {

struct SineVoltageSource {
    struct Params {
        Real v_dc        = Real{0};      // [V] DC offset
        Real v_amplitude = Real{1};      // [V] peak AC amplitude
        Real frequency   = Real{50};     // [Hz]
        Real phase       = Real{0};      // [rad]
    };

    static constexpr topology::BranchKind kind =
        topology::BranchKind::Source;
    static constexpr Size num_terminals = 2;
    static constexpr bool is_linear  = true;
    static constexpr bool is_dynamic = true;   // time-varying
    static constexpr bool needs_branch_unknown = true;

    /// Instantaneous output value at simulation time t.
    /// Degenerate cases:
    ///   * frequency <= 0 → returns v_dc (no AC).
    ///   * v_amplitude = 0 → returns v_dc.
    [[nodiscard]] static Real value_at(
        const Params& p, Real t) noexcept {
        if (p.frequency <= Real{0}) {
            return p.v_dc;
        }
        const Real omega = Real{2} *
            std::numbers::pi_v<Real> * p.frequency;
        return p.v_dc +
               p.v_amplitude * std::sin(omega * t + p.phase);
    }
};

}  // namespace pulsim::models
