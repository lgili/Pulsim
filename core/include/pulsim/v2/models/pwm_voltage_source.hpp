#pragma once

// =============================================================================
// Pulsim v2 — Layer 2 V4: PWMVoltageSource device model
// =============================================================================
//
// First-class PWM voltage source: a square-wave switching
// between `v_high` and `v_low` at a given frequency + duty
// cycle. Eliminates the common pattern of writing a custom
// `b_extra_fn(t)` lambda for SMPS gate drives.
//
// Structurally it's a VoltageSource (same MNA stamping: a
// `Source` branch with a branch-current unknown and a
// constraint row). The DC baseline is 0; the time-varying
// PWM value is overlaid via `b_extra` at runtime by
// `run_transient`'s built-in PWM pass.
//
// Schema:
//   v_high       output voltage when ON  [V]
//   v_low        output voltage when OFF [V]
//   frequency    switching frequency     [Hz]
//   duty         on-time fraction ∈ [0, 1]
//   phase        phase offset            [s] (default 0)
//
// Use case examples:
//   * MOSFET gate drive at 100 kHz / 50 % duty.
//   * Synthetic AC test source (square approximation).
//   * Open-loop chopper / buck-leg driver.

#include "pulsim/v2/numeric/concepts.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/topology/graph.hpp"

#include <cmath>

namespace pulsim::v2::models {

struct PWMVoltageSource {
    struct Params {
        Real v_high     = Real{1};   // [V]
        Real v_low      = Real{0};   // [V]
        Real frequency  = Real{1e3}; // [Hz]
        Real duty       = Real{0.5}; // [unitless, 0..1]
        Real phase      = Real{0};   // [s] — start-of-cycle offset
    };

    static constexpr topology::BranchKind kind =
        topology::BranchKind::Source;
    static constexpr Size num_terminals = 2;
    static constexpr bool is_linear  = true;
    static constexpr bool is_dynamic = true;   // time-varying
    static constexpr bool needs_branch_unknown = true;

    /// Instantaneous PWM output value at simulation time t.
    /// Convention: ON during the first `duty · T` of each
    /// period (where T = 1/frequency), OFF for the rest.
    /// Returns `v_high` or `v_low` (no slew — instant
    /// transitions).
    [[nodiscard]] static Real value_at(
        const Params& p, Real t) noexcept {
        if (p.frequency <= Real{0}) {
            return p.v_low;
        }
        const Real T_period = Real{1} / p.frequency;
        const Real t_shifted = t - p.phase;
        // std::fmod with negative dividend can return
        // negative — wrap into [0, T_period).
        Real phase01 = std::fmod(t_shifted / T_period,
                                   Real{1});
        if (phase01 < Real{0}) phase01 += Real{1};
        return (phase01 < p.duty) ? p.v_high : p.v_low;
    }
};

}  // namespace pulsim::v2::models
