#pragma once

// =============================================================================
// Pulsim v2 — Layer 2 V12: PulseVoltageSource device model
// =============================================================================
//
// First-class pulse / step voltage source. Output:
//   * `v_initial`  for t < t_start
//   * `v_pulsed`   for t ∈ [t_start, t_start + pulse_width)
//   * `v_initial`  for t after the pulse (single-shot mode)
//   * If `period > 0` → repeats: pulse fires again at
//                       t_start + N · period for every N ≥ 1.
//
// Architecturally identical to PWMVoltageSource (V4): a
// Source-kind branch with a branch-current unknown stamped
// at V=0 baseline; the time-varying value is overlaid via
// `b_extra` at runtime by `run_transient`'s built-in AC pass.
//
// Schema:
//   v_initial    pre-pulse / between-pulses level   [V]
//   v_pulsed     level during the pulse window      [V]
//   t_start      delay before the first pulse fires [s]
//   pulse_width  duration of each pulse             [s]
//   period       repetition period (0 = single-shot)[s]
//
// Use cases:
//   * Step response: v_initial=0, v_pulsed=V, t_start=0,
//     pulse_width=very-large → asymptotic step.
//   * Clock signal: v_initial=0, v_pulsed=5, period=100µs,
//     pulse_width=50µs → 5 V square wave at 10 kHz, 50 % duty.
//   * Initial-condition perturbation: a single pulse to
//     kick a system away from equilibrium.
//   * Pulse generator simulation (timer ICs, monoshots).
//
// vs. PWMVoltageSource (V4): PWMVoltageSource is a continuous
// square wave (frequency + duty), useful for power-electronics
// gate drives. PulseVoltageSource adds explicit `t_start`
// delay and single-shot mode — useful for transient-analysis
// step inputs that V4 cannot express.

#include "pulsim/v2/numeric/concepts.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/topology/graph.hpp"

#include <cmath>

namespace pulsim::v2::models {

struct PulseVoltageSource {
    struct Params {
        Real v_initial   = Real{0};     // [V] baseline
        Real v_pulsed    = Real{1};     // [V] during pulse
        Real t_start     = Real{0};     // [s] first-pulse delay
        Real pulse_width = Real{1e-3};  // [s] pulse duration
        Real period      = Real{0};     // [s] 0 = single-shot
    };

    static constexpr topology::BranchKind kind =
        topology::BranchKind::Source;
    static constexpr Size num_terminals = 2;
    static constexpr bool is_linear  = true;
    static constexpr bool is_dynamic = true;   // time-varying
    static constexpr bool needs_branch_unknown = true;

    /// Instantaneous output value at simulation time t.
    /// Returns v_initial outside the pulse window(s),
    /// v_pulsed inside.
    [[nodiscard]] static Real value_at(
        const Params& p, Real t) noexcept {
        if (t < p.t_start) {
            return p.v_initial;
        }
        const Real elapsed = t - p.t_start;
        if (p.period > Real{0}) {
            // Periodic mode: wrap elapsed into [0, period).
            Real phase = std::fmod(elapsed, p.period);
            if (phase < Real{0}) phase += p.period;
            return (phase < p.pulse_width) ? p.v_pulsed
                                             : p.v_initial;
        }
        // Single-shot mode: pulse just once.
        return (elapsed < p.pulse_width) ? p.v_pulsed
                                            : p.v_initial;
    }
};

}  // namespace pulsim::v2::models
