#pragma once

// =============================================================================
// Pulsim — Layer 2 V12: PulseVoltageSource device model
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

#include "pulsim/numeric/concepts.hpp"
#include "pulsim/numeric/pwm_phase.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/graph.hpp"

#include <cmath>
#include <limits>

namespace pulsim::models {

struct PulseVoltageSource {
    struct Params {
        Real v_initial   = Real{0};     // [V] baseline
        Real v_pulsed    = Real{1};     // [V] during pulse
        Real t_start     = Real{0};     // [s] first-pulse delay
        Real pulse_width = Real{1e-3};  // [s] pulse duration (at v_pulsed level)
        Real period      = Real{0};     // [s] 0 = single-shot
        // SPICE-style rise/fall ramps (Layer 2 V14). Default 0
        // = instantaneous transition (matches V12 behavior).
        // `pulse_width` is the time spent AT v_pulsed,
        // excluding the rise/fall ramps. Total period must
        // be > rise_time + pulse_width + fall_time.
        Real rise_time   = Real{0};     // [s] linear ramp 0 → v_pulsed
        Real fall_time   = Real{0};     // [s] linear ramp v_pulsed → 0
    };

    static constexpr topology::BranchKind kind =
        topology::BranchKind::Source;
    static constexpr Size num_terminals = 2;
    static constexpr bool is_linear  = true;
    static constexpr bool is_dynamic = true;   // time-varying
    static constexpr bool needs_branch_unknown = true;

    /// Instantaneous output value at simulation time t.
    /// SPICE PULSE-style with optional rise/fall ramps:
    ///   * t < t_start:                           v_initial
    ///   * 0 ≤ phase < rise_time:                 ramp up
    ///   * rise_time ≤ phase < rise+width:        v_pulsed
    ///   * rise+width ≤ phase < rise+width+fall:  ramp down
    ///   * phase ≥ rise+width+fall (single-shot): v_initial
    /// When period > 0, `phase` wraps into [0, period).
    [[nodiscard]] static Real value_at(
        const Params& p, Real t) noexcept {
        if (t < p.t_start) {
            return p.v_initial;
        }
        Real elapsed = t - p.t_start;
        Real tol = Real{0};
        if (p.period > Real{0}) {
            // Grid-exact edges (pwm_phase.hpp): fmod put an edge on a
            // dt grid point a step early or late (29/30 on-steps per
            // period at period 20 us, width 6 us, dt 0.2 us).
            const auto pos =
                pwm_phase::cycle_position(t, p.t_start, p.period);
            elapsed = pos.u * p.period;
            tol = pos.tol * p.period;
        }
        // `elapsed` has not yet reached `x` (within the snap tolerance).
        const auto before = [&](Real x) { return elapsed < x - tol; };
        // Rising ramp.
        if (p.rise_time > Real{0} &&
                before(p.rise_time)) {
            const Real frac = elapsed / p.rise_time;
            return p.v_initial +
                   (p.v_pulsed - p.v_initial) * frac;
        }
        // Steady at v_pulsed.
        const Real t_after_rise = elapsed - p.rise_time;
        if (before(p.rise_time + p.pulse_width)) {
            return p.v_pulsed;
        }
        // Falling ramp.
        if (p.fall_time > Real{0}) {
            const Real t_into_fall =
                t_after_rise - p.pulse_width;
            if (before(p.rise_time + p.pulse_width + p.fall_time)) {
                const Real frac = t_into_fall / p.fall_time;
                return p.v_pulsed +
                       (p.v_initial - p.v_pulsed) * frac;
            }
        }
        return p.v_initial;
    }

    /// dv/dt just after t (right derivative): the ramp of the piece
    /// `value_at` takes at t, 0 on a level. Same branches and snap
    /// tolerance as `value_at`, so value + slope·τ is that piece.
    [[nodiscard]] static Real slope_at(const Params& p, Real t) noexcept {
        if (t < p.t_start) return Real{0};
        Real elapsed = t - p.t_start;
        Real tol = Real{0};
        if (p.period > Real{0}) {
            const auto pos = pwm_phase::cycle_position(t, p.t_start, p.period);
            elapsed = pos.u * p.period;
            tol = pos.tol * p.period;
        }
        const auto before = [&](Real x) { return elapsed < x - tol; };
        if (p.rise_time > Real{0} && before(p.rise_time)) {
            return (p.v_pulsed - p.v_initial) / p.rise_time;
        }
        if (before(p.rise_time + p.pulse_width)) return Real{0};
        if (p.fall_time > Real{0}
            && before(p.rise_time + p.pulse_width + p.fall_time)) {
            return (p.v_initial - p.v_pulsed) / p.fall_time;
        }
        return Real{0};
    }

    /// The first corner strictly after t (a time at which `value_at`
    /// changes piece), +∞ when none: t_start, then rise / width / fall
    /// ends, every period when periodic.
    [[nodiscard]] static Real next_corner_after(const Params& p, Real t) noexcept {
        if (t < p.t_start) return p.t_start;
        const Real ends[3] = {p.rise_time, p.rise_time + p.pulse_width,
                              p.rise_time + p.pulse_width + p.fall_time};
        if (p.period > Real{0}) {
            Real thr[3];
            for (int i = 0; i < 3; ++i) thr[i] = ends[i] / p.period;
            return pwm_phase::next_constant_crossing(t, p.t_start, p.period, thr);
        }
        for (const Real e : ends) {
            const Real c = p.t_start + e;
            if (c > t) return c;
        }
        return std::numeric_limits<Real>::infinity();
    }
};

}  // namespace pulsim::models
