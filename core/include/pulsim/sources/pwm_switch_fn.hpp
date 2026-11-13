#pragma once

// =============================================================================
// Pulsim — PWM helper: `make_pwm_switch_fn`
// =============================================================================
//
// SMPS users typically drive a controlled switch (MOSFET,
// IGBT, IdealSwitch) with a PWM signal. Without this helper,
// they write a lambda:
//
//   constexpr Real T_sw = 1.0 / 100e3;   // 10 µs
//   auto switch_fn = [n_sw](Real t) {
//       const Real phase = std::fmod(t, T_sw) / T_sw;
//       SwitchStateMask mask(n_sw);
//       if (phase < 0.5) mask.set(0, true);
//       return mask;
//   };
//
// `make_pwm_switch_fn` collapses the lambda into one call:
//
//   auto switch_fn = make_pwm_switch_fn(
//       /*frequency=*/100e3, /*duty=*/0.5,
//       /*switch_idx=*/0, /*num_switches=*/n_sw);
//
// Other switch bits stay OFF; if the user needs MULTIPLE PWM-
// driven switches (e.g. complementary high-side / low-side),
// they can compose multiple `make_pwm_switch_fn` outputs via
// `make_combined_switch_fn(...)` (V1).

#include "pulsim/numeric/pwm_phase.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/solver/options.hpp"        // for SwitchScheduleFn
#include "pulsim/solver/run_transient.hpp"  // SwitchScheduleFn alias
#include "pulsim/sources/switch_schedule.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <array>
#include <cmath>

namespace pulsim::sources {

/// Build a `SwitchSchedule` that toggles `switch_idx` ON
/// during the first `duty · T` fraction of each period
/// (where `T = 1 / frequency`) and OFF for the rest. All
/// other switch bits remain OFF. `phase` is a time offset in
/// seconds: the period starts at t = phase + k · T.
///
/// Edges that fall on a time-grid point take effect AT that
/// point (`pwm_phase::cycle_position`): on the fixed-step
/// engine an on-time of a whole number of steps is exactly
/// that many steps in every period. The schedule also exposes
/// `next_edge_after(t)`, so the variable-step engines land on
/// the edges instead of sampling for them.
///
/// Converts implicitly to `SwitchScheduleFn`, so it can be
/// passed directly to `run_transient`.
///
/// Example: 100 kHz / 50 % PWM on switch 0 of a 3-switch
/// circuit (buck with body diode + freewheel diode):
///
///   auto sw_fn = make_pwm_switch_fn(100e3, 0.5, 0, 3);
///   auto res = run_transient(cache, graph, pool, opts, sw_fn);
[[nodiscard]] inline SwitchSchedule
make_pwm_switch_fn(
    Real frequency,
    Real duty,
    Size switch_idx,
    Size num_switches,
    Real phase = Real{0}) {
    if (frequency <= Real{0}) {
        // Degenerate: never toggle. Flat-OFF schedule.
        return constant_schedule(topology::SwitchStateMask{num_switches});
    }
    const Real T_sw = Real{1} / frequency;
    return SwitchSchedule{
        [T_sw, duty, switch_idx, num_switches, phase](Real t) {
            topology::SwitchStateMask mask{num_switches};
            const auto p = pwm_phase::cycle_position(t, phase, T_sw);
            if (!pwm_phase::reached(p, duty)) {
                mask.set(switch_idx, true);
            }
            return mask;
        },
        [T_sw, duty, phase](Real t) {
            const std::array<Real, 1> thr{duty};
            return pwm_phase::next_constant_crossing(t, phase, T_sw, thr);
        }};
}

}  // namespace pulsim::sources
