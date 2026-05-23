#pragma once

// =============================================================================
// Pulsim v2 — PWM helper: `make_pwm_switch_fn`
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

#include "pulsim/numeric/types.hpp"
#include "pulsim/solver/options.hpp"        // for SwitchScheduleFn
#include "pulsim/solver/run_transient.hpp"  // SwitchScheduleFn alias
#include "pulsim/topology/switch_state.hpp"

#include <cmath>

namespace pulsim::sources {

/// Build a `SwitchScheduleFn` that toggles `switch_idx` ON
/// during the first `duty · T` fraction of each period
/// (where `T = 1 / frequency`) and OFF for the rest. All
/// other switch bits remain OFF.
///
/// Returns a `std::function<SwitchStateMask(Real)>` that
/// can be passed directly to `run_transient`.
///
/// Example: 100 kHz / 50 % PWM on switch 0 of a 3-switch
/// circuit (buck with body diode + freewheel diode):
///
///   auto sw_fn = make_pwm_switch_fn(100e3, 0.5, 0, 3);
///   auto res = run_transient(cache, graph, pool, opts, sw_fn);
[[nodiscard]] inline solver::SwitchScheduleFn
make_pwm_switch_fn(
    Real frequency,
    Real duty,
    Size switch_idx,
    Size num_switches,
    Real phase = Real{0}) {
    if (frequency <= Real{0}) {
        // Degenerate: never toggle. Return a flat-OFF lambda.
        return [num_switches](Real /*t*/) {
            return topology::SwitchStateMask{num_switches};
        };
    }
    const Real T_sw = Real{1} / frequency;
    return [T_sw, duty, switch_idx, num_switches, phase](
        Real t) {
        topology::SwitchStateMask mask{num_switches};
        const Real t_shifted = t - phase;
        Real phase01 = std::fmod(t_shifted / T_sw,
                                   Real{1});
        if (phase01 < Real{0}) phase01 += Real{1};
        if (phase01 < duty) {
            mask.set(switch_idx, true);
        }
        return mask;
    };
}

}  // namespace pulsim::sources
