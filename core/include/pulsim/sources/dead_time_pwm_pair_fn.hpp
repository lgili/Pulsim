#pragma once

// =============================================================================
// Pulsim — Half-bridge PWM pair: `make_dead_time_pwm_pair_fn`
// =============================================================================
//
// Half-bridge SMPS (synchronous buck/boost, full-bridge,
// LLC, every voltage-source inverter leg) drives TWO
// complementary switches: a high-side (HS) and a low-side
// (LS) MOSFET / IGBT. The gate driver inserts a small
// "dead-time" band around each commutation to prevent
// shoot-through (both switches conducting simultaneously
// would short the DC bus).
//
// Without this helper, the user writes a 4-phase lambda:
//
//   auto switch_fn = [n_sw, T_sw, duty, dt](Real t) {
//       Real tc = std::fmod(t, T_sw);
//       SwitchStateMask m(n_sw);
//       if      (tc < duty*T_sw - dt)            m.set(hs, true);
//       else if (tc >= duty*T_sw &&
//                tc <  T_sw    - dt)             m.set(ls, true);
//       // else: dead-time, both OFF.
//       return m;
//   };
//
// `make_dead_time_pwm_pair_fn` collapses that to one call:
//
//   auto switch_fn = make_dead_time_pwm_pair_fn(
//       /*frequency=*/100e3, /*duty=*/0.5,
//       /*hs_switch_idx=*/0, /*ls_switch_idx=*/1,
//       /*num_switches=*/n_sw,
//       /*dead_time=*/100e-9);   // 100 ns
//
// Symmetric dead-time convention: each ON duration is
// shrunk by `dead_time` on its TRAILING edge, so:
//   * HS conducts during [0,           duty·T - dt)
//   * dead-time         [duty·T - dt,  duty·T)
//   * LS conducts during [duty·T,      T      - dt)
//   * dead-time         [T - dt,       T)         (wraps)
//
// Invariant: at NO instant are HS and LS both ON.
//
// Other switch bits (e.g. an auto-commutated diode bit) stay
// OFF — combine with `make_combined_switch_fn(...)` if you
// need to drive additional independent PWM channels.

#include "pulsim/numeric/pwm_phase.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/solver/options.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/sources/switch_schedule.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace pulsim::sources {

/// Build a `SwitchSchedule` driving a complementary HS/LS
/// switch pair at `frequency` with `duty` cycle and symmetric
/// `dead_time` inserted at each commutation. Other switch
/// bits remain OFF.
///
/// Edges on a time-grid point take effect at that point, and
/// `next_edge_after(t)` gives the next of the four edges per
/// period (see `make_pwm_switch_fn`). Converts implicitly to
/// `SwitchScheduleFn` for `run_transient`.
///
/// Example: 100 kHz / 50 % synchronous buck with 100 ns
/// dead-time, HS = switch index 0, LS = switch index 1:
///
///   auto sw = make_dead_time_pwm_pair_fn(
///       100e3, 0.5, 0, 1, /*n_sw=*/2, /*dt=*/100e-9);
///   auto res = run_transient(cache, graph, pool, opts, sw);
[[nodiscard]] inline SwitchSchedule
make_dead_time_pwm_pair_fn(
    Real frequency,
    Real duty,
    Size hs_switch_idx,
    Size ls_switch_idx,
    Size num_switches,
    Real dead_time,
    Real phase = Real{0}) {
    if (frequency <= Real{0}) {
        // Degenerate: never toggle. Flat-OFF schedule.
        return constant_schedule(topology::SwitchStateMask{num_switches});
    }
    const Real T_sw = Real{1} / frequency;
    // Saturate dead-time to [0, T/2] so the function is
    // well-defined even when the user passes nonsense
    // (e.g. dt > T means both switches off forever).
    const Real dt_eff = std::max(
        Real{0}, std::min(dead_time, T_sw * Real{0.5}));
    // The three phase boundaries, as fractions of T. Note:
    // `hs_end` may go negative if dt_eff > duty·T (no HS ON
    // time); HS then never turns on — correct behavior.
    const Real d = dt_eff / T_sw;
    const detail::PairThresholds th{duty - d, duty, Real{1} - d};
    return SwitchSchedule{
        [T_sw, th, hs_switch_idx, ls_switch_idx,
         num_switches, phase](Real t) {
            topology::SwitchStateMask mask{num_switches};
            const auto p = pwm_phase::cycle_position(t, phase, T_sw);
            // Otherwise: dead-time, both OFF (default mask).
            detail::stamp_leg(mask, detail::pair_state(p, th),
                              hs_switch_idx, ls_switch_idx);
            return mask;
        },
        [T_sw, th, phase](Real t) {
            const std::array<Real, 3> thr{th.hs_end, th.ls_start,
                                          th.ls_end};
            return pwm_phase::next_constant_crossing(t, phase, T_sw, thr);
        }};
}

}  // namespace pulsim::sources
