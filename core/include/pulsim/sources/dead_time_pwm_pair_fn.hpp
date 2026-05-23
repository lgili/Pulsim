#pragma once

// =============================================================================
// Pulsim v2 — Half-bridge PWM pair: `make_dead_time_pwm_pair_fn`
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

#include "pulsim/numeric/types.hpp"
#include "pulsim/solver/options.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <algorithm>
#include <cmath>

namespace pulsim::sources {

/// Build a `SwitchScheduleFn` driving a complementary HS/LS
/// switch pair at `frequency` with `duty` cycle and symmetric
/// `dead_time` inserted at each commutation. Other switch
/// bits remain OFF.
///
/// Returns a `std::function<SwitchStateMask(Real)>` that
/// can be passed directly to `run_transient`.
///
/// Example: 100 kHz / 50 % synchronous buck with 100 ns
/// dead-time, HS = switch index 0, LS = switch index 1:
///
///   auto sw = make_dead_time_pwm_pair_fn(
///       100e3, 0.5, 0, 1, /*n_sw=*/2, /*dt=*/100e-9);
///   auto res = run_transient(cache, graph, pool, opts, sw);
[[nodiscard]] inline solver::SwitchScheduleFn
make_dead_time_pwm_pair_fn(
    Real frequency,
    Real duty,
    Size hs_switch_idx,
    Size ls_switch_idx,
    Size num_switches,
    Real dead_time,
    Real phase = Real{0}) {
    if (frequency <= Real{0}) {
        // Degenerate: never toggle. Return a flat-OFF lambda.
        return [num_switches](Real /*t*/) {
            return topology::SwitchStateMask{num_switches};
        };
    }
    const Real T_sw = Real{1} / frequency;
    // Saturate dead-time to [0, T/2] so the function is
    // well-defined even when the user passes nonsense
    // (e.g. dt > T means both switches off forever).
    const Real dt_eff = std::max(
        Real{0}, std::min(dead_time, T_sw * Real{0.5}));
    // Pre-compute the four phase boundaries once. Note:
    // `t_hs_end` may go negative if dt_eff > duty·T (no HS
    // ON time); the comparison `t_cycle < t_hs_end` then
    // always fails — correct behavior. Same for `t_ls_end`.
    const Real t_hs_end   = duty * T_sw - dt_eff;
    const Real t_ls_start = duty * T_sw;
    const Real t_ls_end   = T_sw - dt_eff;
    return [T_sw, t_hs_end, t_ls_start, t_ls_end,
            hs_switch_idx, ls_switch_idx,
            num_switches, phase](Real t) {
        topology::SwitchStateMask mask{num_switches};
        const Real t_shifted = t - phase;
        Real t_cycle = std::fmod(t_shifted, T_sw);
        if (t_cycle < Real{0}) t_cycle += T_sw;
        if (t_cycle < t_hs_end) {
            mask.set(hs_switch_idx, true);
        } else if (t_cycle >= t_ls_start &&
                   t_cycle <  t_ls_end) {
            mask.set(ls_switch_idx, true);
        }
        // Otherwise: dead-time, both OFF (default mask).
        return mask;
    };
}

}  // namespace pulsim::sources
