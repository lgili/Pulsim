#pragma once

// =============================================================================
// Pulsim — Phase-shift full-bridge: `make_phase_shift_full_bridge_fn`
// =============================================================================
//
// Full-bridge isolated DC-DC converters (zero-voltage-
// switching phase-shift, DAB, LLC variants) drive TWO
// half-bridge legs at constant 50 % duty each — the control
// variable is the **phase shift** between the two legs,
// NOT a duty cycle. As the phase shift φ sweeps from 0 to π
// (0 → 180°), the differential output V_AB = V_A − V_B
// transitions from a flat 0 V to a full ±V_bus square wave.
//
// Per leg, the symmetric dead-time convention from
// `make_dead_time_pwm_pair_fn` is reused — shoot-through
// can never happen on either leg.
//
// Without this helper users wrote two separate constant-
// duty pair lambdas and OR'd the masks; this collapses the
// boilerplate.
//
// Conventions:
//   * `phase_shift` is in RADIANS, measured as "leg B lags
//     leg A by phase_shift". 0 = synchronous (no power);
//     π = anti-phase (maximum power delivery, full
//     square wave).
//   * Each leg runs at 50 % duty; users can't change this
//     (it's the definition of a phase-shift converter).
//   * Other switch bits stay OFF.
//
// V_AB pulse pattern (steady state, neglecting dead-time):
//   φ ∈ [0, π]:
//     +V_bus for t in [0,        (T·(π − φ)/2π) / 2)
//     0      ... transition band
//     −V_bus ... half cycle later, mirrored
//   In words: each half cycle has a +V_bus (or −V_bus) pulse
//   of fractional width (π − φ)/π, surrounded by zero-voltage
//   intervals where both legs are at the same rail.

#include "pulsim/numeric/types.hpp"
#include "pulsim/solver/options.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace pulsim::sources {

/// Build a `SwitchScheduleFn` driving a phase-shift full-
/// bridge: two HS/LS half-bridge legs at 50 % duty each,
/// with leg B lagging leg A by `phase_shift` radians.
/// Symmetric `dead_time` inserted at each commutation.
///
/// Example: 100 kHz / quarter-shift (D_eff = 0.5) /
/// 100 ns dead-time:
///
///   auto sw = make_phase_shift_full_bridge_fn(
///       100e3, 0.5 * std::numbers::pi_v<Real>,
///       /*A_HS=*/0, /*A_LS=*/1,
///       /*B_HS=*/2, /*B_LS=*/3,
///       /*n_sw=*/4, /*dead_time=*/100e-9);
[[nodiscard]] inline solver::SwitchScheduleFn
make_phase_shift_full_bridge_fn(
    Real switching_frequency,
    Real phase_shift,
    Size leg_a_hs_idx, Size leg_a_ls_idx,
    Size leg_b_hs_idx, Size leg_b_ls_idx,
    Size num_switches,
    Real dead_time,
    Real carrier_phase = Real{0}) {
    if (switching_frequency <= Real{0}) {
        return [num_switches](Real /*t*/) {
            return topology::SwitchStateMask{num_switches};
        };
    }
    const Real T_sw = Real{1} / switching_frequency;
    // Convert leg-B phase angle (radians) to a time lag.
    // Wrap into [0, T_sw) so out-of-range angles still work.
    constexpr Real two_pi =
        Real{2} * std::numbers::pi_v<Real>;
    Real t_lag = std::fmod(phase_shift, two_pi);
    if (t_lag < Real{0}) t_lag += two_pi;
    t_lag *= T_sw / two_pi;
    const Real dt_eff = std::max(
        Real{0}, std::min(dead_time, T_sw * Real{0.5}));
    // Constant 50 % duty per leg: HS region is the first
    // half of each cycle, LS the second half.
    const Real t_hs_end_a   = T_sw * Real{0.5} - dt_eff;
    const Real t_ls_start_a = T_sw * Real{0.5};
    const Real t_ls_end_a   = T_sw - dt_eff;
    // `dt_eff` is baked into the precomputed boundaries
    // above — no need to capture it.
    return [T_sw, t_hs_end_a, t_ls_start_a, t_ls_end_a,
            t_lag,
            leg_a_hs_idx, leg_a_ls_idx,
            leg_b_hs_idx, leg_b_ls_idx,
            num_switches, carrier_phase](Real t) {
        topology::SwitchStateMask mask{num_switches};

        // Leg A — standard 50 % duty pair.
        const Real t_a = t - carrier_phase;
        Real tc_a = std::fmod(t_a, T_sw);
        if (tc_a < Real{0}) tc_a += T_sw;
        if (tc_a < t_hs_end_a) {
            mask.set(leg_a_hs_idx, true);
        } else if (tc_a >= t_ls_start_a &&
                   tc_a <  t_ls_end_a) {
            mask.set(leg_a_ls_idx, true);
        }

        // Leg B — same 50 % duty pair but lagged by t_lag.
        const Real t_b = t_a - t_lag;
        Real tc_b = std::fmod(t_b, T_sw);
        if (tc_b < Real{0}) tc_b += T_sw;
        if (tc_b < t_hs_end_a) {
            mask.set(leg_b_hs_idx, true);
        } else if (tc_b >= t_ls_start_a &&
                   tc_b <  t_ls_end_a) {
            mask.set(leg_b_ls_idx, true);
        }

        return mask;
    };
}

}  // namespace pulsim::sources
