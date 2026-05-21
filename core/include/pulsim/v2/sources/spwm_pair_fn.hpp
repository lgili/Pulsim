#pragma once

// =============================================================================
// Pulsim v2 — Sine-modulated PWM half-bridge: `make_spwm_pair_fn`
// =============================================================================
//
// Voltage-source inverters (single-phase, motor drives,
// grid-tied PV, UPS) drive a half-bridge HS/LS pair with a
// duty cycle that varies sinusoidally at the fundamental
// frequency. This is "sinusoidal PWM" (SPWM); the carrier is
// a high-frequency triangular reference (or, equivalently
// here, a sawtooth — natural sampling).
//
// At each instant `t`, the modulation reference is:
//
//   m(t) = M · sin(2π · f_mod · t + φ_mod)        ∈ [-M, M]
//
// converted to a duty cycle on the same convention as
// `make_dead_time_pwm_pair_fn`:
//
//   duty(t) = 0.5 + 0.5 · clamp(m(t), -1, 1)       ∈ [0, 1]
//
// The HS/LS pair then switches per the symmetric dead-time
// convention from `make_dead_time_pwm_pair_fn`. Shoot-through
// is impossible by construction.
//
// The half-bridge mid-point time-average equals:
//
//   v_mid_avg(t) = V_bus · duty(t)
//                = V_bus/2 · (1 + M · sin(2π·f_mod·t + φ))
//
// so the AC fundamental at the mid-point has peak amplitude
// V_bus · M / 2. Pair with an LC filter to extract the sine.
//
// Without this helper users wrote:
//
//   auto sw = [...](Real t) {
//       Real ref = M * std::sin(2π*f_mod*t + phi);
//       Real duty = 0.5 + 0.5 * std::clamp(ref, -1.0, 1.0);
//       Real tc = std::fmod(t, T_c);
//       SwitchStateMask m(n_sw);
//       if      (tc < duty*T_c - dt)              m.set(hs, true);
//       else if (tc >= duty*T_c && tc < T_c - dt) m.set(ls, true);
//       return m;
//   };

#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/solver/options.hpp"
#include "pulsim/v2/solver/run_transient.hpp"
#include "pulsim/v2/topology/switch_state.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace pulsim::v2::sources {

/// Build a `SwitchScheduleFn` driving a complementary HS/LS
/// half-bridge with **naturally-sampled SPWM**: carrier at
/// `carrier_frequency`, sine modulation at
/// `modulation_frequency`, modulation index
/// `modulation_index ∈ [0, 1]` (out-of-range values still
/// work — the duty is internally clamped to [0, 1]).
///
/// Symmetric `dead_time` is inserted at each commutation;
/// shoot-through never occurs.
///
/// Example: 10 kHz carrier / 50 Hz fundamental / M = 0.8 /
/// 100 ns dead-time, HS = switch idx 0, LS = switch idx 1:
///
///   auto sw = make_spwm_pair_fn(
///       /*carrier=*/10e3, /*mod=*/50.0, /*M=*/0.8,
///       0, 1, /*n_sw=*/2, /*dead_time=*/100e-9);
[[nodiscard]] inline solver::SwitchScheduleFn
make_spwm_pair_fn(
    Real carrier_frequency,
    Real modulation_frequency,
    Real modulation_index,
    Size hs_switch_idx,
    Size ls_switch_idx,
    Size num_switches,
    Real dead_time,
    Real carrier_phase     = Real{0},
    Real modulation_phase  = Real{0}) {
    if (carrier_frequency <= Real{0}) {
        return [num_switches](Real /*t*/) {
            return topology::SwitchStateMask{num_switches};
        };
    }
    const Real T_c = Real{1} / carrier_frequency;
    const Real omega_m = Real{2} *
        std::numbers::pi_v<Real> * modulation_frequency;
    // Same dead-time saturation policy as the constant-duty
    // pair helper: clamp to [0, T_c/2].
    const Real dt_eff = std::max(
        Real{0}, std::min(dead_time, T_c * Real{0.5}));
    return [T_c, omega_m, modulation_index, modulation_phase,
            carrier_phase, dt_eff,
            hs_switch_idx, ls_switch_idx,
            num_switches](Real t) {
        // 1) Naturally-sampled duty at instant t.
        const Real m_ref = modulation_index *
            std::sin(omega_m * t + modulation_phase);
        const Real m_clamped =
            std::clamp(m_ref, Real{-1}, Real{1});
        const Real duty = Real{0.5} + Real{0.5} * m_clamped;

        // 2) Standard symmetric dead-time pair logic on the
        //    instantaneous duty.
        const Real t_hs_end   = duty * T_c - dt_eff;
        const Real t_ls_start = duty * T_c;
        const Real t_ls_end   = T_c - dt_eff;

        const Real t_shifted = t - carrier_phase;
        Real t_cycle = std::fmod(t_shifted, T_c);
        if (t_cycle < Real{0}) t_cycle += T_c;

        topology::SwitchStateMask mask{num_switches};
        if (t_cycle < t_hs_end) {
            mask.set(hs_switch_idx, true);
        } else if (t_cycle >= t_ls_start &&
                   t_cycle <  t_ls_end) {
            mask.set(ls_switch_idx, true);
        }
        return mask;
    };
}

}  // namespace pulsim::v2::sources
