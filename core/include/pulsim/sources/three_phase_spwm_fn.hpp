#pragma once

// =============================================================================
// Pulsim — 3-phase voltage-source inverter: `make_three_phase_spwm_fn`
// =============================================================================
//
// The 3-phase two-level voltage-source inverter is the
// workhorse topology for motor drives, grid-tied PV
// inverters, UPS, and ac/dc/ac drives. It uses THREE
// half-bridge legs sharing a common DC bus, each switched
// with naturally-sampled SPWM at the same carrier frequency
// but with the modulation reference shifted by 120°
// (= 2π/3 rad) between phases:
//
//   m_a(t) = M · sin(ω·t + φ)
//   m_b(t) = M · sin(ω·t + φ − 2π/3)        (lags A by 120°)
//   m_c(t) = M · sin(ω·t + φ − 4π/3)        (lags A by 240°)
//
// Each leg uses the same symmetric dead-time pair convention
// from `make_dead_time_pwm_pair_fn` / `make_spwm_pair_fn`,
// so shoot-through never happens on any leg.
//
// The time-averaged mid-points produce a balanced 3-phase
// AC set centred on V_bus / 2. The line-to-line peak voltage
// is V_bus · M · √3 / 2 (textbook result).
//
// Caller passes the 6 switch indices via `ThreePhaseLegIndices`
// to keep the parameter list readable.

#include "pulsim/numeric/types.hpp"
#include "pulsim/solver/options.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace pulsim::sources {

/// 6 switch indices required for a 3-phase 2-level VSI.
/// Each leg has a high-side and low-side power switch (the
/// caller is responsible for inserting them in this order so
/// the indices line up).
struct ThreePhaseLegIndices {
    Size hs_a;
    Size ls_a;
    Size hs_b;
    Size ls_b;
    Size hs_c;
    Size ls_c;
};

/// Build a `SwitchScheduleFn` driving a 3-phase VSI with
/// SPWM: common `carrier_frequency`, modulation reference at
/// `modulation_frequency` with phase rotation A → B → C
/// (120° between phases), modulation index
/// `modulation_index ∈ [0, 1]`. Symmetric `dead_time`
/// inserted at each commutation on every leg.
[[nodiscard]] inline solver::SwitchScheduleFn
make_three_phase_spwm_fn(
    Real carrier_frequency,
    Real modulation_frequency,
    Real modulation_index,
    const ThreePhaseLegIndices& legs,
    Size num_switches,
    Real dead_time,
    Real modulation_phase = Real{0},
    Real carrier_phase    = Real{0}) {
    if (carrier_frequency <= Real{0}) {
        return [num_switches](Real /*t*/) {
            return topology::SwitchStateMask{num_switches};
        };
    }
    const Real T_c = Real{1} / carrier_frequency;
    const Real omega_m = Real{2} *
        std::numbers::pi_v<Real> * modulation_frequency;
    const Real dt_eff = std::max(
        Real{0}, std::min(dead_time, T_c * Real{0.5}));
    // 120° phase rotation between legs. `constexpr` is fine
    // here: it's used as an rvalue inside the closure body
    // without an explicit capture.
    return [T_c, omega_m, dt_eff,
            modulation_index, modulation_phase, carrier_phase,
            legs, num_switches](Real t) {
        constexpr Real two_pi_3 =
            Real{2} * std::numbers::pi_v<Real> / Real{3};
        // Carrier position within the current period.
        const Real t_shifted = t - carrier_phase;
        Real t_cycle = std::fmod(t_shifted, T_c);
        if (t_cycle < Real{0}) t_cycle += T_c;

        topology::SwitchStateMask mask{num_switches};

        // Modulation references for the 3 legs.
        const Real base = omega_m * t + modulation_phase;
        const Real ref_a = std::sin(base);
        const Real ref_b = std::sin(base - two_pi_3);
        const Real ref_c = std::sin(base - two_pi_3 * Real{2});

        // Per-leg PWM stamping — identical math to
        // `make_spwm_pair_fn` but inlined to avoid 3x the
        // std::function indirection per tick.
        auto stamp_leg = [&](Real ref, Size hs, Size ls) {
            const Real duty = Real{0.5} + Real{0.5} *
                std::clamp(modulation_index * ref,
                           Real{-1}, Real{1});
            const Real t_hs_end   = duty * T_c - dt_eff;
            const Real t_ls_start = duty * T_c;
            const Real t_ls_end   = T_c - dt_eff;
            if (t_cycle < t_hs_end) {
                mask.set(hs, true);
            } else if (t_cycle >= t_ls_start &&
                       t_cycle <  t_ls_end) {
                mask.set(ls, true);
            }
        };
        stamp_leg(ref_a, legs.hs_a, legs.ls_a);
        stamp_leg(ref_b, legs.hs_b, legs.ls_b);
        stamp_leg(ref_c, legs.hs_c, legs.ls_c);
        return mask;
    };
}

}  // namespace pulsim::sources
