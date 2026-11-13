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

#include "pulsim/numeric/pwm_phase.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/solver/options.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/sources/switch_schedule.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <algorithm>
#include <array>
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

/// Build a `SwitchSchedule` driving a 3-phase VSI with
/// SPWM: common `carrier_frequency`, modulation reference at
/// `modulation_frequency` with phase rotation A → B → C
/// (120° between phases), modulation index
/// `modulation_index ∈ [0, 1]`. Symmetric `dead_time`
/// inserted at each commutation on every leg.
/// `next_edge_after(t)` bisects the crossings like
/// `make_spwm_pair_fn` (same carrier_frequency >
/// pi · M · modulation_frequency assumption).
[[nodiscard]] inline SwitchSchedule
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
        return constant_schedule(topology::SwitchStateMask{num_switches});
    }
    const Real T_c = Real{1} / carrier_frequency;
    const Real omega_m = Real{2} *
        std::numbers::pi_v<Real> * modulation_frequency;
    const Real dt_eff = std::max(
        Real{0}, std::min(dead_time, T_c * Real{0.5}));
    const Real d = dt_eff / T_c;
    // Duty of leg `leg` (0, 1, 2 = A, B, C) at t: 120° phase
    // rotation between legs.
    const auto duty_at = [omega_m, modulation_index,
                          modulation_phase](int leg, Real t) {
        constexpr Real two_pi_3 =
            Real{2} * std::numbers::pi_v<Real> / Real{3};
        const Real ref = std::sin(omega_m * t + modulation_phase
                                  - two_pi_3 * static_cast<Real>(leg));
        return Real{0.5} + Real{0.5} *
            std::clamp(modulation_index * ref, Real{-1}, Real{1});
    };
    return SwitchSchedule{
        [T_c, d, duty_at, carrier_phase, legs, num_switches](Real t) {
            // Carrier position within the current period.
            const auto p = pwm_phase::cycle_position(t, carrier_phase, T_c);
            topology::SwitchStateMask mask{num_switches};
            const std::array<std::array<Size, 2>, 3> idx{{
                {legs.hs_a, legs.ls_a},
                {legs.hs_b, legs.ls_b},
                {legs.hs_c, legs.ls_c}}};
            for (int leg = 0; leg < 3; ++leg) {
                const Real duty = duty_at(leg, t);
                const detail::PairThresholds th{duty - d, duty,
                                                Real{1} - d};
                detail::stamp_leg(mask, detail::pair_state(p, th),
                                  idx[leg][0], idx[leg][1]);
            }
            return mask;
        },
        [T_c, d, duty_at, carrier_phase](Real t) {
            const auto p = pwm_phase::cycle_position(t, carrier_phase, T_c);
            const std::array<Real, 1> fixed{Real{1} - d};
            Real best = pwm_phase::next_constant_crossing(
                t, carrier_phase, T_c, fixed);
            for (int leg = 0; leg < 3; ++leg) {
                best = std::min(best, pwm_phase::next_varying_crossing(
                    t, carrier_phase, T_c, p,
                    [&](Real tau) { return duty_at(leg, tau) - d; }));
                best = std::min(best, pwm_phase::next_varying_crossing(
                    t, carrier_phase, T_c, p,
                    [&](Real tau) { return duty_at(leg, tau); }));
            }
            return best;
        }};
}

}  // namespace pulsim::sources
