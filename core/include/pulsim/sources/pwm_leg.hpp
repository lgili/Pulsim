#pragma once

// =============================================================================
// Pulsim — the complementary-leg thresholds shared by the PWM helpers and
// the CarrierPwm modulator (split out of switch_schedule.hpp so that
// events/modulator.hpp, which the control graph includes, stays light)
// =============================================================================

#include "pulsim/numeric/pwm_phase.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/switch_state.hpp"

namespace pulsim::sources::detail {

/// Thresholds of one complementary HS/LS leg, as fractions of the
/// carrier period: HS on for u < hs_end, LS on for ls_start <= u <
/// ls_end, both off otherwise (dead time).
struct PairThresholds {
    Real hs_end;
    Real ls_start;
    Real ls_end;
};

enum class LegState : unsigned char { Off, High, Low };

[[nodiscard]] inline LegState pair_state(
    const pwm_phase::CyclePosition& p, const PairThresholds& th) noexcept {
    if (!pwm_phase::reached(p, th.hs_end)) return LegState::High;
    if (pwm_phase::reached(p, th.ls_start) &&
        !pwm_phase::reached(p, th.ls_end)) {
        return LegState::Low;
    }
    return LegState::Off;
}

inline void stamp_leg(topology::SwitchStateMask& mask, LegState s,
                      Size hs, Size ls) {
    if (s == LegState::High) mask.set(hs, true);
    if (s == LegState::Low) mask.set(ls, true);
}

}  // namespace pulsim::sources::detail
