#pragma once

// =============================================================================
// Pulsim — phase-control firing schedule for line-commutated converters
// =============================================================================
//
// add-pwl-igbt-and-thyristor task 5.5. For a p-pulse converter synchronised
// to a line of frequency f, gate k (k = 0 .. p-1) is high for `width`
// seconds from
//
//     t_k = (phase + alpha + k * 2*pi/p) / (2*pi*f)   (mod 1/f),
//
// i.e. a PWM of duty width*f on each gate with its own offset, built on the
// same `pwm_phase` arithmetic as `make_pwm_switch_fn` (an edge on a grid
// point takes effect at that point) and with the same edge oracle.
// `alpha` and `phase` are in radians; `phase` places k = 0's natural
// commutation instant (alpha = 0) on the line.

#include "pulsim/numeric/pwm_phase.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/sources/switch_schedule.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace pulsim::sources {

[[nodiscard]] inline SwitchSchedule make_phase_control_fn(
    Real frequency, Real alpha, Real width,
    std::vector<Size> switch_indices, Size num_switches,
    Real phase = Real{0}) {
    const Size p = switch_indices.size();
    const Real T = Real{1} / frequency;
    if (!(frequency > Real{0}) || !std::isfinite(frequency) || p == 0 ||
        !(width > Real{0}) || !(width < T) || !std::isfinite(alpha) ||
        !std::isfinite(phase)) {
        throw std::invalid_argument(std::format(
            "make_phase_control_fn: needs frequency > 0, at least one gate, "
            "0 < width < 1/frequency and finite alpha, phase (got f = {}, "
            "{} gates, width = {}, alpha = {}, phase = {}).",
            frequency, p, width, alpha, phase));
    }
    for (const Size i : switch_indices) {
        if (i >= num_switches) {
            throw std::invalid_argument(std::format(
                "make_phase_control_fn: switch index {} >= num_switches {}.",
                i, num_switches));
        }
    }
    std::vector<Real> offsets(p);
    for (Size k = 0; k < p; ++k) {
        const Real ang = phase + alpha
            + Real(k) * Real{2} * std::numbers::pi_v<Real> / Real(p);
        offsets[k] = ang / (Real{2} * std::numbers::pi_v<Real>) * T;
    }
    const Real duty = width / T;
    return SwitchSchedule{
        [=](Real t) {
            topology::SwitchStateMask mask{num_switches};
            for (Size k = 0; k < p; ++k) {
                const auto c = pwm_phase::cycle_position(t, offsets[k], T);
                if (!pwm_phase::reached(c, duty)) {
                    mask.set(switch_indices[k], true);
                }
            }
            return mask;
        },
        [=](Real t) {
            const std::array<Real, 1> thr{duty};
            Real best = std::numeric_limits<Real>::infinity();
            for (Size k = 0; k < p; ++k) {
                best = std::min(best, pwm_phase::next_constant_crossing(
                                          t, offsets[k], T, thr));
            }
            return best;
        }};
}

}  // namespace pulsim::sources
