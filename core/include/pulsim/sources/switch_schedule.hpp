#pragma once

// =============================================================================
// Pulsim — `SwitchSchedule`: a switch_fn that knows its own gate edges
// =============================================================================
//
// The built-in PWM helpers (`make_pwm_switch_fn`,
// `make_dead_time_pwm_pair_fn`, `make_spwm_pair_fn`,
// `make_phase_shift_full_bridge_fn`, `make_three_phase_spwm_fn`,
// `make_combined_switch_fn`) return a `SwitchSchedule`: the mask
// function plus `next_edge_after(t)`, the first gate edge strictly after
// `t`. A variable-step engine lands its steps exactly on those edges
// instead of sampling the mask:
//
//   * DSED without it polls a plain switch_fn every dt_max / 10 (1 us at
//     the default dt_max) and moved every edge onto that poll grid. A
//     synchronous buck at 100 kHz, D = 0.25, 100 ns dead time read
//     14.28 V against 11.49 V, its body diodes never conducted and the
//     low-side switch dissipated 1.27 W against 7.2 mW.
//   * TR-BDF2 bisects a plain switch_fn inside every step; with the
//     oracle it skips the ~30 mask evaluations per edge.
//
// A `SwitchSchedule` converts implicitly to `solver::SwitchScheduleFn`,
// so every existing caller keeps compiling; `as_schedule(fn)` recovers
// the edge oracle from a `SwitchScheduleFn` that stores one.

#include "pulsim/numeric/pwm_phase.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/solver/run_transient.hpp"  // SwitchScheduleFn alias
#include "pulsim/topology/switch_state.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <utility>

namespace pulsim::sources {

/// A switch schedule with an analytic gate-edge oracle.
struct SwitchSchedule {
    /// t -> mask.
    solver::SwitchScheduleFn mask_fn;
    /// t -> first edge strictly after t. Empty: unknown (+inf).
    std::function<Real(Real)> next_edge_fn;

    SwitchSchedule() = default;
    explicit SwitchSchedule(solver::SwitchScheduleFn mask,
                            std::function<Real(Real)> next_edge = {})
        : mask_fn{std::move(mask)}, next_edge_fn{std::move(next_edge)} {}

    explicit operator bool() const noexcept {
        return static_cast<bool>(mask_fn);
    }

    [[nodiscard]] topology::SwitchStateMask operator()(Real t) const {
        return mask_fn(t);
    }

    /// First gate edge strictly after `t`; +inf when unknown. A
    /// returned edge may be a no-op (the mask is equal on both sides),
    /// never a missed one.
    [[nodiscard]] Real next_edge_after(Real t) const {
        return next_edge_fn ? next_edge_fn(t)
                            : std::numeric_limits<Real>::infinity();
    }

    [[nodiscard]] bool has_edges() const noexcept {
        return static_cast<bool>(next_edge_fn);
    }
};

/// The `SwitchSchedule` stored in `fn`, or nullptr when `fn` holds any
/// other callable.
[[nodiscard]] inline const SwitchSchedule* as_schedule(
    const solver::SwitchScheduleFn& fn) noexcept {
    return fn.target<SwitchSchedule>();
}

/// A schedule that never changes: `mask` at every t.
[[nodiscard]] inline SwitchSchedule constant_schedule(
    topology::SwitchStateMask mask) {
    return SwitchSchedule{
        [mask](Real /*t*/) { return mask; },
        [](Real /*t*/) { return std::numeric_limits<Real>::infinity(); }};
}

namespace detail {

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

}  // namespace detail

}  // namespace pulsim::sources
