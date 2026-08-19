#pragma once

// =============================================================================
// Pulsim — DSED: relative time-comparison helpers (Phase-0 fix #6).
// =============================================================================
//
// The schedulers used ABSOLUTE epsilons for every time comparison:
//
//   * `switch_fn_(t_event + 1e-15)` to sample the mask "just after"
//     an event. ULP(t) grows with t; once t ≳ 16 s (ULP > 2e-15 →
//     half-ULP > 1e-15) the addition rounds back to `t_event`
//     itself, an edge-exclusive switch_fn returns the OLD mask, and
//     the event is dropped as "spurious same-mask" — gate events
//     silently vanish on long grid-frequency / thermal runs.
//   * `|t - t_gate| < 1e-12` gate-landing tests and `1e-15`
//     backtrack/tie guards degrade the same way as t grows.
//
// These helpers make every such comparison RELATIVE to the magnitude
// of the times involved while preserving the old absolute behaviour
// near t = 0 (the floor keeps small-t semantics bit-compatible).
// Shared by scheduler.hpp / scheduler_auto.hpp / scheduler_bdf2.hpp.

#include <algorithm>
#include <cmath>
#include <limits>

#include "pulsim/numeric/types.hpp"

namespace pulsim::dsed {

/// Smallest meaningful time margin at magnitude |t|: a few ULPs,
/// floored at `abs_floor` so behaviour near t = 0 matches the old
/// absolute constants.
[[nodiscard]] inline Real time_margin(
    Real t, Real abs_floor = Real{1e-15}) noexcept {
    const Real rel = Real{4} *
        std::numeric_limits<Real>::epsilon() * std::abs(t);
    return std::max(abs_floor, rel);
}

/// A time strictly after `t` by at least one representable step —
/// use when re-sampling `switch_fn` just past an event so an
/// edge-exclusive driver sees the NEW interval at any simulated t.
[[nodiscard]] inline Real advance_past(Real t) noexcept {
    return t + time_margin(t);
}

/// Scale-aware "same instant" test (gate-landing, event ties).
/// Keeps the legacy 1e-12 absolute floor near t = 0 and switches to
/// a ULP-proportional band as t grows.
[[nodiscard]] inline bool near_time(
    Real a, Real b, Real abs_floor = Real{1e-12}) noexcept {
    const Real rel = Real{8} *
        std::numeric_limits<Real>::epsilon() *
        std::max(std::abs(a), std::abs(b));
    return std::abs(a - b) <= std::max(abs_floor, rel);
}

}  // namespace pulsim::dsed
