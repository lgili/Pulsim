#pragma once

// =============================================================================
// Pulsim — Compose multiple switch schedulers: `make_combined_switch_fn`
// =============================================================================
//
// SMPS circuits frequently drive several INDEPENDENT switch
// groups simultaneously: e.g. an isolated converter with a
// phase-shift full-bridge on the primary AND a synchronous
// rectifier (S/R pair) on the secondary; or a 3-φ inverter
// with an auxiliary boost-PFC stage in front.
//
// Without composition, the user manually combines the masks
// from individual helpers:
//
//   auto sw = [psfb, sr](Real t) {
//       auto m = psfb(t);
//       auto s = sr(t);
//       for (Size b = 0; b < m.size(); ++b)
//           if (s.get(b)) m.set(b, true);
//       return m;
//   };
//
// `make_combined_switch_fn` collapses that to one call:
//
//   auto sw = make_combined_switch_fn(
//       n_sw,
//       {make_phase_shift_full_bridge_fn(...),
//        make_dead_time_pwm_pair_fn(...)});
//
// Each component helper drives DIFFERENT switch indices; the
// combined result is the bitwise OR of all sub-masks.
//
// Precondition: all sub-callbacks must return masks of size
// >= num_switches (typical: they were all built with the
// same num_switches argument). The combined mask has size
// exactly num_switches.

#include "pulsim/numeric/types.hpp"
#include "pulsim/solver/options.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/sources/switch_schedule.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace pulsim::sources {

/// Build a `SwitchSchedule` that returns, for each `t`,
/// the bitwise OR of every sub-callback's mask. All sub-
/// callbacks are evaluated at `t` and combined.
///
/// `num_switches` is the size of the returned mask; bits
/// beyond `num_switches` in any sub-mask are ignored.
///
/// `next_edge_after(t)` is the earliest edge of the sub-
/// schedules when every non-empty sub-callback is a
/// `SwitchSchedule` with an edge oracle (all the built-in
/// helpers); otherwise it is +inf (unknown), and the
/// variable-step engines search for the edges themselves.
[[nodiscard]] inline SwitchSchedule
make_combined_switch_fn(
    Size num_switches,
    std::vector<solver::SwitchScheduleFn> fns) {
    auto shared = std::make_shared<
        const std::vector<solver::SwitchScheduleFn>>(std::move(fns));
    bool all_known = true;
    for (const auto& fn : *shared) {
        if (!fn) continue;
        const auto* sched = as_schedule(fn);
        if (sched == nullptr || !sched->has_edges()) all_known = false;
    }
    std::function<Real(Real)> next_edge;
    if (all_known) {
        next_edge = [shared](Real t) {
            Real best = std::numeric_limits<Real>::infinity();
            for (const auto& fn : *shared) {
                if (!fn) continue;
                best = std::min(best, as_schedule(fn)->next_edge_after(t));
            }
            return best;
        };
    }
    return SwitchSchedule{
        [num_switches, shared](Real t) {
            topology::SwitchStateMask result{num_switches};
            for (const auto& fn : *shared) {
                if (!fn) continue;
                auto sub = fn(t);
                const Size sz = std::min(num_switches,
                                           sub.size());
                for (Size b = 0; b < sz; ++b) {
                    if (sub.get(b)) result.set(b, true);
                }
            }
            return result;
        },
        std::move(next_edge)};
}

}  // namespace pulsim::sources
