#pragma once

// =============================================================================
// Pulsim v2 — Compose multiple switch schedulers: `make_combined_switch_fn`
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

#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/solver/options.hpp"
#include "pulsim/v2/solver/run_transient.hpp"
#include "pulsim/v2/topology/switch_state.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace pulsim::v2::sources {

/// Build a `SwitchScheduleFn` that returns, for each `t`,
/// the bitwise OR of every sub-callback's mask. All sub-
/// callbacks are evaluated at `t` and combined.
///
/// `num_switches` is the size of the returned mask; bits
/// beyond `num_switches` in any sub-mask are ignored.
[[nodiscard]] inline solver::SwitchScheduleFn
make_combined_switch_fn(
    Size num_switches,
    std::vector<solver::SwitchScheduleFn> fns) {
    return [num_switches, fns = std::move(fns)](Real t) {
        topology::SwitchStateMask result{num_switches};
        for (const auto& fn : fns) {
            if (!fn) continue;
            auto sub = fn(t);
            const Size sz = std::min(num_switches,
                                       sub.size());
            for (Size b = 0; b < sz; ++b) {
                if (sub.get(b)) result.set(b, true);
            }
        }
        return result;
    };
}

}  // namespace pulsim::v2::sources
