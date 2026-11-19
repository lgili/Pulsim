#pragma once

// =============================================================================
// Pulsim — the one mode-set service (add-unified-event-queue task 8.2)
// =============================================================================
//
// What every engine does at a mode decision, written once:
//
//   * `crossing_signal` / `forbidden` / `held_tolerance`: a PWL diode's
//     per-direction commutation signal at a solved state, the side its own
//     rule forbids, and the noise band of a forbidden-side reading.
//   * `redecide`: the diodes re-decide their bits from a solved state.
//   * `MaskCycle`: the masks one settle call solved under; a repeat is a
//     CYCLE (on a memoryless linear solve, a proof of non-convergence). It
//     is reset per call, so a cycle is reported only by the call that
//     found it: a restart stage never stops on an earlier stage's cycle
//     (test_bridge_ground_truth pins it).
//   * `settle`: the fixed-point iteration "solve under the mask, re-decide,
//     repeat until nothing flips", with a report of the rounds taken and of
//     how it ended (settled, a cycle, or stopped by the caller).
//
// The engines own what a round is (the fixed engine's companion solve or
// Newton solve, TR-BDF2's zero-time probe); this owns the loop shape and
// the report. Header-only, no allocation, every call inlined.

#include <array>
#include <cmath>
#include <limits>

#include "pulsim/models/switched_diode.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/diode_event_state.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/topology/switch_state.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define PULSIM_MODE_SET_INLINE [[gnu::always_inline]] inline
#else
#define PULSIM_MODE_SET_INLINE inline
#endif

namespace pulsim::events {

/// Per-direction diode crossing signal at a solved state. ON watches
/// v_D − V_f (i_D = g_on·(v_D − V_f), zero at v_D = V_f), OFF watches
/// v_D − V_on (`SwitchedDiode::crossing_signal`).
inline Real crossing_signal(const pwl::DiodeEventState& diodes,
                            Size entry_idx, const Vector& x) {
    const auto& e = diodes.entries()[entry_idx];
    // A branch-form diode reads its solved current i_s while ON (D5).
    const Real v_d = e.law_voltage(x, e.is_on);
    return models::SwitchedDiode::crossing_signal(e.is_on, v_d, e.params());
}

/// True when `s` (the diode's crossing_signal) puts it on the side its
/// own commutation rule would leave — `SwitchedDiode::decide`: ON stays
/// iff g_on·(v_D − V_f) > 0, OFF stays iff v_D < V_on.
inline bool forbidden(const pwl::DiodeEventState& diodes,
                      Size entry_idx, Real s) {
    return diodes.entries()[entry_idx].is_on ? (s <= Real{0})
                                             : (s >= Real{0});
}

/// What counts as numerical noise in a forbidden-side signal: 1e-8 of
/// the diode's terminal voltages (plus V_th, V_f and 1 V). A V_th = 0 diode
/// riding its zero-current boundary sits there (the chatter guard owns
/// that case); a missed commutation leaves it volts away.
inline Real held_tolerance(const pwl::DiodeEventState& diodes,
                           Size entry_idx, const Vector& x) {
    const auto& e = diodes.entries()[entry_idx];
    // An ON branch-form diode's signal is its current (review L5).
    if (e.is_on && e.var >= 0) {
        return Real{1e-8} * (Real{1} + std::abs(e.V_f) + std::abs(x[e.var]));
    }
    // An ON conductance-form diode's signal is V_f + i/g_on: 1e-8 of the
    // node voltages is i = 0.7 mA at g_on = 1e3 on a 36 V boost, so a step
    // from +0.37 to -0.71 mA was "riding the boundary", never landed, and
    // the diode turned off later carrying -0.7 mA (sw at -3.5e5 V, 9577
    // floor-accepted steps). Its noise is a leakage-scale current,
    // g_off * (1 + |v_D|) (a bridge diode riding at -1.6e-7 A against
    // 3.3e-7 A), read as voltage through 1/g_on, plus the round-off of
    // forming v_D from its node voltages; a DCM buck's 9e-7 A overshoot
    // (446 V over g_off) is then a crossing. v_D, not the node voltages:
    // a diode floating at 10 kV common mode had a floor of 2e-8 V, a
    // 2e-5 A reverse current taken for noise.
    if (e.is_on && e.g_on > Real{0}) {
        constexpr Real eps = std::numeric_limits<Real>::epsilon();
        const Real va = stamping::read_node_voltage(x, e.from);
        const Real vk = stamping::read_node_voltage(x, e.to);
        // a latch with its gate off holds down to I_H: its threshold sits
        // I_H/g_on above V_f (add-pwl-igbt-and-thyristor 4.2; 0 ungated)
        return e.g_off / e.g_on
                   * (Real{1} + std::abs(e.V_th) + std::abs(e.V_f)
                      + std::abs(e.I_H) / e.g_on + std::abs(va - vk))
               + Real{64} * eps * (std::abs(va) + std::abs(vk));
    }
    return Real{1e-8}
           * (Real{1} + std::abs(e.V_th) + std::abs(e.V_f)
              + std::abs(stamping::read_node_voltage(x, e.from))
              + std::abs(stamping::read_node_voltage(x, e.to)));
}

/// The diodes re-decide their bits from the solved state `x`; true when
/// any bit changed. An OFF ideal (R_on = 0) diode turns ON only past the
/// round-off floor of `update_from_state_floored`.
PULSIM_MODE_SET_INLINE bool redecide(pwl::DiodeEventState& diodes, const Vector& x) {
    return diodes.update_from_state_floored(x);
}

/// The masks ONE settle call solved under (at most N kept; past N a repeat
/// goes unseen and the round budget bounds the call).
template <Size N>
class MaskCycle {
public:
    /// A new call: nothing seen.
    PULSIM_MODE_SET_INLINE void reset() noexcept { n_ = 0; }

    /// True when `m` was solved under earlier in this call.
    [[nodiscard]] PULSIM_MODE_SET_INLINE bool seen(const topology::SwitchStateMask& m) const {
        for (Size ms = 0; ms < n_; ++ms) {
            if (seen_[ms] == m) return true;
        }
        return false;
    }

    /// True (a cycle) when `m` was seen in this call; otherwise records it.
    PULSIM_MODE_SET_INLINE bool repeat(const topology::SwitchStateMask& m) {
        if (seen(m)) return true;
        if (n_ < N) seen_[n_++] = m;
        return false;
    }

private:
    std::array<topology::SwitchStateMask, N> seen_{};
    Size n_ = 0;
};

/// How one settle round ended.
enum class Round {
    Settled,   ///< solved, nothing flipped: the fixed point
    Flipped,   ///< solved, a bit flipped: another round
    Cycle,     ///< not solved: the mask repeats one of this call's
    Stop,      ///< not solved: the caller stopped (a failed probe)
};

/// What a settle call did. `rounds` counts the SOLVED rounds.
struct SettleReport {
    Size rounds = 0;
    bool settled = false;
    bool cycle = false;
    bool stopped = false;
};

/// Runs `round(i)` (i = rounds solved so far) until it settles, reports a
/// cycle or a stop, or `max_rounds` rounds were solved.
template <class RoundFn>
PULSIM_MODE_SET_INLINE SettleReport settle(Size max_rounds, RoundFn&& round) {
    SettleReport r;
    while (r.rounds < max_rounds) {
        const Round o = round(r.rounds);
        if (o == Round::Cycle) { r.cycle = true; break; }
        if (o == Round::Stop) { r.stopped = true; break; }
        ++r.rounds;
        if (o == Round::Settled) { r.settled = true; break; }
    }
    return r;
}

}  // namespace pulsim::events
