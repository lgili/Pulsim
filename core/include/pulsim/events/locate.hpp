#pragma once

// =============================================================================
// Pulsim — the one state-event locator (add-unified-event-queue task 8.1)
// =============================================================================
//
// Illinois (modified regula falsi) over a TRIAL EVALUATOR: the caller owns
// what a trial is (a trapezoidal re-step from the committed state, an
// ESDIRK re-step, an interpolant) and what the signal is; this owns the
// bracket arithmetic. TR-BDF2's crossing localisation runs on it.
//
// One iteration:
//   * t = the secant root of the bracket, clamped into its inner 90 %;
//   * `skip(t, bracket)` true ends the search before any trial (the
//     caller's conditioning floor);
//   * `trial(t, s)` solves at t and writes the signal s there; false (a
//     failed or non-finite trial) ends the search with the bracket as is;
//   * the end whose signal has the same sign as `sa` moves to t;
//     `moved(side)` lets the caller keep that end's state; the OTHER end's
//     signal is halved when the same end moved twice running (the Illinois
//     weight; `sa_raw` / `sb_raw` keep the unhalved values);
//   * `done(bracket)` true after a move ends it.
//
// The bracket's ends are where the caller says they are: TR-BDF2 starts
// `ta` at the COMMITTED state's time with the committed state as its state
// (never a probe or a trial state; test_bridge_ground_truth pins it).
//
// Header-only, no allocation; every call is inlined into the caller.

#include <algorithm>

#include "pulsim/numeric/types.hpp"

namespace pulsim::events {

/// The bracket [ta, tb] of a sign change, with the signals at its ends.
/// `sa` / `sb` carry the Illinois weights; `sa_raw` / `sb_raw` do not.
struct Bracket {
    Real ta = 0;
    Real sa = 0;
    Real tb = 0;
    Real sb = 0;
    Real sa_raw = 0;
    Real sb_raw = 0;
    int side = 0;   // -1: ta moved last, +1: tb moved last, 0: none yet
};

/// Which end of the bracket a trial replaced.
enum class BracketEnd : int { A = -1, B = +1 };

/// The Illinois search (see the file comment). Returns the number of
/// trials taken.
template <class Skip, class Trial, class Moved, class Done>
#if defined(__GNUC__) || defined(__clang__)
[[gnu::always_inline]]
#endif
inline int illinois(Bracket& b, int max_iter, Skip&& skip, Trial&& trial,
                    Moved&& moved, Done&& done) {
    int trials = 0;
    for (int it = 0; it < max_iter; ++it) {
        const Real t_loc = std::clamp(
            b.ta - b.sa * (b.tb - b.ta) / (b.sb - b.sa),
            b.ta + Real{0.05} * (b.tb - b.ta),
            b.tb - Real{0.05} * (b.tb - b.ta));
        if (skip(t_loc, static_cast<const Bracket&>(b))) {
            break;
        }
        Real sm = 0;
        ++trials;
        if (!trial(t_loc, sm)) {
            break;
        }
        if ((sm > Real{0}) == (b.sa > Real{0})) {
            b.ta = t_loc;
            b.sa = sm;
            b.sa_raw = sm;
            moved(BracketEnd::A);
            if (b.side == -1) { b.sb *= Real{0.5}; }
            b.side = -1;
        } else {
            b.tb = t_loc;
            b.sb = sm;
            b.sb_raw = sm;
            moved(BracketEnd::B);
            if (b.side == +1) { b.sa *= Real{0.5}; }
            b.side = +1;
        }
        if (done(static_cast<const Bracket&>(b))) {
            break;
        }
    }
    return trials;
}

}  // namespace pulsim::events
