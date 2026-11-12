#pragma once

// =============================================================================
// Pulsim — carrier phase with grid-exact edges
// =============================================================================
//
// Every built-in PWM generator compares "where am I in the carrier
// period" against a duty threshold. Computed naively,
//
//     phase01 = fmod((t - offset) / T, 1);   on = phase01 < duty;
//
// an edge that lies exactly on a time-grid point flips a step early or
// late at random: the fixed-step engine evaluates the schedule at
// t = t_start + k * dt, and k * dt / T carries a few ulp of rounding.
// At 100 kHz with dt = 1e-7 a 50 % PWM measured 49, 50 or 51 on-steps
// per 100-step period instead of 50, and the duty jitter kept a buck's
// LC filter ringing (67 mV pp of output ripple instead of 17 mV).
//
// `cycle_position` returns the position inside the period together
// with a snap tolerance, and `reached` compares against a threshold
// with it: a position within `tol` of a threshold counts as having
// reached it, and a position within `tol` of the period end rolls into
// the next period. The tolerance is kSnapUlps ulp of the operands
// (|t| + |offset|) / T, far below any physical time step (1.4e-14 s at
// t = 1 s, 100 kHz) and well above the rounding of t_start + k * dt.
// An edge that is not on the grid is unaffected.

#include "pulsim/numeric/types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace pulsim::pwm_phase {

/// Snap tolerance, in ulp of the carrier-phase magnitude.
inline constexpr Real kSnapUlps = Real{64};

struct CyclePosition {
    Real k;    ///< carrier period index: floor((t - offset) / T), snapped
    Real u;    ///< position inside the period, in [0, 1)
    Real tol;  ///< snap tolerance in units of u
};

/// Position of `t` inside the carrier period `T` that starts at
/// `offset` (i.e. t = offset + (k + u) * T). `T` must be > 0.
[[nodiscard]] inline CyclePosition cycle_position(
    Real t, Real offset, Real T) noexcept {
    const Real x = (t - offset) / T;
    const Real scale = std::max(
        Real{1}, (std::abs(t) + std::abs(offset)) / T);
    const Real tol = kSnapUlps *
        std::numeric_limits<Real>::epsilon() * scale;
    Real k = std::floor(x);
    Real u = x - k;
    if (u >= Real{1} - tol) {   // on (or rounding just below) the next period start
        k += Real{1};
        u = Real{0};
    }
    return CyclePosition{k, u, tol};
}

/// Has the position `u` reached the threshold `thr` (both fractions of
/// a period)? A position within `tol` below `thr` counts as reached, so
/// an edge that sits on a grid point always takes effect at that point.
[[nodiscard]] inline bool reached(Real u, Real thr, Real tol) noexcept {
    return u >= thr - tol;
}

/// Same, for a CyclePosition.
[[nodiscard]] inline bool reached(const CyclePosition& p,
                                   Real thr) noexcept {
    return reached(p.u, thr, p.tol);
}

/// Time at which the period `p.k` (+ `cycles`) reaches the fraction
/// `thr`: offset + (k + cycles + thr) * T.
[[nodiscard]] inline Real time_of(const CyclePosition& p, Real offset,
                                   Real T, Real thr,
                                   Real cycles = Real{0}) noexcept {
    return offset + (p.k + cycles + thr) * T;
}

/// First instant after `t` at which a CONSTANT threshold set of a
/// carrier (period `T`, starting at `offset`) is crossed, or the next
/// period start.
template <class Thresholds>
[[nodiscard]] inline Real next_constant_crossing(
    Real t, Real offset, Real T, const Thresholds& thresholds) noexcept {
    const auto p = cycle_position(t, offset, T);
    Real best = time_of(p, offset, T, Real{1});
    for (Real thr : thresholds) {
        if (thr > Real{0} && thr < Real{1} &&
            !reached(p, thr)) {
            best = std::min(best, time_of(p, offset, T, thr));
        }
    }
    return best;
}

/// First instant after `t`, inside the current carrier period, at
/// which the position reaches the TIME-VARYING threshold `thr(tau)`;
/// +inf if it does not within this period. Bisection to the last
/// representable time, returning the side on which the threshold is
/// reached, so the mask evaluated there is already the post-edge one.
/// Assumes the threshold moves slower than the carrier ramp
/// (|d thr / d tau| < 1 / T; for a sine reference of index M at
/// f_mod that is f_carrier > pi * M * f_mod).
template <class ThrFn>
[[nodiscard]] inline Real next_varying_crossing(
    Real t, Real offset, Real T, const CyclePosition& p,
    ThrFn&& thr) {
    const Real t_end = time_of(p, offset, T, Real{1});
    const auto g = [&](Real tau) {
        const Real u = (tau - offset) / T - p.k;
        return u - thr(tau) + p.tol;   // >= 0: reached
    };
    if (g(t) >= Real{0}) return std::numeric_limits<Real>::infinity();
    if (g(t_end) < Real{0}) return std::numeric_limits<Real>::infinity();
    Real lo = t;
    Real hi = t_end;
    for (int it = 0; it < 200; ++it) {
        const Real mid = lo + Real{0.5} * (hi - lo);
        if (!(mid > lo && mid < hi)) break;
        if (g(mid) >= Real{0}) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return hi;
}

}  // namespace pulsim::pwm_phase
