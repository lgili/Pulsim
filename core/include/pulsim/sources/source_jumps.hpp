#pragma once

// =============================================================================
// Pulsim — jump instants of the built-in piecewise-constant sources
// =============================================================================
//
// add-discontinuity-restart-rule, task 3. The fixed engine restarts
// (three SDIRK(2,2) steps) after a source jump, and the step
// that ENDS on a jump reads the source's left limit. Jumps come from
// PWM voltage sources (period start and duty edge) and pulse voltage
// and current sources with a zero rise or fall time; `b_extra_fn`, the
// sine sources and the table current source (continuous by validation:
// a periodic table must end where it starts) contribute none. Edges on a grid point are snapped exactly as
// `value_at` snaps them (pwm_phase.hpp), so "a jump at t" here and
// "the value changes at t" there are the same statement.

#include "pulsim/models/controlled_sources.hpp"
#include "pulsim/models/pulse_voltage_source.hpp"
#include "pulsim/models/pwm_voltage_source.hpp"
#include "pulsim/numeric/pwm_phase.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/topology/graph.hpp"

#include <cmath>
#include <limits>

namespace pulsim::sources {

namespace detail {

/// Jumps of one source over [t_a, t_b) and at t: `fam(cb)` calls
/// cb(offset, T, theta, all_n) for each periodic family
/// offset + (n + theta)·T (n >= 0, or every n when all_n), and
/// cb(tau, 0, 0, false) for a single instant tau.
struct JumpCount {
    Real le = 0;   // jumps tau <= t (snapped), up to a constant
    Real at = 0;   // jumps tau == t (snapped)
};

inline JumpCount count_family(Real t, Real offset, Real T, Real theta,
                              bool all_n) noexcept {
    JumpCount c;
    if (!(T > Real{0})) {   // single instant `offset`
        const Real tol = pwm_phase::kSnapUlps *
            std::numeric_limits<Real>::epsilon() *
            std::max(Real{1e-300}, std::abs(t) + std::abs(offset));
        c.le = t >= offset - tol ? 1 : 0;
        c.at = std::abs(t - offset) <= tol ? 1 : 0;
        return c;
    }
    const auto pos = pwm_phase::cycle_position(t, offset, T);
    if (!all_n && pos.k < Real{0}) return c;   // before the first one
    c.le = pos.k + (pwm_phase::reached(pos, theta) ? 1 : 0);
    c.at = std::abs(pos.u - theta) <= pos.tol ? 1 : 0;
    return c;
}

template <class Fn>
void for_each_family(const pwl::DevicePool& pool, Index branch_id, Fn&& fn) {
    using K = pwl::DevicePool::StoredKind;
    const auto kind = pool.kind_of(branch_id);
    if (kind == K::PWMVoltageSource) {
        const auto& p = pool.pwm_voltage_source_params(branch_id);
        if (p.frequency > Real{0} && p.duty > Real{0} && p.duty < Real{1}) {
            const Real T = Real{1} / p.frequency;
            fn(p.phase, T, Real{0}, true);
            fn(p.phase, T, p.duty, true);
        }
    } else if (kind == K::PulseVoltageSource || kind == K::PulseCurrentSource) {
        // The pulse current source is the same trapezoid (and edge rules).
        const auto& p = kind == K::PulseVoltageSource
            ? pool.pulse_voltage_source_params(branch_id)
            : pool.pulse_current_source_params(branch_id).w;
        const Real fall_at = p.rise_time + p.pulse_width;
        if (p.period > Real{0}) {
            if (p.rise_time == Real{0}) fn(p.t_start, p.period, Real{0}, false);
            if (p.fall_time == Real{0} && fall_at < p.period)
                fn(p.t_start, p.period, fall_at / p.period, false);
        } else {
            if (p.rise_time == Real{0}) fn(p.t_start, Real{0}, Real{0}, false);
            if (p.fall_time == Real{0}) fn(p.t_start + fall_at, Real{0}, Real{0}, false);
        }
    }
}

}  // namespace detail

/// Does any built-in source jump at an instant tau with
/// t_a <= tau < t_b (a jump AT t_b belongs to the next step)?
inline bool source_jump_in(const pwl::DevicePool& pool,
                           const topology::Graph& graph,
                           Real t_a, Real t_b) {
    for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
        if (graph.branch(b_id).kind != topology::BranchKind::Source) continue;
        bool hit = false;
        detail::for_each_family(pool, b_id,
            [&](Real off, Real T, Real th, bool all_n) {
                const auto a = detail::count_family(t_a, off, T, th, all_n);
                const auto b = detail::count_family(t_b, off, T, th, all_n);
                if (b.le - a.le + a.at - b.at > Real{0}) hit = true;
            });
        if (hit) return true;
    }
    return false;
}

/// First jump of any built-in source strictly after `t` (snapped like
/// `value_at`); +inf when none. add-fixed-step-event-splitting locates
/// the jumps that fall inside a step with it.
inline Real next_source_jump_after(const pwl::DevicePool& pool,
                                   const topology::Graph& graph, Real t) {
    Real best = std::numeric_limits<Real>::infinity();
    for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
        if (graph.branch(b_id).kind != topology::BranchKind::Source) continue;
        detail::for_each_family(pool, b_id,
            [&](Real off, Real T, Real th, bool all_n) {
                if (!(T > Real{0})) {
                    if (detail::count_family(t, off, T, th, all_n).le == Real{0})
                        best = std::min(best, off);
                    return;
                }
                const auto pos = pwm_phase::cycle_position(t, off, T);
                Real n = pos.k + (pwm_phase::reached(pos, th) ? Real{1} : Real{0});
                if (!all_n && n < Real{0}) n = Real{0};
                best = std::min(best, off + (n + th) * T);
            });
    }
    return best;
}

/// Does the source on `branch_id` jump exactly at `t`?
inline bool source_jumps_at(const pwl::DevicePool& pool, Index branch_id,
                            Real t) {
    bool hit = false;
    detail::for_each_family(pool, branch_id,
        [&](Real off, Real T, Real th, bool all_n) {
            if (detail::count_family(t, off, T, th, all_n).at > Real{0}) hit = true;
        });
    return hit;
}

}  // namespace pulsim::sources
