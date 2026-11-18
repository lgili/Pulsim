#pragma once

// =============================================================================
// Pulsim — the run's one queue of time events (add-unified-event-queue)
// =============================================================================
//
// Every scheduled instant at which the drive, the observers or the run
// change: control ticks, gate edges, source breakpoints, output samples
// and the end of the run. Each source answers "the next event of mine";
// the queue keeps ONE pending event per source and merges them.
//
// Rules (design.md D1, D2):
//   * Periodic sources compute t = (t0 + offset) + k*period from an
//     integer k, never by accumulation: event k is the same double on
//     every engine and after any number of periods.
//   * A periodic source fires at every k >= 0 with t_k < t_end (half-open
//     run, the fix-control-tick-semantics rule): ceil((t_end - t0 -
//     offset) / period) events when that quotient is not within rounding
//     of an integer; the t_k < t_end test, not the formula, decides.
//   * Events taken together are returned in Kind order, then in
//     registration order (the order of `Kind` is the order of D2).
//   * Nothing allocates after `start` (setup reserves every buffer).
//
// shortcut: the pending events are scanned linearly, not kept in a heap;
// a run has a handful of sources (gates, a few rates, the end). Upgrade
// to a binary heap if a run ever registers more than ~32 sources.

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace pulsim::events {

/// Kind of a time event; also the order of events at one instant (D2).
enum class Kind : std::uint8_t {
    OutputSample = 0,
    ControlTick = 1,
    GateEdge = 2,
    SourceBreakpoint = 3,
    EndOfRun = 4,
};

/// Bit of `kind` in a kind mask (`take_due`).
[[nodiscard]] constexpr std::uint32_t kind_bit(Kind k) noexcept {
    return std::uint32_t{1} << static_cast<unsigned>(k);
}
inline constexpr std::uint32_t kAllKinds = 0x1Fu;

struct TimeEvent {
    Real t;
    Kind kind;
    std::uint32_t source;    // id returned by add_*
    std::uint64_t ordinal;   // k of a periodic source; else events taken before
};

/// A periodic schedule: events at t0 + offset + k*period, k >= 0.
struct Rate {
    Real period;
    Real offset = Real{0};
    /// The grid's origin: ticks at (origin + offset) + k*period. NaN (the
    /// default) is the run's t0. A RESUMED run passes the original run's
    /// t0 here, so it ticks bit-exactly where the uninterrupted run would,
    /// with the same k (the first k whose instant is >= t0 - start tol).
    Real origin = std::numeric_limits<Real>::quiet_NaN();
};

/// What a tick handler is told (task 4.1): the instant the tick was
/// scheduled for and its integer index. On an engine that lands its
/// ticks (trbdf2) the step time equals `scheduled_t` to the landing
/// tolerance; on the fixed grid the step time is the first grid point at
/// or after it.
struct TickInfo {
    Real scheduled_t;
    std::uint64_t k;
};

/// Control-tick sources of one engine run, registered on the run's queue
/// in this order (source j = rates[j]). `fn(j, info, t, x)` runs at each
/// tick with the step time t and the state there. The tick registration
/// API of add-unified-event-queue: a modulator or a control graph that
/// needs its ticks scheduled by the kernel fills one of these.
struct TickPlan {
    std::vector<Rate> rates;
    std::function<void(std::uint32_t j, const TickInfo& info, Real t,
                       const Vector& x)> fn;
    /// Optional: runs at every observer call BEFORE the ticks (the fixed
    /// engine's label-time graphs and internal drivers, which ran before
    /// the Python tick sources when those were part of the observer).
    std::function<void(Real t, const Vector& x)> before;
};

class EventQueue {
public:
    /// "First event strictly after t", +inf when none.
    using NextFn = std::function<Real(Real)>;

    /// Periodic source (ticks, sample grids). Throws on a period that is
    /// not a positive finite number or a negative / non-finite offset.
    std::uint32_t add_periodic(Kind kind, Rate rate, bool lands = true) {
        if (!(rate.period > Real{0}) || !std::isfinite(rate.period)) {
            throw std::invalid_argument(
                "EventQueue: a period must be a positive finite number of "
                "seconds");
        }
        if (!(rate.offset >= Real{0}) || !std::isfinite(rate.offset)) {
            throw std::invalid_argument(
                "EventQueue: an offset must be a finite number of seconds "
                ">= 0");
        }
        if (std::isinf(rate.origin)) {
            throw std::invalid_argument(
                "EventQueue: a grid origin must be finite (or NaN for the "
                "run's t0)");
        }
        Slot s = make(kind, lands);
        s.periodic = true;
        s.rate = rate;
        return push(std::move(s));
    }

    /// Source with an oracle `next(t)` (first event strictly after t). Its
    /// pending event is fetched lazily by `next_after` and refilled from
    /// the event's own time when taken.
    std::uint32_t add_function(Kind kind, NextFn next, bool lands = true) {
        Slot s = make(kind, lands);
        s.next = std::move(next);
        return push(std::move(s));
    }

    /// Source whose events a locator finds (`locate`), e.g. the bisection
    /// of an opaque gate function.
    std::uint32_t add_located(Kind kind, bool lands = true) {
        return push(make(kind, lands));
    }

    /// One event at `t` (the end of the run).
    std::uint32_t add_once(Kind kind, Real t, bool lands = true) {
        Slot s = make(kind, lands);
        s.once = true;
        s.once_t = t;
        return push(std::move(s));
    }

    /// Arms every source for a run over [t0, t_end). Resets the counters.
    /// A periodic source with an `origin` starts at its first instant
    /// >= t0 - t0_tol (t0_tol: the engine's take tolerance, so a tick the
    /// previous run left out at its t_end is the resumed run's first).
    void start(Real t0, Real t_end, Real t0_tol = Real{0}) {
        t_end_ = t_end;
        taken_buf_.resize(slots_.size());   // the only allocation, here
        for (auto& s : slots_) {
            s.k = 0;
            s.n_taken = 0;
            s.valid = false;
            s.pending = kNone;
            if (s.periodic) {
                if (std::isnan(s.rate.origin)) {
                    s.base = t0 + s.rate.offset;
                } else {
                    s.base = s.rate.origin + s.rate.offset;
                    s.k = first_k_at_or_after(s, t0 - t0_tol);
                }
                set_periodic(s);
            } else if (s.once) {
                s.pending = s.once_t;
                s.valid = true;
            }
        }
    }

    [[nodiscard]] std::size_t size() const noexcept { return slots_.size(); }
    [[nodiscard]] Kind kind(std::uint32_t id) const { return slots_.at(id).kind; }

    /// Time of event k of a periodic source: (t0 + offset) + k*period.
    [[nodiscard]] Real periodic_time(std::uint32_t id, std::uint64_t k) const {
        const Slot& s = slots_.at(id);
        return s.base + static_cast<Real>(k) * s.rate.period;
    }

    /// The source's pending event time as stored (+inf when it has none;
    /// may be stale for a function or located source, see `next_after`).
    [[nodiscard]] Real pending(std::uint32_t id) const {
        const Slot& s = slots_.at(id);
        return s.valid ? s.pending : kNone;
    }

    /// The source's first event after `t`, refreshing a stale one: a
    /// function source whose pending event is invalid or not after `t`
    /// asks its oracle `next(t)`; a located source drops an event not
    /// after `t`. Periodic and one-shot sources are returned as they are
    /// (their events are consumed by `take_due` only).
    Real next_after(std::uint32_t id, Real t) {
        Slot& s = slots_.at(id);
        if (s.next) {
            if (!s.valid || s.pending <= t) {
                s.pending = s.next(t);
                s.valid = true;
            }
        } else if (!s.periodic && !s.once && s.valid && s.pending <= t) {
            s.valid = false;
        }
        return s.valid ? s.pending : kNone;
    }

    /// Earliest pending event of `kind` (+inf when none), without refresh.
    [[nodiscard]] Real next_of(Kind k) const noexcept {
        Real best = kNone;
        for (const auto& s : slots_) {
            if (s.kind == k && s.valid && s.pending < best) {
                best = s.pending;
            }
        }
        return best;
    }

    /// Earliest pending event that must end a step (+inf when none).
    [[nodiscard]] Real next_landing() const noexcept {
        Real best = kNone;
        for (const auto& s : slots_) {
            if (s.lands && s.valid && s.pending < best) {
                best = s.pending;
            }
        }
        return best;
    }

    /// A located source's next event (a locator found it at `t`).
    void locate(std::uint32_t id, Real t) {
        Slot& s = slots_.at(id);
        s.pending = t;
        s.valid = true;
    }

    /// The source's next event changed (a duty register was written): a
    /// function source re-asks its oracle, a located source forgets it.
    /// Periodic and one-shot sources do not depend on any register and
    /// are left as they are.
    void invalidate(std::uint32_t id) {
        Slot& s = slots_.at(id);
        if (!s.periodic && !s.once) {
            s.valid = false;
        }
    }
    void invalidate(Kind k) noexcept {
        for (auto& s : slots_) {
            if (s.kind == k && !s.periodic && !s.once) {
                s.valid = false;
            }
        }
    }

    /// Takes every pending event of the kinds in `kinds` that is due at
    /// `t` (`t >= pending - tol`), one per source, in Kind order and then
    /// registration order; each source is refilled from its event's time
    /// (a periodic one with k + 1). The span points into a buffer owned
    /// by the queue, valid until the next call.
    std::span<const TimeEvent> take_due(Real t, Real tol,
                                        std::uint32_t kinds = kAllKinds) {
        std::size_t n = 0;
        for (std::uint32_t id = 0; id < slots_.size(); ++id) {
            Slot& s = slots_[id];
            if (!s.valid || !(kinds & kind_bit(s.kind))
                || !(t >= s.pending - tol)) {
                continue;
            }
            TimeEvent ev{s.pending, s.kind, id,
                         s.periodic ? s.k : s.n_taken};
            // Insertion by (kind, id); ids arrive in increasing order.
            std::size_t j = n++;
            while (j > 0 && taken_buf_[j - 1].kind > ev.kind) {
                taken_buf_[j] = taken_buf_[j - 1];
                --j;
            }
            taken_buf_[j] = ev;
            ++s.n_taken;
            if (s.periodic) {
                ++s.k;
                set_periodic(s);
            } else if (s.next) {
                // An oracle that does not move past its own event is
                // asked again from the next `next_after`, not re-taken.
                s.pending = s.next(ev.t);
                s.valid = s.pending > ev.t;
            } else {
                s.valid = false;
                s.pending = kNone;
            }
        }
        return {taken_buf_.data(), n};
    }

    /// Events taken from a source since `start`.
    [[nodiscard]] std::uint64_t taken(std::uint32_t id) const {
        return slots_.at(id).n_taken;
    }
    [[nodiscard]] std::uint64_t taken(Kind k) const noexcept {
        std::uint64_t n = 0;
        for (const auto& s : slots_) {
            if (s.kind == k) {
                n += s.n_taken;
            }
        }
        return n;
    }

private:
    static constexpr Real kNone = std::numeric_limits<Real>::infinity();

    struct Slot {
        Kind kind = Kind::GateEdge;
        bool lands = true;
        bool periodic = false;
        bool once = false;
        bool valid = false;
        Rate rate{Real{1}, Real{0}};
        Real base = Real{0};
        Real once_t = Real{0};
        Real pending = kNone;
        std::uint64_t k = 0;
        std::uint64_t n_taken = 0;
        NextFn next;
    };

    static Slot make(Kind kind, bool lands) {
        Slot s;
        s.kind = kind;
        s.lands = lands;
        return s;
    }

    std::uint32_t push(Slot s) {
        slots_.push_back(std::move(s));
        return static_cast<std::uint32_t>(slots_.size() - 1);
    }

    // Smallest k >= 0 with base + k*period >= t (the same expression as
    // set_periodic, so the comparison is on the instants actually used).
    static std::uint64_t first_k_at_or_after(const Slot& s, Real t) {
        const Real x = std::ceil((t - s.base) / s.rate.period);
        std::uint64_t k = x > Real{0} ? static_cast<std::uint64_t>(x) : 0;
        const auto at = [&](std::uint64_t j) {
            return s.base + static_cast<Real>(j) * s.rate.period;
        };
        while (k > 0 && at(k - 1) >= t) --k;
        while (at(k) < t) ++k;
        return k;
    }

    void set_periodic(Slot& s) const noexcept {
        const Real tk = s.base + static_cast<Real>(s.k) * s.rate.period;
        s.valid = tk < t_end_;
        s.pending = s.valid ? tk : kNone;
    }

    std::vector<Slot> slots_;
    std::vector<TimeEvent> taken_buf_;
    Real t_end_ = kNone;
};

}  // namespace pulsim::events
