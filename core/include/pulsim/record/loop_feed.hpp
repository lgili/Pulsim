#pragma once

// =============================================================================
// Pulsim — in-loop feeding of recorders from the step hooks (task 0b.6)
// =============================================================================
//
// `LoopFeed` owns a run's recorders and feeds them from `StepHooks` while
// the engine runs, so nothing is read back from the stored arrays: a run
// with `store_every = n` (or one that stores nothing a recorder needs)
// still gives statistics over EVERY accepted step.
//
// Readings are functions of the state vector a hook delivers:
//   reading = (Σ c_k x[col_k]) / div            (linear: v(n), v(a,b), a
//                                                state column, a resistor)
//           = lin_a(x) · lin_b(x)               (product: p() of those)
// A reading that is not of this form (a switched device's law under the
// mask, a Newton device) is not offered here; the caller keeps the
// stored-array pass for that run (and says so).
//
// Per accepted step [t0, t] (`StepHooks::step`):
//   * the first step seeds the record with the start state at t0;
//   * a start state that differs from the previous step's end (bitwise in
//     x, or in the mask) is a jump the engine made at t0: an event row with
//     the start state as its right limit (resolution `Settled`, `n_jumps`),
//     and the line of the step starts from it;
//   * the end state is the sample at t.
// Commutations (`StepHooks::on_event`):
//   * fixed engine (`Events::Bracket`): the pair brackets the commutation
//     within one step; recorded with `event_bracket` (left at the step's
//     start, right at its end, resolution `WithinDt`; the line is not moved);
//   * variable-step engine (`Events::Settled`): the right side is the
//     settled post-event state at the instant: an event row there, the next
//     line starts from it. Before the step hook that lands t (a diode
//     landing) it waits for that sample; after it (t == the last sample) it
//     applies at once; before a hook step that ALSO covers the committed
//     post-event probe (a landed gate edge: [t0, t + dtp] as one step) the
//     event's left side is the sample at t. A step solved under a new mode
//     with no settled report at its start is bracketed [t0, t] (`WithinDt`).
// Reports the feed cannot place are counted (`n_dropped`), never invented.
// A pure observer: the callbacks read their arguments only. No allocation
// per step for keep = none / decimate-free statistics once `reserve` has
// sized the period windows; kept rows and event rows grow their vectors.

#include "pulsim/events/step_hooks.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/record/recorder.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pulsim::record {

struct LinearReading {
    std::vector<Index> cols;  ///< state columns (no ground: the caller drops it)
    std::vector<Real> coef;
    Real div = Real{1};

    [[nodiscard]] Real eval(const Vector& x) const noexcept {
        Real s = Real{0};
        for (Size k = 0; k < cols.size(); ++k) s += coef[k] * x[cols[k]];
        return div == Real{1} ? s : s / div;
    }
};

struct StateReading {
    LinearReading a;
    bool product = false;
    LinearReading b;  ///< used when `product`

    [[nodiscard]] Real eval(const Vector& x) const noexcept {
        const Real va = a.eval(x);
        return product ? va * b.eval(x) : va;
    }
};

class LoopFeed {
public:
    LoopFeed(std::vector<std::string> reading_names, std::vector<StateReading> readings,
             std::vector<Recorder> recorders)
        : names_(std::move(reading_names)), readings_(std::move(readings)),
          recs_(std::move(recorders)) {
        if (names_.size() != readings_.size()) {
            throw std::invalid_argument("LoopFeed: one reading per name");
        }
        for (const auto& r : readings_) {
            if (r.a.cols.size() != r.a.coef.size() || r.b.cols.size() != r.b.coef.size()) {
                throw std::invalid_argument("LoopFeed: a reading's columns and coefficients differ");
            }
        }
        for (auto& r : recs_) r.bind(names_);
        row_.resize(names_.size());
    }

    LoopFeed(const LoopFeed&) = delete;
    LoopFeed& operator=(const LoopFeed&) = delete;

    /// Pre-size kept rows for `n_samples` and the period windows up to t_last.
    void reserve(Size n_samples, Real t_last, Size n_events = 0) {
        for (auto& r : recs_) {
            r.reserve(n_samples, t_last);
            if (n_events > 0) r.reserve_events(n_events);
        }
    }

    /// How the engine reports a commutation: the fixed engine BRACKETS it
    /// (`on_event` pairs a step-start state with the end of the first step
    /// under the new stamp: `event_bracket`, resolution WithinDt); the
    /// variable-step engine SETTLES it at the instant (`on_event`'s right
    /// side is the settled post-event state: an event row at t with
    /// resolution Settled, and the next step's line starts from it).
    enum class Events : std::uint8_t { Bracket, Settled };

    /// Register on `hooks` (which must outlive the run, and this object it).
    void attach(events::StepHooks& hooks, Events kind) {
        if (attached_) throw std::logic_error("LoopFeed: already attached (every sample would repeat)");
        hooks.on_step([this](Real t0, const Vector& xs, const topology::SwitchStateMask& ms,
                             Real t, const Vector& xe, const topology::SwitchStateMask& me) {
            on_step(t0, xs, ms, t, xe, me);
        });
        settled_ = kind == Events::Settled;
        hooks.on_event([this](Real t, const Vector& xl, const topology::SwitchStateMask&,
                              const Vector& xr, const topology::SwitchStateMask& mr) {
            on_event(t, xl, xr, mr);
        });
        pending_.resize(names_.size());
        pending_left_.resize(names_.size());
        attached_ = true;
    }

    /// End of the run (`Keep::Times` rows at the last sample).
    void finish() {
        for (auto& r : recs_) r.flush();
    }

    /// `attach` on the feed's own hooks (the Python binding's route: the
    /// hooks then live exactly as long as the feed).
    void attach_own(bool bracket_events) {
        attach(hooks_, bracket_events ? Events::Bracket : Events::Settled);
    }
    [[nodiscard]] events::StepHooks& hooks() noexcept { return hooks_; }
    [[nodiscard]] bool attached() const noexcept { return attached_; }
    [[nodiscard]] Size n_steps() const noexcept { return n_steps_; }
    [[nodiscard]] Size n_jumps() const noexcept { return n_jumps_; }
    [[nodiscard]] Size n_events() const noexcept { return n_events_; }
    /// `on_event` reports the feed could not place (an instant that is not
    /// the last or the next sample): counted, never fabricated.
    [[nodiscard]] Size n_dropped() const noexcept { return n_dropped_; }
    [[nodiscard]] const std::vector<Recorder>& recorders() const noexcept { return recs_; }

private:
    std::vector<std::string> names_;
    std::vector<StateReading> readings_;
    std::vector<Recorder> recs_;
    events::StepHooks hooks_;
    std::vector<Real> row_, pending_, pending_left_;
    Vector x_last_;
    topology::SwitchStateMask mask_last_, mask_pending_;
    bool started_ = false, has_pending_ = false, attached_ = false, settled_ = false;
    bool event_at_last_ = false;  // a settled event row at the last sample
    Real t_last_ = 0, t_pending_ = 0;
    Size n_steps_ = 0, n_jumps_ = 0, n_events_ = 0, n_dropped_ = 0;

    void eval(const Vector& x, std::vector<Real>& out) const noexcept {
        for (Size j = 0; j < readings_.size(); ++j) out[j] = readings_[j].eval(x);
    }

    static bool same(const Vector& a, const Vector& b) noexcept {
        if (a.size() != b.size()) return false;
        for (Index k = 0; k < a.size(); ++k) {
            if (!(a[k] == b[k]) && !(a[k] != a[k] && b[k] != b[k])) return false;
        }
        return true;
    }

    void settled_event(const std::vector<Real>& right) {
        for (auto& r : recs_) r.event(right, Recorder::Resolution::Settled);
        event_at_last_ = true;
        ++n_events_;
    }

    void on_event(Real t, const Vector& xl, const Vector& xr,
                  const topology::SwitchStateMask& mr) {
        if (settled_ && !started_) {
            ++n_dropped_;
            return;
        }
        if (settled_ && t == t_last_) {  // after the sample at t
            eval(xr, row_);
            settled_event(row_);
            mask_last_ = mr;  // the mode the next step runs in
            return;
        }
        if (has_pending_) ++n_dropped_;  // two reports for one step: keep the last
        eval(xr, pending_);
        eval(xl, pending_left_);
        mask_pending_ = mr;
        t_pending_ = t;
        has_pending_ = true;
    }

    void on_step(Real t0, const Vector& xs, const topology::SwitchStateMask& ms, Real t,
                 const Vector& xe, const topology::SwitchStateMask& me) {
        if (!started_) {
            eval(xs, row_);
            for (auto& r : recs_) r.sample(t0, row_);
            x_last_ = xe;  // sizes the buffer once
            t_last_ = t0;
            started_ = true;
        } else if (settled_ && has_pending_ && t_last_ < t_pending_ && t_pending_ < t) {
            // trbdf2 committed [t0, t_ev] and then the post-event probe
            // [t_ev, t] as ONE hook step (a landed gate edge / breakpoint
            // with refresh_after_event): the left limit at t_ev is the
            // event's committed side, the settled right side starts the line
            has_pending_ = false;
            for (auto& r : recs_) r.sample(t_pending_, pending_left_);
            settled_event(pending_);
            mask_last_ = mask_pending_;
        } else if (!event_at_last_ && (!(ms == mask_last_) || !same(xs, x_last_))) {
            // the engine restarted from another state than it committed
            eval(xs, row_);
            for (auto& r : recs_) r.event(row_, Recorder::Resolution::Settled);
            ++n_jumps_;
        }
        const bool had_event = event_at_last_;
        event_at_last_ = false;
        eval(xe, row_);
        if (has_pending_ && !settled_) {
            for (auto& r : recs_) r.event_bracket(t, pending_);
            has_pending_ = false;
            ++n_events_;
        } else if (settled_ && !had_event && n_steps_ > 0 && !(me == mask_last_)) {
            // the variable-step engine solved this step under a new mode
            // without a settled report at its start: bracket [t0, t]
            for (auto& r : recs_) r.event_bracket(t, row_);
            ++n_events_;
        }
        for (auto& r : recs_) r.sample(t, row_);
        x_last_ = xe;  // same size: no allocation
        mask_last_ = me;
        t_last_ = t;
        ++n_steps_;
        if (has_pending_ && settled_) {
            has_pending_ = false;
            if (t_pending_ == t) {
                settled_event(pending_);
                mask_last_ = mask_pending_;
            } else {
                ++n_dropped_;
            }
        }
    }
};

}  // namespace pulsim::record
