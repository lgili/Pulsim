#pragma once

// =============================================================================
// Pulsim — the exact engine's in-loop recorder feed (add-probes-and-recorders
// R5 / task 0b.11)
// =============================================================================
//
// `record::LoopFeed` reads FULL MNA vectors and wants one
// `step(t0, x_start, m_start, t, x_end, m_end)` per accepted step and an
// `event(t, left, right)` per settled instant. The exact engine's scheduler
// (PEDSimulator) drives `StepHooks` with its REDUCED state: `start`, then per
// accepted step `accepted(t0, t, x_end, mask)` (x_end the pre-event state when
// an event ended the step), and `event(t, ...)` from inside the step it ends,
// BEFORE that step's `accepted`.
//
// This adapter registers on the scheduler's hooks and drives the feed's hooks
// with the stored arrays' conventions (python/dsed_bindings.cpp
// reconstruct_full_states / reconstruct_right_limits):
//   * every state is translated with `recover_full` under its own mask; a
//     step's END reads its sources' left limit (a source jumping at t is read
//     at t − h/2: the step integrated the old piece), a right limit reads them
//     at t;
//   * the step's start is the last step's end, or the right limit of an event
//     at that instant;
//   * an event reported before the accept of the step it ends is held and
//     forwarded right after that step (the feed then has its sample at t: an
//     event row there); one reported a few ulps past that step's end (a gate
//     edge the step stopped at to rounding) is that instant; one at the last
//     accepted instant goes at once (a zero-length gate edge, the landing of a
//     resume, reported at the snapshot's instant when the oracle's edge is an
//     ulp off it); two reports for one instant are one event (the first left
//     limit, the last right limit);
//   * a source corner that changes no mask fires no event hook (fire_event_
//     drops same-mask gate edges) but moves the source-driven nodes: when a
//     step's end read at the sources' left and right limits differs, it is an
//     event too, forwarded when the next step starts (none at the run's end:
//     its new piece is never integrated, as in the stored arrays);
//   * events before `start` (the ticks at t = 0) are not the record's (row 0
//     is the state after them), as for the loss port.
// The feed is given this adapter's own start state for every step, so its jump
// check (`n_jumps`) cannot see an unreported change of the scheduler's state:
// what this relies on is that every such change is an event hook or one of the
// above (state projection is never set from Python; the junction loop's kick
// is refused by the binding).
// A pure observer of the scheduler's states; no allocation is made when no
// feed is attached (the scheduler's hooks are then the loss port's or none).

#include "pulsim/dsed/time_eps.hpp"
#include "pulsim/events/step_hooks.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/switch_state.hpp"

namespace pulsim::dsed {

template <class AdapterT>
class ExactFeedAdapter {
public:
    using MaskT = topology::SwitchStateMask;

    /// `out`: the feed's hooks (record::LoopFeed::hooks()). `t_start`: the
    /// scheduler's clock at `start` (0, or a resume's instant).
    ExactFeedAdapter(AdapterT& ad, events::StepHooks& out, Real t_start)
        : ad_{ad}, out_{out}, t_last_{t_start} {}

    void attach(events::StepHooks& h) {
        h.on_start([this](const Vector& x0, const MaskT& m0) {
            xs_ = full_(t_last_, x0, m0, Real{0});
            ms_ = m0;
            out_.start(xs_, m0);
            started_ = true;
        });
        h.on_event([this](Real t_ev, const Vector& xl, const MaskT& ml, const Vector& xr,
                          const MaskT& mr) {
            if (!started_) return;
            // a resume's landing: the oracle's edge time may sit an ulp off the
            // snapshot's instant; the record (and row 0 of the stored arrays)
            // has it at the snapshot's instant
            const Real t = !stepped_ && near_time(t_ev, t_last_) ? t_last_ : t_ev;
            if (held_ && t == ht_) {   // a second report of the same instant
                hr_ = full_(t, xr, mr, Real{0});
                hmr_ = mr;
            } else {
                if (held_) forward_(false);   // not a step's end: the feed decides
                ht_ = t;
                hxl_ = xl;
                hml_ = ml;
                hl_valid_ = false;
                hr_ = full_(t, xr, mr, Real{0});
                hmr_ = mr;
                held_ = true;
            }
            if (t == t_last_) forward_(true);
        });
        h.on_accept([this](Real t0, Real t, const Vector& x, const MaskT& m) {
            // an instant this step starts from that was not the last accepted
            // one to the bit (a resume landing snapped onto a source corner),
            // or the source corner the last step ended on
            if (held_ && ht_ <= t0) forward_(true);
            // an edge the step stopped at to rounding: the event is at t
            if (held_ && ht_ != t && near_time(ht_, t)) ht_ = t;
            stepped_ = true;
            lb_ = Real{0.5} * (t - t0);
            Vector xe = full_(t, x, m, lb_);
            out_.step(t0, xs_, ms_, t, xe, m);
            xs_.swap(xe);
            ms_ = m;
            t_last_ = t;
            if (held_ && ht_ == t) {
                hl_ = xs_;   // the step's end: the left limit at its sources' too
                hl_valid_ = true;
                forward_(true);
            } else if (!held_ && ad_.has_dynamic_sources()) {
                Vector r = full_(t, x, m, Real{0});
                if ((r.array() != xs_.array()).any()) {   // a same-mask source corner
                    ht_ = t;
                    hl_ = xs_;
                    hml_ = m;
                    hl_valid_ = true;
                    hr_.swap(r);
                    hmr_ = m;
                    held_ = true;   // forwarded when the next step starts from t
                }
            }
        });
    }

private:
    AdapterT& ad_;
    events::StepHooks& out_;
    Vector xs_, hl_, hr_, hxl_;
    MaskT ms_, hml_, hmr_;
    Real t_last_ = 0, ht_ = 0, lb_ = 0;
    Index n_full_ = -1;
    bool started_ = false, held_ = false, stepped_ = false, hl_valid_ = false;

    void forward_(bool starts_next) {
        // a left limit at the last accepted instant reads the sources as that
        // step's end did; before any step (row 0) at t
        if (!hl_valid_) hl_ = full_(ht_, hxl_, hml_, stepped_ && ht_ == t_last_ ? lb_ : Real{0});
        out_.event(ht_, hl_, hml_, hr_, hmr_);
        if (starts_next) {   // the next step starts from the right limit
            xs_ = hr_;
            ms_ = hmr_;
        }
        held_ = false;
        hl_valid_ = false;
    }

    /// x under mask m (the hooks run after an event moved the adapter to the
    /// new mask; a left limit is read under the old), as the loss port does.
    /// `left_by` > 0 reads a source jumping at t at its left limit.
    [[nodiscard]] Vector full_(Real t, const Vector& x, const MaskT& m, Real left_by) {
        const bool same = ad_.current_mask() == m;
        const MaskT keep = ad_.current_mask();
        if (!same) ad_.set_mask(m);
        if (n_full_ < 0) n_full_ = ad_.recover_full(t, x).size();
        Vector f(n_full_);
        ad_.recover_full_into(t, x, f, left_by);
        if (!same) ad_.set_mask(keep);
        return f;
    }
};

}  // namespace pulsim::dsed
