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
// This adapter registers on the scheduler's hooks and drives the feed's hooks:
//   * every state is translated with `recover_full` under its own mask;
//   * the step's start is the last step's end, or the right limit of an event
//     at that instant;
//   * an event reported before the accept of the step it ends is held and
//     forwarded right after that step (the feed then has its sample at t: an
//     event row there); one at the last accepted instant goes at once (a
//     zero-length gate edge, the landing of a resume, reported at the
//     snapshot's instant when the oracle's edge is an ulp off it); two reports for one
//     instant are one event (the first left limit, the last right limit);
//   * events before `start` (the ticks at t = 0) are not the record's (row 0
//     is the state after them), as for the loss port.
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
            xs_ = full_(t_last_, x0, m0);
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
                hr_ = full_(t, xr, mr);
                hmr_ = mr;
            } else {
                if (held_) forward_(false);   // not a step's end: the feed decides
                ht_ = t;
                hl_ = full_(t, xl, ml);
                hml_ = ml;
                hr_ = full_(t, xr, mr);
                hmr_ = mr;
                held_ = true;
            }
            if (t == t_last_) forward_(true);
        });
        h.on_accept([this](Real t0, Real t, const Vector& x, const MaskT& m) {
            // an instant this step starts from that was not the last accepted
            // one to the bit (a resume landing snapped onto a source corner)
            if (held_ && ht_ <= t0) forward_(true);
            stepped_ = true;
            Vector xe = full_(t, x, m);
            out_.step(t0, xs_, ms_, t, xe, m);
            xs_.swap(xe);
            ms_ = m;
            t_last_ = t;
            if (held_ && ht_ == t) forward_(true);
        });
    }

private:
    AdapterT& ad_;
    events::StepHooks& out_;
    Vector xs_, hl_, hr_;
    MaskT ms_, hml_, hmr_;
    Real t_last_ = 0, ht_ = 0;
    bool started_ = false, held_ = false, stepped_ = false;

    void forward_(bool starts_next) {
        out_.event(ht_, hl_, hml_, hr_, hmr_);
        if (starts_next) {   // the next step starts from the right limit
            xs_ = hr_;
            ms_ = hmr_;
        }
        held_ = false;
    }

    /// x under mask m (the hooks run after an event moved the adapter to the
    /// new mask; a left limit is read under the old), as the loss port does.
    [[nodiscard]] Vector full_(Real t, const Vector& x, const MaskT& m) {
        if (ad_.current_mask() == m) return ad_.recover_full(t, x);
        const MaskT keep = ad_.current_mask();
        ad_.set_mask(m);
        Vector f = ad_.recover_full(t, x);
        ad_.set_mask(keep);
        return f;
    }
};

}  // namespace pulsim::dsed
