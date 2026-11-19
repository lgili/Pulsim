#pragma once

// =============================================================================
// Pulsim — accepted-step and commutation callbacks (add-unified-event-queue)
// =============================================================================
//
// The engines call `StepHooks::accepted(t_prev, t, x, mask)` once per
// accepted (committed) step: the state at the end of the step and the
// switch mask the step was solved under (gates and diodes together). A
// listener registered with `on_accept` sees every such step; one registered
// with `on_commutation` sees every instant at which that mask changed:
//
//   t         = the instant (the end of the last step under `before`, which
//               is the start of the first step under `after`)
//   x_before  = the state at t, solved under `before`
//   x_after   = the state at the end of the first step under `after`
//
// which is the reading rule of add-in-kernel-event-losses (turn-on: voltage
// before, current after; turn-off the reverse). A mask that changes and
// changes back inside one step is not a commutation here (the engine never
// solved under the intermediate mask).
//
// Granularity is the engine's accepted step. trbdf2 lands its edges and
// crossings, so `t` is the event instant. The fixed engine reports one
// accepted step per grid step with the mask it ended under: an edge or a
// crossing that the event split places INSIDE a step is reported at the
// step's start (up to dt early) with the state there, the same shortcut as
// its D4 re-initialisation record. The static (no dynamic device) path
// arms at its first solve.
//
// Order: listeners run in registration order; `on_accept` listeners before
// `on_commutation` listeners of the same step. Nothing allocates after
// `start` (the previous state and mask are sized there). With no listener
// `accepted` returns at once: an engine run without hooks is unchanged.

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <functional>
#include <vector>

namespace pulsim::pwl {
class StatefulHistories;
}

namespace pulsim::events {

struct Commutation {
    Real t;
    const topology::SwitchStateMask& before;
    const topology::SwitchStateMask& after;
    const Vector& x_before;
    const Vector& x_after;
};

class StepHooks {
public:
    using AcceptFn = std::function<void(Real t_prev, Real t, const Vector& x,
                                        const topology::SwitchStateMask& mask)>;
    using CommutationFn = std::function<void(const Commutation&)>;

    void on_accept(AcceptFn f) { accept_.push_back(std::move(f)); }
    void on_commutation(CommutationFn f) { commut_.push_back(std::move(f)); }

    // refactor-unified-transient-stepper slice 3a: called by the step
    // skeleton, the only driver of both engines (DECISIONS 13, 17, 20).
    // probes R1: once per accepted step, after commit, with the state the
    // step started from and its mask beside the end state and the mask the
    // step was stamped under. On the fixed engine `x_start` is the
    // committed state at t_prev and `mask_start` the previous step's stamp
    // (the LEFT limit: pwl solves no separate right limit at a grid-point
    // edge; its re-initialisation is inside the step), the first step's the
    // run's initial stamp.
    using StepFn = std::function<void(Real t_prev, const Vector& x_start,
                                      const topology::SwitchStateMask& mask_start,
                                      Real t, const Vector& x_end,
                                      const topology::SwitchStateMask& mask_end)>;
    // device interface DECISIONS #31: right after the stateful histories'
    // commit of an accepted step, `t` the accepted time.
    using DeviceStateFn = std::function<void(Real t, const Vector& x,
                                             const pwl::StatefulHistories&)>;
    // probes R2 (slice 3c, DECISIONS 17): at an event, after the settle,
    // the two sides of the commutation, each with the mask it was solved
    // under (mask_left != mask_right). TR-BDF2: t is the landed instant,
    // x_left the committed solve there, x_right the settled post-event
    // probe (a trapezoidal step of `h_probe`, femtoseconds to picoseconds:
    // a right limit to that step). The fixed engine solves no right limit
    // at the instant (its events are inside a grid step): t is the step's
    // start, x_left the committed state there under the previous step's
    // stamp, x_right the settled end of the first step under the new
    // stamp, so the pair BRACKETS the commutation within one dt.
    // Not reported (slice 3c): TR-BDF2 gate edges with
    // `refresh_after_event` off, a failed or non-finite probe, or entry
    // into a mode with dependent states (no usable right side); a cascade
    // that hit `max_event_iterations` reports its last, unsettled probe.
    // The fixed engine compares the stamps at a grid step's two ends: a
    // mask that changes and changes back inside one step is not reported,
    // several commutations in one step are one report, and a seeded start
    // (`initial_state`, a snapshot without its stamp) reports nothing at
    // its first step. The fixed engine's static path reports none.
    using EventFn = std::function<void(Real t, const Vector& x_left,
                                       const topology::SwitchStateMask& mask_left,
                                       const Vector& x_right,
                                       const topology::SwitchStateMask& mask_right)>;
    void on_event(EventFn f) { event_.push_back(std::move(f)); }
    [[nodiscard]] bool has_event() const noexcept { return !event_.empty(); }
    void event(Real t, const Vector& x_left, const topology::SwitchStateMask& mask_left,
               const Vector& x_right, const topology::SwitchStateMask& mask_right) const {
        for (const auto& f : event_) f(t, x_left, mask_left, x_right, mask_right);
    }
    void on_step(StepFn f) { step_.push_back(std::move(f)); }
    void on_device_state(DeviceStateFn f) { device_state_.push_back(std::move(f)); }
    [[nodiscard]] bool has_step() const noexcept { return !step_.empty(); }
    void step(Real t_prev, const Vector& x_start,
              const topology::SwitchStateMask& mask_start, Real t,
              const Vector& x_end, const topology::SwitchStateMask& mask_end) const {
        for (const auto& f : step_) f(t_prev, x_start, mask_start, t, x_end, mask_end);
    }
    void device_state(Real t, const Vector& x, const pwl::StatefulHistories& h) const {
        for (const auto& f : device_state_) f(t, x, h);
    }

    [[nodiscard]] bool empty() const noexcept {
        return accept_.empty() && commut_.empty();
    }

    /// Arms the hooks for a run: the initial state and the mask it was
    /// stamped under are the "previous step" of the first accepted step.
    void start(const Vector& x0, const topology::SwitchStateMask& mask0) {
        if (empty()) return;
        x_prev_ = x0;          // sizes the buffer (the only allocation)
        mask_prev_ = mask0;
        armed_ = true;
        n_commutations_ = 0;
    }

    /// An engine accepted the step [t_prev, t] under `mask`, ending in x.
    void accepted(Real t_prev, Real t, const Vector& x,
                  const topology::SwitchStateMask& mask) {
        if (empty()) return;
        for (const auto& f : accept_) f(t_prev, t, x, mask);
        if (!armed_) {
            // start() not called: this step only arms the comparison.
            x_prev_ = x;
            mask_prev_ = mask;
            armed_ = true;
            return;
        }
        if (!(mask == mask_prev_)) {
            ++n_commutations_;
            const Commutation c{t_prev, mask_prev_, mask, x_prev_, x};
            for (const auto& f : commut_) f(c);
        }
        x_prev_ = x;           // same size: no allocation
        mask_prev_ = mask;
    }

    /// Mask changes reported since `start`.
    [[nodiscard]] Size commutations() const noexcept { return n_commutations_; }

private:
    std::vector<AcceptFn> accept_;
    std::vector<CommutationFn> commut_;
    std::vector<StepFn> step_;
    std::vector<DeviceStateFn> device_state_;
    std::vector<EventFn> event_;
    Vector x_prev_;
    topology::SwitchStateMask mask_prev_;
    bool armed_ = false;
    Size n_commutations_ = 0;
};

}  // namespace pulsim::events
