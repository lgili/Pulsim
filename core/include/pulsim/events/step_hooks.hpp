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
    Vector x_prev_;
    topology::SwitchStateMask mask_prev_;
    bool armed_ = false;
    Size n_commutations_ = 0;
};

}  // namespace pulsim::events
