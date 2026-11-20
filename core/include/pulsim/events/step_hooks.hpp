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
#include <stdexcept>
#include <vector>

namespace pulsim::pwl {
class StatefulHistories;
class HistoryState;
}

#if defined(__GNUC__) || defined(__clang__)
#define PULSIM_HOOKS_INLINE __attribute__((always_inline)) inline
#define PULSIM_HOOKS_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define PULSIM_HOOKS_INLINE inline
#define PULSIM_HOOKS_NOINLINE __declspec(noinline)
#else
#define PULSIM_HOOKS_INLINE inline
#define PULSIM_HOOKS_NOINLINE
#endif

namespace pulsim::events {

struct Commutation {
    Real t;
    const topology::SwitchStateMask& before;
    const topology::SwitchStateMask& after;
    const Vector& x_before;
    const Vector& x_after;
};

/// add-probes-and-recorders item 1: what a listener may READ during a
/// run besides a hook's arguments: the linear companion history (a
/// capacitor's committed current `i_prev`, the KCL-consistent one), the
/// stateful devices' histories, and the step of the last post-event
/// probe (TR-BDF2: the probe that produced an event's right side).
/// Set by the engine once per run; null where an engine has none.
struct StepRunView {
    const pwl::HistoryState* history = nullptr;
    const pwl::StatefulHistories* stateful = nullptr;
    const Real* probe_dt = nullptr;
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
    // commit of an accepted step, `t` the accepted time. TR-BDF2 fires it
    // once per committed time: each step end and also each post-event
    // probe commit and diode landing (external-device ABI #35), so it can
    // fire more often than `step`, never twice at one time.
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
               const Vector& x_right, const topology::SwitchStateMask& mask_right) {
        flush_steps();   // the sink sees its steps before an event at their end
        for (const auto& f : event_) f(t, x_left, mask_left, x_right, mask_right);
    }
    void on_step(StepFn f) { step_.push_back(std::move(f)); }
    void on_device_state(DeviceStateFn f) { device_state_.push_back(std::move(f)); }
    // release D3: a STORED sample that is not an accepted step. TR-BDF2
    // stores a gate edge's right limit x(t_e+) at t_e + h_p (the edge's
    // probe) and starts the next step from t_e, so that step ends after the
    // sample but begins before it. Called when the row is stored, with the
    // histories already the probe's; `x == nullptr` withdraws the sample at
    // `t` (a later commit did not lie after it: the row was dropped).
    using SampleFn = std::function<void(Real t, const Vector* x,
                                        const topology::SwitchStateMask* mask)>;
    void on_sample(SampleFn f) { sample_.push_back(std::move(f)); }
    [[nodiscard]] bool has_sample() const noexcept { return !sample_.empty(); }
    void sample(Real t, const Vector* x, const topology::SwitchStateMask* mask) const {
        for (const auto& f : sample_) f(t, x, mask);
    }
    /// probes 0b.10: ONE listener may take the accepted steps through a
    /// plain function pointer (no std::function hop) that is told when the
    /// step starts exactly where the previous call to it ended: `x_start` /
    /// `mask_start` are then nullptr (contiguous: bit for bit the previous
    /// call's `x_end` / `mask_end`), otherwise the step's start. The first
    /// call of a run always carries the start. `step()` calls it too.
    using StepSinkFn = void (*)(void* ctx, Real t_prev, const Vector* x_start,
                                const topology::SwitchStateMask* mask_start, Real t,
                                const Vector& x_end, const topology::SwitchStateMask& mask_end);
    /// The sink may also take CONTIGUOUS steps under one mask in batches:
    /// `rows` holds n rows of `stride` = 1 + state size reals each, `t` then
    /// the end state (copies: nothing points into the engine), all solved
    /// under `mask` and each starting where the previous one ended. The
    /// hooks hand them over before any other sink call or event, when the
    /// buffer is full, and at `flush_steps()` (the engines call it at the
    /// end of a run; a run that throws leaves its rows held until the next
    /// `start()` drops them).
    using StepBatchFn = void (*)(void* ctx, const Real* rows, Size n, Size stride,
                                 const topology::SwitchStateMask& mask);
    void set_step_sink(StepSinkFn f, void* ctx, StepBatchFn batch = nullptr) {
        flush_steps();   // rows held for the sink being replaced go to it
        if (sink_ != nullptr && f != nullptr) {
            throw std::logic_error("StepHooks: a step sink is already set (one per hooks)");
        }
        sink_ = f;
        sink_ctx_ = ctx;
        batch_fn_ = f != nullptr ? batch : nullptr;
    }
    /// Any step listener (an `on_step` function or the sink).
    [[nodiscard]] bool has_step() const noexcept { return !step_.empty() || sink_ != nullptr; }
    /// `on_step` functions: they need the start state on every step.
    [[nodiscard]] bool has_step_fns() const noexcept { return !step_.empty(); }
    [[nodiscard]] bool has_device_state() const noexcept { return !device_state_.empty(); }
    void step(Real t_prev, const Vector& x_start,
              const topology::SwitchStateMask& mask_start, Real t,
              const Vector& x_end, const topology::SwitchStateMask& mask_end) {
        step(t_prev, x_start, mask_start, t, x_end, mask_end, false);
    }
    /// The engines' call. `contiguous`: the step starts bit for bit where the
    /// previous call of this run ended (x and mask); x_start / mask_start
    /// need be valid only when it is false or `has_step_fns()`.
    PULSIM_HOOKS_INLINE void step(Real t_prev, const Vector& x_start,
                                  const topology::SwitchStateMask& mask_start, Real t,
                                  const Vector& x_end, const topology::SwitchStateMask& mask_end,
                                  bool contiguous) {
        // the common step, in the engine's loop: one more row of an open
        // batch under its mask (no step function to call)
        if (contiguous && batch_n_ != 0 && step_.empty() &&
            static_cast<Size>(x_end.size()) + 1 == batch_stride_ && mask_end == batch_mask_) {
            Real* const row = batch_.data() + batch_n_ * batch_stride_;
            row[0] = t;
            const Real* const src = x_end.data();
            for (Size k = 1; k < batch_stride_; ++k) row[k] = src[k - 1];
            if (++batch_n_ == batch_cap_) flush_steps();
            return;
        }
        step_other(t_prev, x_start, mask_start, t, x_end, mask_end, contiguous);
    }
    /// Every other step: out of the engine's loop.
    PULSIM_HOOKS_NOINLINE void step_other(Real t_prev, const Vector& x_start,
                                          const topology::SwitchStateMask& mask_start, Real t,
                                          const Vector& x_end,
                                          const topology::SwitchStateMask& mask_end,
                                          bool contiguous) {
        for (const auto& f : step_) f(t_prev, x_start, mask_start, t, x_end, mask_end);
        if (sink_ == nullptr) return;
        if (contiguous && batch_cap_ != 0 &&
            static_cast<Size>(x_end.size()) + 1 == batch_stride_) {
            if (batch_n_ == 0) {
                batch_mask_ = mask_end;
            } else if (!(mask_end == batch_mask_)) {
                flush_steps();
                batch_mask_ = mask_end;
            }
            Real* const row = batch_.data() + batch_n_ * batch_stride_;
            row[0] = t;
            const Real* const src = x_end.data();
            for (Size k = 1; k < batch_stride_; ++k) row[k] = src[k - 1];
            if (++batch_n_ == batch_cap_) flush_steps();
            return;
        }
        flush_steps();
        sink_(sink_ctx_, t_prev, contiguous ? nullptr : &x_start,
              contiguous ? nullptr : &mask_start, t, x_end, mask_end);
    }
    /// Hand the batched steps to the sink (no-op when none are held).
    void flush_steps() {
        if (batch_n_ == 0) return;
        const Size n = batch_n_;
        batch_n_ = 0;
        batch_fn_(sink_ctx_, batch_.data(), n, batch_stride_, batch_mask_);
    }
    [[nodiscard]] bool steps_held() const noexcept { return batch_n_ != 0; }
    void device_state(Real t, const Vector& x, const pwl::StatefulHistories& h) const {
        for (const auto& f : device_state_) f(t, x, h);
    }

    [[nodiscard]] bool empty() const noexcept {
        return accept_.empty() && commut_.empty();
    }

    using RunView = StepRunView;
    [[nodiscard]] const RunView& run_view() const noexcept { return view_; }
    /// The run's start state and stamp, with the run view already set (a
    /// recorder seeds the readings that depend on the histories there).
    using StartFn = std::function<void(const Vector& x0, const topology::SwitchStateMask& mask0)>;
    void on_start(StartFn f) { start_.push_back(std::move(f)); }

    /// Arms the hooks for a run: the initial state and the mask it was
    /// stamped under are the "previous step" of the first accepted step.
    void start(const Vector& x0, const topology::SwitchStateMask& mask0, RunView view = {}) {
        view_ = view;
        batch_n_ = 0;   // rows of a run that threw are dropped
        batch_cap_ = 0;
        if (batch_fn_ != nullptr) {
            // the batch buffer: sized here, never regrown during the run
            batch_stride_ = static_cast<Size>(x0.size()) + 1;
            batch_cap_ = kBatchRows;
            batch_.assign(batch_cap_ * batch_stride_, Real{0});
            batch_mask_ = mask0;   // its words too (a wide mask's heap words)
        }
        for (const auto& f : start_) f(x0, mask0);
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
    std::vector<SampleFn> sample_;
    std::vector<EventFn> event_;
    std::vector<StartFn> start_;
    StepSinkFn sink_ = nullptr;
    void* sink_ctx_ = nullptr;
    StepBatchFn batch_fn_ = nullptr;
    static constexpr Size kBatchRows = 128;
    std::vector<Real> batch_;
    Size batch_cap_ = 0, batch_n_ = 0, batch_stride_ = 0;   // rows
    topology::SwitchStateMask batch_mask_;
    Vector x_prev_;
    topology::SwitchStateMask mask_prev_;
    RunView view_;
    bool armed_ = false;
    Size n_commutations_ = 0;
};

}  // namespace pulsim::events
