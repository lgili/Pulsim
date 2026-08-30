#pragma once

// =============================================================================
// Pulsim — Layer 5: SimulationResult (transient output container)
// =============================================================================
//
// `pulsim-v2-solver-and-events` Phase 1.
//
// Value-type aggregate holding the recorded `(t, x)` pairs from
// `run_transient`. Two parallel vectors:
//   * `times[k]`  → simulation time of sample k
//   * `states[k]` → state vector at `times[k]`
//
// `reserve(n)` pre-allocates both; `run_transient` calls it
// before the time-stepping loop to avoid per-step reallocation.
//
// v2.0 Phase 1 (audit finding `waveform-storage-vector-of-vectors`):
// `states` is no longer a `std::vector<Vector>` (one heap block per
// sample) but a `StateTrajectory` — ONE contiguous row-major
// buffer, allocated once, handed to Python as a zero-copy 2-D
// numpy view. The read API (`states[k]`, `size()`, `back()`,
// range-for) is unchanged except that element access yields an
// `Eigen::Map<const Vector>` by value; bind with `const auto&` or
// `const Vector&`, never a non-const `auto&`.
//
// Decimation now exists too: `SimulationOptions::store_every`
// records every m-th step (a PURE stride — steps 0, m, 2m, …, so
// the recorded grid stays strictly uniform at m·dt; the final step
// is included only when it lands on the stride), so a long
// high-fidelity run no longer has to hold every sample.

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/solver/state_trajectory.hpp"

#include <string>
#include <stdexcept>
#include <vector>

namespace pulsim::solver {

/// Layer 5 V2.2 — sub-step commutation timing diagnostic.
/// `t_estimated` is the linear-interpolated zero crossing of the
/// watched signal (i_diode for ON → OFF, v_diode − V_th for
/// OFF → ON) between two grid points. The state vectors
/// themselves remain at the dt grid; only the COMMUTATION
/// timestamp is sub-step accurate.
struct CommutationEvent {
    Real  t_estimated;
    Index branch_id;
    bool  new_state;
};

/// Phase-0 fix #9 — one record per time step where the diode
/// event-iteration loop could NOT reach a stable diode state:
/// either the iteration budget ran out, or a mask CYCLE was
/// detected (mask A → B → A …, the classic un-hysteresed diode
/// pair around a resonant node, which no budget resolves).
///
/// Pre-fix behaviour was a hard throw that discarded the whole
/// run ("minutes of compute lost at t = 37 ms"). The solver now
/// accepts the last consistent solve, records this breach, and
/// continues; the Python layer surfaces a loud warning. Set
/// `SimulationOptions::strict_event_iterations = true` to
/// restore the old throw.
struct EventIterationBreach {
    Real t;                 ///< step time of the breach
    Size iterations;        ///< solves attempted this step
    bool cycle_detected;    ///< true = mask cycle, false = budget
};

/// v2.0 Phase 2 (B.4) — a step the solver could not take at the
/// nominal dt, and had to re-take as 2^`halvings` sub-steps.
///
/// The output grid is UNCHANGED: sub-steps are internal, and a
/// sample is still only recorded at a nominal grid point. What this
/// records is that the trajectory between two samples was integrated
/// more finely than the user asked for — which is a change in
/// accuracy, not in the sampling, and the user should be told.
struct DtRetry {
    Real t;             ///< step time that had to be re-taken
    Size halvings;      ///< 2^halvings sub-steps were used
    std::string reason; ///< what the nominal-dt attempt reported
};

/// v2.0 Phase 2 — a post-solve guard OVERWROTE the solver's answer
/// for an inductor's branch current.
///
/// `inductor_freeze_di_max` and `inductor_abs_clamp` do not solve
/// anything: they replace a number the solver produced with one the
/// user configured. That is sometimes the only way to get a run to
/// finish, but it means the current being plotted is the LIMIT, not
/// the circuit — on the drive this was written for, the reported
/// line current peaks at exactly 100.000 A because the clamp is
/// 100 A. A guard that fires is evidence the model is missing
/// something (usually a snubber across a path that opens), and the
/// user has to be told, or they will read the limit as physics.
///
/// One record per inductor that fired, not per step.
struct InductorGuardAction {
    Index branch_id = kInvalidIndex;
    Size freeze_count = Size{0};   ///< steps snapped back to i_prev
    Size clamp_count  = Size{0};   ///< steps hard-limited
    Real t_first = Real{0};        ///< when it first fired
    Real worst_solved = Real{0};   ///< largest |i| the solver produced
    Real reported_limit = Real{0}; ///< the |i| the user sees instead

    [[nodiscard]] Size total() const noexcept {
        return freeze_count + clamp_count;
    }
};

/// v2.0 Phase 4 — the COMPLETE state of a run at one instant.
///
/// `initial_state=` restores only the MNA vector and then INVENTS
/// the companion history from it (a capacitor gets i_prev = 0, an
/// inductor v_prev = 0), so resuming from it does not reproduce
/// the run: a continuous 2T RLC and a T-then-resume differ by
/// 2.3e-4 where a true resume is ~1e-15. A snapshot carries the
/// parts that were missing, so `resume_from=` is exact.
struct SolverSnapshot {
    Real t = Real{0};
    /// MNA unknowns at `t`.
    Vector x;
    /// Trapezoidal companion history: 2 reals (v_prev, i_prev)
    /// per dynamic device, in HistoryState::entries() order.
    std::vector<Real> history;
    /// Which switched diodes were conducting. Solver-owned bits
    /// that a mask alone cannot reconstruct.
    std::vector<bool> diode_on;
    /// True once populated by a run.
    bool valid = false;
};

struct SimulationResult {
    std::vector<Real> times;

    /// v2.0 Phase 4 — the run's final state, complete enough to
    /// resume from exactly (`simulate(resume_from=...)`).
    SolverSnapshot final_snapshot;

    /// Recorded state samples, contiguous (v2.0 Phase 1).
    /// `states[k]` is the state at `times[k]`.
    StateTrajectory states;

    /// v2.0 Phase 2 — steps re-taken at a reduced dt. Empty on a run
    /// that never needed one.
    std::vector<DtRetry> dt_retries;

    /// v2.0 Phase 2 — inductors whose current a post-solve guard
    /// overwrote. Empty unless `inductor_freeze_di_max` or
    /// `inductor_abs_clamp` was set AND actually fired.
    std::vector<InductorGuardAction> inductor_guard_actions;

    /// Phase-0 fix #9 — steps where diode event iteration hit a
    /// cycle or the budget. Empty on a fully-converged run.
    std::vector<EventIterationBreach> event_iteration_breaches;

    /// Per-step count of how many `cache.solve` invocations were
    /// needed before the diode state stabilised. Parallel to
    /// `times` and `states`. Zero means "first solve was already
    /// consistent" (no diode flips, or no diodes in the circuit
    /// at all).
    ///
    /// Diagnostic only — Layer 5 V2.1 populates it when diodes
    /// are present; otherwise it's left empty (Layer 5 V0/V1
    /// behaviour).
    std::vector<Size> event_iteration_count;

    /// Layer 5 V2.2 — list of estimated commutation times for
    /// every diode flip detected during the simulation. Ordered
    /// by `t_estimated`. Empty if no diodes are in the circuit
    /// or no flips occurred.
    std::vector<CommutationEvent> commutation_events;

    [[nodiscard]] Size num_steps() const noexcept {
        return times.size();
    }

    [[nodiscard]] bool empty() const noexcept {
        return times.empty();
    }

    /// Pre-allocate space for `n` samples on all internal
    /// vectors. Does NOT change `num_steps()` — capacity only.
    ///
    /// Pass `state_size` (v2.0 Phase 1) so the contiguous sample
    /// buffer can be allocated in ONE shot for the whole run; the
    /// no-size overload leaves it to the first recorded sample.
    void reserve(Size n) {
        times.reserve(n);
        states.reserve(n);
        event_iteration_count.reserve(n);
    }

    void reserve(Size n, Index state_size) {
        states.set_state_size(state_size);
        reserve(n);
    }

    /// Estimated bytes held by the recorded waveform (contiguous
    /// sample buffer + the parallel time / diagnostic vectors).
    [[nodiscard]] std::size_t approx_bytes() const noexcept {
        return states.bytes() +
               times.size() * sizeof(Real) +
               event_iteration_count.size() * sizeof(Size) +
               commutation_events.size() * sizeof(CommutationEvent) +
               event_iteration_breaches.size() *
                   sizeof(EventIterationBreach) +
               inductor_guard_actions.size() *
                   sizeof(InductorGuardAction) +
               // Each DtRetry carries the full Newton failure text
               // (a few hundred bytes with the row naming Phase 1
               // added), so the strings dominate the structs.
               [this] {
                   std::size_t n = 0;
                   for (const auto& r : dt_retries) {
                       n += sizeof(DtRetry) + r.reason.capacity();
                   }
                   return n;
               }();
    }
};

/// Thrown when a transient cannot continue — and it brings the run
/// with it.
///
/// v2.0 Phase 2. A simulation that dies at 90 % used to return
/// nothing at all: the exception carried a message and every sample
/// computed before the failure was destroyed with the stack. On a
/// run that takes minutes that is the difference between "here is
/// where it broke, and here is the waveform leading into it" and
/// "start again".
///
/// The default remains an exception, deliberately. Returning a
/// truncated result as if it were a whole one is exactly the silent
/// wrong answer this project keeps removing; a caller who wants the
/// partial trace has to ask for it by catching this type.
class SimulationAborted : public std::runtime_error {
public:
    SimulationAborted(const std::string& what,
                       SimulationResult partial,
                       Real t_failed)
        : std::runtime_error(what),
          partial_(std::move(partial)),
          t_failed_(t_failed) {}

    /// Everything that WAS computed, up to but not including the
    /// step that failed.
    [[nodiscard]] const SimulationResult& partial() const noexcept {
        return partial_;
    }

    /// The step time the solver could not reach.
    [[nodiscard]] Real t_failed() const noexcept { return t_failed_; }

private:
    SimulationResult partial_;
    Real t_failed_;
};

}  // namespace pulsim::solver
