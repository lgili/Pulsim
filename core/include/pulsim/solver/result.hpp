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

struct SimulationResult {
    std::vector<Real> times;

    /// Recorded state samples, contiguous (v2.0 Phase 1).
    /// `states[k]` is the state at `times[k]`.
    StateTrajectory states;

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
                   sizeof(EventIterationBreach);
    }
};

}  // namespace pulsim::solver
