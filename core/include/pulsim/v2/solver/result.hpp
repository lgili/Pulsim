#pragma once

// =============================================================================
// Pulsim v2 — Layer 5: SimulationResult (transient output container)
// =============================================================================
//
// `pulsim-v2-solver-and-events` Phase 1.
//
// Value-type aggregate holding the recorded `(t, x)` pairs from
// `run_transient`. Two parallel vectors:
//   * `times[k]`  → simulation time of sample k
//   * `states[k]` → state vector at `times[k]`
//
// `reserve(n)` pre-allocates both vectors; `run_transient` calls
// it before the time-stepping loop to avoid per-step
// reallocation.
//
// V0 keeps every sample. Strided / downsampled output is a V1
// add (the solver loop already records into this struct; adding
// stride is a 2-line change inside the loop).

#include "pulsim/v2/numeric/dense.hpp"
#include "pulsim/v2/numeric/types.hpp"

#include <vector>

namespace pulsim::v2::solver {

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

struct SimulationResult {
    std::vector<Real>   times;
    std::vector<Vector> states;

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
    void reserve(Size n) {
        times.reserve(n);
        states.reserve(n);
        event_iteration_count.reserve(n);
    }
};

}  // namespace pulsim::v2::solver
