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

struct SimulationResult {
    std::vector<Real>   times;
    std::vector<Vector> states;

    [[nodiscard]] Size num_steps() const noexcept {
        return times.size();
    }

    [[nodiscard]] bool empty() const noexcept {
        return times.empty();
    }

    /// Pre-allocate space for `n` samples on both internal
    /// vectors. Does NOT change `num_steps()` — capacity only.
    void reserve(Size n) {
        times.reserve(n);
        states.reserve(n);
    }
};

}  // namespace pulsim::v2::solver
