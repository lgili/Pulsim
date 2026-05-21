#pragma once

// =============================================================================
// Pulsim v2 — Layer 5: SimulationOptions (fixed-dt time-stepping inputs)
// =============================================================================
//
// `pulsim-v2-solver-and-events` Phase 1.
//
// Value-type aggregate that holds the inputs to `run_transient`:
// the simulation window [t_start, t_end] and the fixed dt.
//
// `valid()` performs a self-check (finite values, dt > 0,
// t_end > t_start) — `run_transient` calls it and throws
// `std::invalid_argument` if invalid, so the user gets a clean
// error rather than a silent infinite loop.
//
// `expected_step_count()` is the size hint for output
// pre-allocation. It uses `floor((t_end - t_start) / dt) + 1` so
// the result includes both endpoint samples (at t_start and at
// t_end, or the last sample <= t_end if dt doesn't divide the
// span evenly).

#include "pulsim/v2/numeric/types.hpp"

#include <cmath>

namespace pulsim::v2::solver {

struct SimulationOptions {
    Real t_start = Real{0};
    Real t_end   = Real{0};
    Real dt      = Real{0};

    [[nodiscard]] bool valid() const noexcept {
        // All three values must be finite (no NaN, no infinity).
        if (!std::isfinite(t_start) || !std::isfinite(t_end) ||
            !std::isfinite(dt)) {
            return false;
        }
        // Forward-progress invariants.
        if (dt <= Real{0}) {
            return false;
        }
        if (t_end <= t_start) {
            return false;
        }
        return true;
    }

    /// Number of output samples that `run_transient` will record
    /// for valid options. Includes both endpoint samples — the
    /// loop visits k = 0, 1, …, N - 1 with t = t_start + k · dt
    /// and the last sample is the largest k such that
    /// `t_start + k · dt <= t_end`.
    [[nodiscard]] Size expected_step_count() const noexcept {
        if (!valid()) {
            return 0;
        }
        const Real span = t_end - t_start;
        // floor(span/dt) + 1 — the "+1" counts the t_start sample.
        // Use std::floor + integer cast to defend against tiny FP
        // overshoot at the last step.
        const Real n_real = std::floor(span / dt) + Real{1};
        return static_cast<Size>(n_real);
    }
};

}  // namespace pulsim::v2::solver
