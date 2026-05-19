#pragma once

// simplify-and-harden-numerical-surface — Phase 7.1.
//
// Canonical location for the homotopy continuation DC strategy — the
// fifth fallback in the `DCStrategy::Auto` ladder (Direct →
// SourceStepping → GminStepping → PseudoTransient → Homotopy).
//
// Algorithm: step parameter `λ` from `0` (all nonlinear devices
// replaced by their `g_off` linear conductance — solves in one direct
// call) to `1` (full nonlinear model), with warm-started Newton at
// each step.
//
// The actual implementation lives as `DCConvergenceSolver::try_homotopy`
// in `convergence_aids.hpp` → `numerical/dc_strategy.hpp`. Extracting
// into a free function `solve_homotopy_dc(...)` would require
// re-plumbing the device store, scaled-residual helpers, and warm-start
// state — extraction is deferred to a Phase 3-style operator-network
// refactor.
//
// Tuning surface (post-Phase 3):
//
//   opts.advanced().dc.homotopy_config.enable             = true;   // default
//   opts.advanced().dc.homotopy_config.ladder_steps       = 5;      // default (10 in HighFidelity)
//   opts.advanced().dc.homotopy_config.max_newton_per_step = 10;
//
// Telemetry:
//   result.dc_result.homotopy_steps               // # λ increments
//   result.dc_result.homotopy_ladder_completed    // reached λ=1?
//
// Reference test: `core/tests/test_homotopy_dc.cpp` (5 cases, 18
// assertions).

#include "pulsim/v1/numerical/dc_strategy.hpp"
