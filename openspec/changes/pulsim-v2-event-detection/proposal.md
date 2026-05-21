## Why

Layer 5 V2 added auto-commutating diodes, but the per-step
state-decision logic chatters at the DCM/CCM boundary. Concrete
failure mode: the boost converter integration test stays stuck
at V_out ≈ 25 mV instead of converging to 24 V because:

- At the moment of commutation (Q OFF, D should turn ON), the
  V2 loop solves with the PREVIOUS step's diode state.
- If the previous state was wrong for the new instant (e.g.,
  D was OFF and is now reverse-biased on the wrong side of a
  fast event), `v_sw` blows up to enormous values during the
  brief "both switches OFF" window.
- The next step catches the error and flips the diode — but the
  L's current has already collapsed numerically.

**The fix**: iterate the switch state at each step until
consistent BEFORE recording the sample. This is the "single-
event-per-step" V0 of event detection — much simpler than
sub-step bisection (Brent's method etc.), and sufficient to
handle the boost converter cleanly.

## What Changes

**Scope decision — Layer 5 V2.1 (event-detection V0)**:

- After each `cache.solve` in the dynamic / static loop, run a
  **state-consistency iteration**:
  1. Call `diodes.update_from_state(x)`.
  2. If any diode flipped, re-solve with the new combined mask.
  3. Repeat until no flips or `max_event_iterations` hit.
- New `SimulationOptions::max_event_iterations` field
  (default 16).
- New `SimulationResult::event_iteration_count` per sample
  (vector<Size>) — diagnostics showing how many iterations each
  step needed. Zero for steps where no flip happened; positive
  for commutation steps.
- **Boost converter integration test** re-enabled in
  `tests/v2/layer5_v2/test_integration_boost.cpp`.

**V0 limitations** (documented but not blocking):
- Still uses fixed `dt` — events get caught WITHIN a step, but
  the EXACT crossing time within that step isn't resolved.
  Sub-step bisection lands in a future
  `pulsim-v2-substep-bisection` follow-up.
- One commutation per device per step max (after that, the
  iteration limit is hit and the simulation throws). For
  typical PE workloads this is more than sufficient.

## Impact

- **Affected specs**: MODIFIED `kernel-v2-solver` (ADDED
  Requirement for event iteration loop + SimulationOptions
  extension).
- **Affected code** (estimate):
  - MODIFIED `solver/options.hpp` (~5 LOC for new field)
  - MODIFIED `solver/result.hpp` (~5 LOC for new vector)
  - MODIFIED `solver/run_transient.hpp` (~30 LOC for the
    iteration loop)
  - NEW `tests/v2/layer5_v2/test_integration_boost.cpp` (~200
    LOC, re-enabled from V2 with parameters that exercise the
    iteration)
  - MODIFIED CMake target to include the boost test.
- **Migration**: zero. Default `max_event_iterations = 16` is
  silently applied to all existing simulations. For circuits
  without diodes the iteration loop runs 0 times.
- **Risk**: low. The iteration is bounded; the safety check
  throws clean errors on non-convergence; existing tests
  (half-wave rectifier, RC/RL/RLC, chopper, buck) are
  unaffected (they don't trigger the iteration).
