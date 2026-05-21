## Why

V2.2 (sub-step commutation diagnostics) detects zero-crossing
events INSIDE a time step via linear interpolation of the
watched diode signal between `t_prev` and `t = t_prev + dt`,
and records the estimated commutation time `t_est`. But the
state vector `x` STILL evolves from `t_prev` to `t` as a
single trap-companion step using a SINGLE switch state —
either pre-event or post-event, whichever the diode-iteration
loop converges to. The "wrong" switch state for part of the
interval produces visible commutation-wobble in the output.

V7 (multi-dt cache) shipped `solve_at(mask, dt, b_extra, x)`,
the primitive that lets us solve at arbitrary auxiliary `dt`
values without rebuilding the cache. This is the missing
piece — combined with V2.2's `t_est`, we can RETROACTIVELY
correct `x` after a commutation event by splitting the step
into two sub-steps:
- `dt₁ = t_est − t_prev`: solve with the PRE-event mask.
- `dt₂ = t − t_est`: solve with the POST-event mask.

The result is a state vector at `t` that reflects the
correct commutation timing, eliminating the single-shot
wobble.

## What Changes

**Scope decision — Layer 5 V3** (substep state correction):

- Extend `SimulationOptions`:
  ```cpp
  bool enable_substep_state_correction = false;
  ```
  Default `false` to preserve V2.2 behaviour (timestamps
  only, no state correction).

- Extend `HistoryState` with snapshot/restore:
  ```cpp
  [[nodiscard]] std::vector<HistoryEntry> snapshot() const;
  void restore(const std::vector<HistoryEntry>& snap);
  ```

- Extend `DiodeEventState` with snapshot/restore:
  ```cpp
  [[nodiscard]] std::vector<bool> snapshot_on_bits() const;
  void restore_on_bits(const std::vector<bool>& bits);
  ```

- In `run_transient`'s dynamic path, when
  `enable_substep_state_correction = true` AND a
  commutation event is detected during a step:
  1. Roll back `x`, `history`, and `diodes` to their
     pre-step state.
  2. Sub-step 1 with pre-event mask via `cache.solve_at(
     mask_pre, dt₁, b_extra_at_dt1, x)`.
  3. Apply commutation: `diodes.update_from_state(x)`.
  4. Sub-step 2 with post-event mask via `cache.solve_at(
     mask_post, dt₂, b_extra_at_dt2, x)`.
  5. Continue with the corrected `x` at `t`.

  V0 handles ONLY the first detected event per step
  (multi-event substepping is V1).

- **Test** (coarse-dt half-wave rectifier): same circuit as
  the V2 test but with dt 10× coarser (so commutation
  wobble is visible). Verify that:
  - Without substep correction: output near zero-crossing
    is OFF by up to `dt × V_amp × ω` (the linear sweep
    distance).
  - With substep correction enabled: output near
    zero-crossing matches `max(V_sine − V_F0, 0)` within
    a much tighter tolerance.

## Impact

- **Affected specs**: ADDED requirement on
  `kernel-v2-solver` for `enable_substep_state_correction`
  and helper methods on HistoryState/DiodeEventState.
- **Affected code** (~250 LOC):
  - MODIFIED `solver/options.hpp` (+ flag)
  - MODIFIED `solver/run_transient.hpp` (+ sub-step path)
  - MODIFIED `pwl/history_state.hpp` (+ snapshot/restore)
  - MODIFIED `pwl/diode_event_state.hpp` (+ snapshot/restore)
  - NEW `tests/v2/layer5_v3/test_substep_correction.cpp`
- **Migration**: zero. Default `false` keeps V2.2
  behaviour. Existing tests stay green.
- **Risk**: low for the primitive (uses well-tested
  `solve_at` + the established trap-companion). Medium
  for the per-step bookkeeping (history/diode rollback
  must be exact). Tests cover both paths.
