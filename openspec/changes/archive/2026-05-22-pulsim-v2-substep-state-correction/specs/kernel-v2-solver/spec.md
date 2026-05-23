## ADDED Requirements

### Requirement: enable_substep_state_correction option

`SimulationOptions::enable_substep_state_correction` (default `false`) SHALL gate sub-step state correction in `run_transient`. When `true` AND `cache.dt() > 0` (dynamic path), each time step that detects a commutation event MUST be retroactively split into two sub-steps at the estimated event time `t_est`.

The sub-step procedure MUST:
1. Roll back `x`, `history`, and `diodes` to the pre-step
   snapshots.
2. Sub-step 1: solve via `cache.solve_at(mask_pre, dt₁,
   b_extra_at_t_est, x)` where `dt₁ = t_est − t_prev`.
3. Apply commutation via `diodes.update_from_state(x)`.
4. Sub-step 2: solve via `cache.solve_at(mask_post, dt₂,
   b_extra_at_t, x)` where `dt₂ = dt − dt₁`.
5. Commit `history` updates after each sub-step.

V0 corrects only the FIRST detected event per step. Multiple
events in a single step trigger V0 sub-step on the first
only; the rest carry over to the next step's detection.

When `enable_substep_state_correction = false`, run_transient
MUST behave exactly as in V2.2 (timestamp diagnostics only,
no state correction).

#### Scenario: Default behaviour preserves V2.2

- **GIVEN** `SimulationOptions{}` (default constructed)
- **THEN** `enable_substep_state_correction` SHALL be `false`
- **AND** `run_transient` SHALL produce bit-identical output
  to V2.2 for the same circuit.

#### Scenario: Substep correction keeps output finite and bounded

- **GIVEN** an RC half-wave rectifier (V_sine=10V, 60Hz,
  R=100Ω, C=1µF, ideal-switch diode) at dt = 200 µs
- **WHEN** the simulation runs with
  `enable_substep_state_correction = true`
- **THEN** all recorded state vectors SHALL be finite and
  bounded by the input amplitude (with margin for the
  cap-discharge transient)
- **AND** the mean cap voltage over the last cycle SHALL
  match the non-correction run within 10 % (long-time
  behaviour is similar; the correction's per-step impact
  is bounded).

### Requirement: HistoryState snapshot/restore

`HistoryState` SHALL expose `snapshot()` returning the
per-device `(v_prev, i_prev)` state, and `restore(snap)`
that overwrites the internal state with the given snapshot.

The snapshot MUST be sufficient for round-trip preservation:
applying `restore(snapshot())` is a no-op.

#### Scenario: Round-trip snapshot/restore is a no-op

- **GIVEN** a `HistoryState` with non-zero v_prev / i_prev
- **WHEN** the user captures `snap = h.snapshot()`, mutates
  `h` via `h.update_from_state(x, dt)`, and calls
  `h.restore(snap)`
- **THEN** subsequent calls to `h.compute_b_extra(dt)` SHALL
  produce the same vector as before the mutation.

### Requirement: DiodeEventState snapshot/restore

`DiodeEventState` SHALL expose `snapshot_on_bits()` returning
the per-diode `is_on` bits, and `restore_on_bits(bits)` that
overwrites the internal state with the given bits.

The methods MUST validate that the bit-vector size matches
the number of registered diodes; size mismatch SHALL throw
`std::invalid_argument`.

#### Scenario: Restore reverses update_from_state

- **GIVEN** a `DiodeEventState` with two diodes both OFF
- **WHEN** the user captures
  `snap = d.snapshot_on_bits()`, calls
  `d.update_from_state(x)` that flips both diodes ON, then
  calls `d.restore_on_bits(snap)`
- **THEN** `d.current_diode_mask()` SHALL match the original
  all-OFF mask.
