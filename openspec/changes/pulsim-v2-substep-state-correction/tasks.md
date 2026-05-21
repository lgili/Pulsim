## Phase 1 — Snapshot/restore helpers (~0.2 days)

- [x] 1.1 `HistoryState::snapshot()` / `restore(snap)`.
- [x] 1.2 `DiodeEventState::snapshot_on_bits()` /
      `restore_on_bits(bits)` with size validation.
- [x] 1.3 Unit tests for both round-trips.

## Phase 2 — Options flag + run_transient wiring (~0.4 days)

- [x] 2.1 `enable_substep_state_correction` in
      `SimulationOptions` (default false).
- [x] 2.2 Dynamic-path substep correction: snapshot before
      step, detect events after step, roll back + replay
      as two `solve_at` calls.
- [x] 2.3 Apply commutation via the V2.2 event's
      `new_state` (not via `update_from_state` —
      decision logic at the exact zero-crossing keeps the
      pre-event state).
- [x] 2.4 Skip correction when `dt1` or `dt2` is below
      1 % of `opts.dt` (avoids ill-conditioned trap
      companion at boundary events).

## Phase 3 — Integration test (~0.3 days)

- [x] 3.1 RC half-wave rectifier (dt = 200 µs, R = 100 Ω,
      C = 1 µF, V_sine = 10 V at 60 Hz).
- [x] 3.2 Run twice: with and without
      `enable_substep_state_correction`.
- [x] 3.3 Verify output stays finite + bounded; mean cap
      voltage matches the non-correction run within 10 %.

## Phase 4 — Regression + docs (~0.1 days)

- [x] 4.1 All 13 v2 test binaries pass (4348 assertions /
      267 cases). Default `false` keeps V2.2 behaviour
      bit-identical.
- [x] 4.2 `openspec validate
      pulsim-v2-substep-state-correction --strict` passes.
- [x] 4.3 `docs/pulsim-v2/layer5-v3-substep-correction.md`.

## What was tried and learned

The original V0 design aimed for "substep correction
reduces zero-crossing tracking error at coarse dt." During
implementation we discovered:

* For **ideal-switch diodes (V_th = 0)**: V2.2's
  linear-interpolation event timing uses `v_diode` as the
  watched signal. While conducting, `v_diode ≈ 0`; while
  off, `v_diode ≈ V_sine`. The sign-change interpolation
  thus lands at the step boundary (where `v_diode`
  abruptly jumps from ≈0 to `V_sine_curr`). Sub-step
  durations `dt₁` or `dt₂` collapse to ≈0, and the
  correction is correctly skipped (the trap-companion
  `g_eq = 2C/dt` becomes ill-conditioned at tiny dt).

* For **smooth-blend diodes (V2.2's t_est is more
  accurate)**: substep correction would produce a
  measurable improvement, but smooth-blend devices
  currently use the Newton path (`nl_refresh`), which
  doesn't interact with V2.2's diode-state tracking.
  Wiring substep correction into the Newton path is a
  larger V1 effort.

V3 ships the substep correction mechanics. The flag is
opt-in and defaults to `false`. Future OpenSpecs may
extend the trigger to Newton-path diodes or smarter
t_est estimators.
