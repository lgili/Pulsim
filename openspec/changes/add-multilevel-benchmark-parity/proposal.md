# Multilevel converter benchmark suite + PLECS / PSIM parity gates

## Why

`simplify-and-harden-numerical-surface` shipped the underlying machinery
needed to claim multilevel-converter parity with industrial-grade
SPICE-like tools (PLECS / PSIM): the MMC topology template (Phase 12),
homotopy DC convergence (Phase 7), iterative refinement on direct solves
(Phase 6), simultaneous PWL event coalescence (Phase 5), and the
collapsed `Preset` selector (Phase 2). But Phase 13 of that change —
the actual benchmark + RMS-error gates against external golden CSVs —
was externally blocked because the golden CSVs from PLECS / PSIM
licensed installations weren't available on the implementer's branch.

This change picks up Phase 13 + 15.2 + 15.3 as its own scope, with the
contract that the golden CSVs become available as a prerequisite.

## What Changes

1. Add 4 multilevel benchmark circuits as YAML netlists:
   - `benchmarks/multilevel/3level_npc.yaml`
   - `benchmarks/multilevel/5level_flying_cap.yaml`
   - `benchmarks/multilevel/ttype_3level.yaml`
   - `benchmarks/multilevel/mmc_9sub.yaml`

2. Export golden waveforms from PLECS (for the three NPC / flying-cap /
   T-type variants) and PSIM (for the MMC, since MMC modelling is
   PSIM's strong suit historically). Version-tag each CSV by the
   exporting tool's version string and the simulation parameters used
   so future re-exports remain reproducible.

3. Add 4 Catch2 regression tests gating on:
   - ≤ 0.5 % RMS error for NPC / flying-cap / T-type vs PLECS goldens
     across the canonical observables (V_phase, I_load, V_caps)
   - ≤ 1 % RMS error for the MMC vs PSIM goldens (looser bound because
     MMC controller details vary across implementations)

4. Add a wall-clock comparison runner
   (`tools/multilevel_bench_runner.py`) that runs the Pulsim simulation
   alongside PLECS / PSIM exports of the same circuit and reports the
   relative wall-clock factor. The contract is: Pulsim SHALL be within
   2× of the slower of the two competitors on each circuit.

5. Document the benchmark gates + the parity contract in
   `docs/benchmarks-and-parity.md`.

## Impact

- **Affected spec**: `benchmark-suite` (adds 4 new regression
  requirements + 1 wall-clock parity requirement).
- **Affected code**:
  - New: `benchmarks/multilevel/*.yaml`, `benchmarks/multilevel/golden/*.csv`,
    `core/tests/test_multilevel_*.cpp`, `tools/multilevel_bench_runner.py`,
    `docs/benchmarks-and-parity.md`.
  - No changes to kernel code (the convergence-aid machinery is already
    shipped).
- **Prerequisites**: access to PLECS + PSIM licensed installations for
  the one-time golden export.
- **Risk**: low. The kernel-side work is already validated by the
  Phase 12 MMC template's own cold-start tests; this change adds
  parity-with-competitor gates on top.
