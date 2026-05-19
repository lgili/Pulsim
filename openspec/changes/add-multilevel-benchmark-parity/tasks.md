## 1. Phase 1 — Golden CSV exports (one-time, requires PLECS / PSIM)

- [ ] 1.1 Export 3-level NPC golden from PLECS — observables
      `V_phase_A`, `I_load_A`, `V_cap_neutral` over one fundamental
      period. Version-tag PLECS release in CSV header.
- [ ] 1.2 Export 5-level flying-cap golden from PLECS — same observables
      plus `V_cap_FC_{1..4}`.
- [ ] 1.3 Export T-type 3-level golden from PLECS.
- [ ] 1.4 Export 9-submodule MMC golden from PSIM — observables include
      arm currents, circulating current, and per-submodule V_cap.

## 2. Phase 2 — Benchmark YAML netlists

- [ ] 2.1 Add `benchmarks/multilevel/3level_npc.yaml` matching the
      PLECS export circuit exactly (PWM frequency, modulation index,
      load, DC link).
- [ ] 2.2 Add `benchmarks/multilevel/5level_flying_cap.yaml`.
- [ ] 2.3 Add `benchmarks/multilevel/ttype_3level.yaml`.
- [ ] 2.4 Add `benchmarks/multilevel/mmc_9sub.yaml` using
      `templates::mmc_3phase_inverter` (the helper shipped in
      `simplify-and-harden-numerical-surface` Phase 12).

## 3. Phase 3 — Regression tests

- [ ] 3.1 `test_multilevel_npc.cpp` — load YAML, run transient,
      compute RMS error vs golden CSV on each observable, gate ≤ 0.5 %.
- [ ] 3.2 `test_multilevel_flying_cap.cpp` — same gate.
- [ ] 3.3 `test_multilevel_ttype.cpp` — same gate.
- [ ] 3.4 `test_multilevel_mmc.cpp` — gate ≤ 1 % RMS (looser because
      MMC controller details vary across implementations).

## 4. Phase 4 — Wall-clock parity runner

- [ ] 4.1 Add `tools/multilevel_bench_runner.py` — invokes Pulsim
      simulation + reads PLECS/PSIM export timings, computes the
      relative factor, reports per-circuit.
- [ ] 4.2 Gate ≤ 2× the slower competitor on each of the 4 circuits.

## 5. Phase 5 — Documentation + archive

- [ ] 5.1 `docs/benchmarks-and-parity.md` — document the RMS-error
      gates, the wall-clock contract, and the procedure for refreshing
      golden CSVs when PLECS / PSIM ships new versions.
- [ ] 5.2 `openspec archive add-multilevel-benchmark-parity --yes`.
