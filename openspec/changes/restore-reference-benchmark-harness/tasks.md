## 1. Metrics and reader
- [ ] 1.1 `benchmarks/tools/metrics.py` with tests: ramp `y = t` on [0, 1] sampled 1000 times in [0, 0.01] and 10 times after → time-weighted mean 0.5000 (±1e-12), where the unweighted mean is 0.0104; `err_rms` of a trace against itself is 0; `err_rms` of `y + 0.1·span` is 0.1
- [ ] 1.2 `benchmarks/tools/v1_reader.py`: the mapping table of design.md decision 1; unit test per v1 component type with one hand-checked circuit each (`bridge_resistive_pure`: `V(a) = 6.6667 V`, `V(b) = 4.0000 V` within 1e-6)
- [ ] 1.3 `benchmarks/manifests/v2_triage.yaml`: 126 entries. Test: count is 126; every `runs` entry loads and completes on `pwl`; every `skipped` entry raises its stated reason; a file on disk without an entry fails
- [ ] 1.4 Port `benchmark_runner.py` to the reader and `pulsim.simulate`; delete the `pulsim_python_backend` import; `benchmark_ngspice.py` and the `Makefile` targets run

## 2. Analytic reference set
- [ ] 2.1 `benchmarks/analytic/`: the 20 cases of design.md decision 4, each with `exact(t)` and a self-check of the closed form (case 4: `v_C(∞) = 10 V`, first peak `10·(1 + e^{−απ/ω_d})` = 18.545 V at `t = π/ω_d`; case 9: 6.6667 V and 4.0000 V; case 15: 1.2 V, 12 W)
- [ ] 2.2 `python/tests/test_benchmark_harness.py::test_analytic`: every case × engine that accepts it; initial ceilings on `err_rms` from the measured values with a factor of 2 — case 1: `pwl` (`dt = 10 µs`) ≤ 3.2e-3 (measured 1.59e-3), `trbdf2` (`rtol = 1e-6`) ≤ 1.2e-6 (6.0e-7), `dsed` ≤ 1e-6 on the trace's own grid; case 4: `pwl` (`dt = 1 µs`) ≤ 1.3e-3 (6.13e-4), `trbdf2` ≤ 1.9e-4 (9.24e-5)
- [ ] 2.3 Non-vacuity: with a 1 % error injected into one component value, the case fails on every engine
- [ ] 2.4 `benchmarks/manifests/analytic.yaml` with the ceilings; a test that a ceiling in the file is not looser than in the previous commit unless the entry carries a `reason`

## 3. Differential runner
- [ ] 3.1 `differential_runner.py`: outcomes, reference selection, `results.json`
- [ ] 3.2 Run it over the triaged suite; record each `disagree` and `error` cell as a triage entry with an issue reference, not as a tolerance
- [ ] 3.3 Test on two fixed cells: the junction-capacitor RC is `refused` on `dsed` (or, before `add-engine-conformance-matrix`, `disagree`: 6.32 V against 1.02 V); the RC charge is `agree` on three engines

## 4. Counters and performance gate
- [ ] 4.1 `RunCounters` in `solver/result.hpp`; increments in the fixed and Newton paths; reuse `CacheMetrics`, `TrBdf2Stats`, DSED stats; bind as `result.counters`; stubs via `scripts/generate_stubs.py`
- [ ] 4.2 Gate: results of the whole existing suite are bit-identical with the counters added (they are increments only)
- [ ] 4.3 `benchmarks/manifests/perf_counters.yaml` for 12 circuits × engines; test: `pwl` counters ≤ frozen, `trbdf2` / `dsed` ≤ 1.02 × frozen; a deliberate extra factorisation per step in a scratch build makes it fail
- [ ] 4.4 Inventory the wall-clock assertions in `python/tests`; convert each to a count or ratio or remove it; list them in the PR

## 5. Golden checkpoints
- [ ] 5.1 Golden format, `--update-goldens`, comparison (exact integers and digest, `atol + rtol·|g|` for values)
- [ ] 5.2 Freeze the 20 analytic circuits and at least 20 triaged circuits on `pwl`, after `fix-control-tick-semantics`, `update-composite-diode-forward-drop` and `fix-nonlinear-diode-passivity`, or re-freeze in those pull requests
- [ ] 5.3 Set each `rtol` from a run on the three CI platforms; caps 1e-9 (no Newton device) and 1e-6
- [ ] 5.4 Non-vacuity: changing one resistor by 1e-6 relative fails the linear goldens

## 6. CI
- [ ] 6.1 `ci.yml`: job "reference-smoke" (analytic, goldens, counters); non-blocking until one week green, then required
- [ ] 6.2 `.github/workflows/nightly-reference.yml`: full differential run and ngspice comparison, artifact upload, issue on outcome change
- [ ] 6.3 Local entry points in the `Makefile` (`make reference-smoke`, `make reference-nightly`) honouring the shared-machine rule (`-j2`, `nice`)

## 7. Docs
- [ ] 7.1 `benchmarks/README.md`: layout, how to add an analytic case, how to re-freeze goldens and counters, what the legacy baselines are
- [ ] 7.2 `CHANGELOG.md` [Unreleased]: harness restored, `result.counters`
- [ ] 7.3 `KNOWN_LIMITATIONS.md`: the triage summary (how many of 126 run, the skip reasons); first sample of algebraic nodes
- [ ] 7.4 `python/pulsim/AI_GUIDE.md`: `result.counters`; "compare engines with a time-weighted metric"
- [ ] 7.5 `CONTRIBUTING.md`: the reference-smoke job is part of the verification set

## 8. Verification
- [ ] 8.1 `ruff check` (including `benchmarks/tools`)
- [ ] 8.2 `pytest python/tests`
- [ ] 8.3 `ctest -j2`
- [ ] 8.4 `openspec validate --all --strict`
- [ ] 8.5 Stub check
- [ ] 8.6 One manual nightly run through `workflow_dispatch`; attach its summary table to the PR
