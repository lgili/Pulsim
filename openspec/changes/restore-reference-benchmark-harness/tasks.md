## 1. Metrics and reader
- [x] 1.1 `benchmarks/tools/metrics.py` with tests: ramp `y = t` on [0, 1] sampled 1000 times in [0, 0.01] and 10 times after → time-weighted mean 0.5000 (±1e-12), where the unweighted mean is 0.0104; `err_rms` of a trace against itself is 0; `err_rms` of `y + 0.1·span` is 0.1
- [ ] 1.2 `benchmarks/tools/v1_reader.py`: the mapping table of design.md decision 1; unit test per v1 component type with one hand-checked circuit each (`bridge_resistive_pure`: `V(a) = 6.6667 V`, `V(b) = 4.0000 V` within 1e-6) — OPEN: mapping table implemented; hand-checked: resistor/dc source (bridge), current source, ideal transformer (2x step-up), vcswitch + pwm schedule, pwm phase/dead-time; the other types are covered only by the triage run
- [x] 1.3 `benchmarks/manifests/v2_triage.yaml`: 126 entries. Test: count is 126; every `runs` entry loads and completes on `pwl`; every `skipped` entry raises its stated reason; a file on disk without an entry fails — 107 runs, 15 skipped, 4 error (status `error` with `match`)
- [x] 1.4 Port `benchmark_runner.py` to the reader and `pulsim.simulate`; delete the `pulsim_python_backend` import; `benchmark_ngspice.py` and the `Makefile` targets run — `run_pulsim` uses the reader; ngspice parity smoke-tested on 3 circuits with ngspice 45.2

## 2. Analytic reference set
- [x] 2.1 `benchmarks/analytic/`: the 20 cases of design.md decision 4, each with `exact(t)` and a self-check of the closed form (case 4: `v_C(∞) = 10 V`, first peak `10·(1 + e^{−απ/ω_d})` = 18.545 V at `t = π/ω_d`; case 9: 6.6667 V and 4.0000 V; case 15: 1.2 V, 12 W) — one module `benchmarks/analytic/__init__.py` (DECISIONS 3)
- [x] 2.2 `python/tests/test_benchmark_harness.py::test_analytic`: every case × engine that accepts it; initial ceilings on `err_rms` from the measured values with a factor of 2 — case 1: `pwl` (`dt = 10 µs`) ≤ 3.2e-3 (measured 1.59e-3), `trbdf2` (`rtol = 1e-6`) ≤ 1.2e-6 (6.0e-7), `dsed` ≤ 1e-6 on the trace's own grid; case 4: `pwl` (`dt = 1 µs`) ≤ 1.3e-3 (6.13e-4), `trbdf2` ≤ 1.9e-4 (9.24e-5) — ceilings on the trace's own grid: case 1 pwl 3.18e-3, trbdf2 4.52e-7, dsed 1e-6; case 4 pwl 1.23e-3, trbdf2 1.84e-4
- [x] 2.3 Non-vacuity: with a 1 % error injected into one component value, the case fails on every engine — three `known` defect cells are asserted to still fail instead (DECISIONS 9)
- [x] 2.4 `benchmarks/manifests/analytic.yaml` with the ceilings; a test that a ceiling in the file is not looser than in the previous commit unless the entry carries a `reason` — in-file 2x rule + `git show HEAD:` comparison (DECISIONS 4)

## 3. Differential runner
- [x] 3.1 `differential_runner.py`: outcomes, reference selection, `results.json`
- [ ] 3.2 Run it over the triaged suite; record each `disagree` and `error` cell as a triage entry with an issue reference, not as a tolerance — OPEN: 76 disagree/error cells over 60 circuits recorded as `known:` in v2_triage.yaml; no issue numbers filed (DECISIONS 10)
- [x] 3.3 Test on two fixed cells: the junction-capacitor RC is `refused` on `dsed` (or, before `add-engine-conformance-matrix`, `disagree`: 6.32 V against 1.02 V); the RC charge is `agree` on three engines — dsed today runs it and reads 6.32 V against 1.02 V -> not `agree`

## 4. Counters and performance gate
- [ ] 4.1 `RunCounters` in `solver/result.hpp`; increments in the fixed and Newton paths; reuse `CacheMetrics`, `TrBdf2Stats`, DSED stats; bind as `result.counters`; stubs via `scripts/generate_stubs.py` — OPEN: no kernel change in this pass; the gate reads the counters the kernel already reports (DECISIONS 1)
- [ ] 4.2 Gate: results of the whole existing suite are bit-identical with the counters added (they are increments only) — OPEN: nothing compiled in yet
- [ ] 4.3 `benchmarks/manifests/perf_counters.yaml` for 12 circuits × engines; test: `pwl` counters ≤ frozen, `trbdf2` / `dsed` ≤ 1.02 × frozen; a deliberate extra factorisation per step in a scratch build makes it fail — OPEN: frozen for 12 circuits x engines and gated; non-vacuity is an injected extra refactor per step in memory, not a scratch build
- [ ] 4.4 Inventory the wall-clock assertions in `python/tests`; convert each to a count or ratio or remove it; list them in the PR — OPEN: inventory in DECISIONS 11, not converted

## 5. Golden checkpoints
- [x] 5.1 Golden format, `--update-goldens`, comparison (exact integers and digest, `atol + rtol·|g|` for values)
- [x] 5.2 Freeze the 20 analytic circuits and at least 20 triaged circuits on `pwl`, after `fix-control-tick-semantics`, `update-composite-diode-forward-drop` and `fix-nonlinear-diode-passivity`, or re-freeze in those pull requests — 45 goldens (20 analytic + 25 triaged) frozen BEFORE the three Phase 0 changes; those pull requests re-freeze
- [ ] 5.3 Set each `rtol` from a run on the three CI platforms; caps 1e-9 (no Newton device) and 1e-6 — OPEN: generated and checked on macOS arm64 only; caps used as rtol until the reference-smoke job has run on Linux and Windows
- [x] 5.4 Non-vacuity: changing one resistor by 1e-6 relative fails the linear goldens

## 6. CI
- [x] 6.1 `ci.yml`: job "reference-smoke" (analytic, goldens, counters); non-blocking until one week green, then required
- [x] 6.2 `.github/workflows/nightly-reference.yml`: full differential run and ngspice comparison, artifact upload, issue on outcome change
- [x] 6.3 Local entry points in the `Makefile` (`make reference-smoke`, `make reference-nightly`) honouring the shared-machine rule (`-j2`, `nice`)

## 7. Docs
- [x] 7.1 `benchmarks/README.md`: layout, how to add an analytic case, how to re-freeze goldens and counters, what the legacy baselines are
- [x] 7.2 `CHANGELOG.md` [Unreleased]: harness restored, `result.counters` — no `result.counters` (4.1 open)
- [x] 7.3 `KNOWN_LIMITATIONS.md`: the triage summary (how many of 126 run, the skip reasons); first sample of algebraic nodes
- [ ] 7.4 `python/pulsim/AI_GUIDE.md`: `result.counters`; "compare engines with a time-weighted metric" — OPEN: `result.counters` does not exist yet; the guide already says to time-weight (AI_GUIDE.md:19)
- [x] 7.5 `CONTRIBUTING.md`: the reference-smoke job is part of the verification set

## 8. Verification
- [x] 8.1 `ruff check` (including `benchmarks/tools`) — the new tools are clean; `benchmark_runner.py` keeps its 4 pre-existing E741, as at HEAD
- [x] 8.2 `pytest python/tests` — 2352 passed, 4 skipped, 8 xfailed
- [ ] 8.3 `ctest -j2` — not run: this change touches no C++
- [x] 8.4 `openspec validate --all --strict`
- [x] 8.5 Stub check
- [ ] 8.6 One manual nightly run through `workflow_dispatch`; attach its summary table to the PR
