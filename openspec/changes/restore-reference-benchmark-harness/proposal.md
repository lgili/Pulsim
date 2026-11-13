## Why

The core roadmap changes the numerical method of every engine. The
repository has a reference suite for that, and none of it runs:

- All **126** circuits under `benchmarks/circuits/` declare
  `schema: pulsim-v1`, which the 2.0 loader refuses by design
  (`core/include/pulsim/yaml/loader.hpp:669-682`).
- `benchmarks/tools/benchmark_runner.py:22-24` imports
  `pulsim_python_backend`, a module that is not in the repository;
  `benchmark_ngspice.py` imports from the runner, and the `Makefile`
  benchmark targets (`:306-340`) call both.
- CI (`.github/workflows/ci.yml`) runs ctest, pytest and ruff. No job
  touches `benchmarks/`.
- The 70 files in `benchmarks/baselines/` were captured from the v1
  engine with `--generate-baselines` (`benchmarks/README.md:153-155`):
  they are self-references of a retired solver.
- What does run is narrow: 7 of 127 Python test files iterate over
  engines; the only performance gate is the allocation test
  `core/tests/perf/test_zero_alloc.cpp`; there is no external
  reference for any device model, although 66 ngspice netlists sit in
  `benchmarks/spice/ngspice/` and ngspice 45.2 is installed at
  `/opt/homebrew/bin/ngspice`.

Two recent observations (reported by the owner, not measured here) fix
the shape of the replacement: a Windows CI run differed from macOS by
9e-5 on a `dsed` result, so bit-exact hashes of floating-point output
cannot be the regression key; and a wall-clock assertion flaked in CI,
so time cannot be the performance key. A third is recorded in the
owner's notes: an unweighted `.mean()` over a variable-step trace once
reported 3.04 V where the time-weighted value was 6.33 V and was filed
as an engine defect.

Without a harness, the fixed-engine restart rule, the stepper
unification and the device-interface port have no oracle beyond the
unit tests written for the code they replace.

## What Changes

- **A harness-only reader for the v1 benchmark circuits.**
  `benchmarks/tools/v1_reader.py` turns a `pulsim-v1` benchmark file
  into a `CircuitBuilder`, a gate schedule and run options, in memory.
  The 126 files, their `benchmark:` metadata and their pairing with the
  ngspice netlists stay as they are. Every circuit is listed in a triage
  manifest as `runs` or `skipped: <reason>`; nothing is skipped
  silently.
- **`benchmark_runner.py` runs on `pulsim.simulate`**; the missing
  import is removed.
- **Differential runner**, circuit × engine (`pwl`, `trbdf2`, `dsed`):
  each cell is `agree`, `refused` (typed refusal) or `disagree`, against
  the best reference available (analytic, ngspice, or the other
  engines). All waveform metrics are time-weighted.
- **Analytic reference set**: 20 circuits with closed-form solutions
  (listed in design.md), each with a tolerance per engine.
- **Golden checkpoints of the fixed engine**: per circuit, exact
  integers (step count, switch-state sequence digest) and state values
  at 8 instants compared with a recorded tolerance, so refactors that
  must not change `engine='pwl'` are provable on every CI platform.
- **Counter-based performance gate**: factorisations, solves, Newton
  iterations and cache builds per circuit × engine, compared with a
  frozen table. Wall time is reported and never asserted.
- **Nightly ngspice job** comparing every circuit that has a netlist.
- CI: a fast job on every pull request (analytic set, goldens, counter
  gate) and the nightly job (all 126 × engines, ngspice).
- No kernel behaviour changes. Counters that do not exist yet are added
  as increments only.

## Impact

- Affected specs: `benchmark-suite` (two requirements that name the v1
  kernel and the v1 loader are modified; six are added).
- Affected code: `benchmarks/tools/` (`v1_reader.py`,
  `differential_runner.py`, `metrics.py` new; `benchmark_runner.py`,
  `benchmark_ngspice.py` ported), `benchmarks/manifests/`
  (`v2_triage.yaml`, `analytic.yaml`, `perf_counters.yaml` new),
  `benchmarks/goldens/pwl/` (new), `benchmarks/analytic/` (new),
  `python/tests/test_benchmark_harness.py` (new),
  `.github/workflows/ci.yml` (one job), a new
  `.github/workflows/nightly-reference.yml`, `Makefile` benchmark
  targets, `benchmarks/README.md`. Kernel: a `RunCounters` record on
  the fixed-engine and Newton paths
  (`core/include/pulsim/solver/result.hpp`, `pwl/nonlinear_solve.hpp`),
  bound in `python/bindings.cpp`; stubs.
- Dependencies: none to start. It is the "harness" that
  `update-fixed-step-loop-bookkeeping`, `update-newton-persistent-solver`,
  `update-trbdf2-factor-reuse`, `update-linear-time-assembly`,
  `add-discontinuity-restart-rule`, `update-trbdf2-post-event-restart`,
  `add-unified-device-interface` and `refactor-unified-transient-stepper`
  list under "Needs". The Phase 0 changes that move `pwl` numbers
  (`fix-control-tick-semantics`, `update-composite-diode-forward-drop`,
  `fix-nonlinear-diode-passivity`) should land before the goldens are
  frozen, or the goldens are re-frozen in their pull requests. The
  differential runner uses `UnsupportedError` when
  `add-engine-conformance-matrix` has landed and `ValueError` otherwise.
- Users affected: none at run time. Contributors gain two CI jobs and a
  documented way to re-freeze goldens and counters.
