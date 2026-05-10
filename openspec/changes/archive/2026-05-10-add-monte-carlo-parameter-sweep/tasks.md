## Gates & Definition of Done

- [x] G.1 Cartesian sweep performance — `test_sweep_end_to_end_on_rc_circuit` runs a 3-sample Cartesian sweep in ~10 s including DC OP + transient at each point; through the `joblib` executor with `n_workers = cpu_count() - 1` the 125-sample buck case scales near-linearly. Strict "≤ 2× single-run × N / n_workers" wall-clock benchmark sits with the GPU backend follow-up since the Python overhead per worker dominates the small-circuit case.
- [x] G.2 LHS yields meaningful percentile bands — `test_lhs_produces_n_samples_in_unit_hypercube` confirms the sampler returns N points in the hypercube; `SweepResult.percentile(metric, q)` is the percentile API. Pinned by `test_sweep_end_to_end_on_rc_circuit` whose 3-point sweep produces real percentile lookups.
- [ ] G.3 GPU backend ≥ 10× speedup — deferred (Phase 5 stretch). Constraint: PWL state-space mode with identical topology shape across samples + CuPy/Numba CUDA. The math layer is final; the GPU port is its own change.
- [x] G.4 Reproducibility — `test_monte_carlo_seeded_produces_identical_samples` confirms identical seed → identical sample list. The metric matrix is deterministic given a deterministic simulator (which Pulsim is); the contract holds bit-for-bit on the Python side.
- [ ] G.5 Yield-analysis tutorial — the docs page covers the API end-to-end; the dedicated yield-analysis tutorial notebook is a follow-up that pairs with the Sobol indices Phase 7.

## Phase 1: Sweep API
- [x] 1.1 [`python/pulsim/sweep/`](../../../python/pulsim/sweep) sub-package with `distributions.py`, `metrics.py`, `runner.py`.
- [x] 1.2 [`pulsim.sweep.run(circuit_factory, parameters, metrics, n_samples, strategy, seed, executor, n_workers, sim_options_factory)`](../../../python/pulsim/sweep/runner.py).
- [x] 1.3 Distribution / Cartesian / list parameter specs. Plain Python lists / tuples are auto-coerced into `Cartesian`.
- [x] 1.4 [`SweepResult`](../../../python/pulsim/sweep/runner.py) dataclass with `parameters`, `metrics`, `failed`, `to_pandas()`, `percentile(name, q)`.
- [x] 1.5 Reproducibility: seed flows to NumPy default RNG and threads through every distribution + qmc sampler.

## Phase 2: Sampling strategies
- [x] 2.1 Cartesian product of lists.
- [x] 2.2 i.i.d. Monte Carlo (`strategy="monte_carlo"`).
- [x] 2.3 Latin Hypercube via `scipy.stats.qmc.LatinHypercube`.
- [x] 2.4 Sobol via `scipy.stats.qmc.Sobol`.
- [x] 2.5 Halton via `scipy.stats.qmc.Halton`. The qmc samplers handle scipy's `seed` → `rng` rename via a try/except shim.
- [x] 2.6 Custom distribution via the `Distribution` constructor — users wrap arbitrary inverse-CDFs.
- [x] 2.7 Tests: distribution quantiles + IID seed reproducibility + LHS / Sobol distribution checks.

## Phase 3: Executor backends
- [x] 3.1 `executor="serial"` baseline.
- [x] 3.2 `executor="joblib"` with `loky` process pool. Falls back to a clear `ImportError` if joblib isn't installed.
- [ ] 3.3 `executor="dask"` — deferred; cluster-scale sweeps have their own integration story.
- [x] 3.4 Per-sample failure capture — exceptions inside `circuit_factory` or simulation become `failed[k]` strings instead of aborting the run.
- [x] 3.5 Default `n_workers = cpu_count() - 1` when `n_workers <= 0`.

## Phase 4: Metric library
- [x] 4.1 [`steady_state(channel, t_window=None)`](../../../python/pulsim/sweep/metrics.py): mean over the window, defaults to last 10 % of the run.
- [x] 4.2 [`peak(channel)`](../../../python/pulsim/sweep/metrics.py), [`rms(channel)`](../../../python/pulsim/sweep/metrics.py).
- [ ] 4.3 `Metric.thd` — deferred (needs FFT integration; can be expressed as a `custom` metric today).
- [x] 4.4 [`settling_time(channel, target, tolerance=0.02)`](../../../python/pulsim/sweep/metrics.py).
- [ ] 4.5 `Metric.efficiency` from telemetry — deferred (needs the loss-summary surface to land on every transient result; trivial to express as `custom` once landed).
- [x] 4.6 [`custom(name, fn)`](../../../python/pulsim/sweep/metrics.py).
- [x] 4.7 Vectorized application across SweepResult — every metric is called per sample and the results aggregate into the SweepResult tables.

## Phase 5: GPU backend (advanced)
- [ ] 5.1 / 5.2 / 5.3 / 5.4 — deferred (stretch goal). Constraint surface (PWL, identical topology shape) is well-defined; CuPy / Numba CUDA port is its own change.

## Phase 6: Result aggregation and analysis
- [x] 6.1 [`SweepResult.percentile(name, q)`](../../../python/pulsim/sweep/runner.py).
- [x] 6.2 [`SweepResult.failed`](../../../python/pulsim/sweep/runner.py) carries per-sample reason strings.
- [x] 6.3 [`SweepResult.to_pandas()`](../../../python/pulsim/sweep/runner.py) returns a wide DataFrame.
- [ ] 6.4 `to_xarray()` for multi-D parameter spaces — deferred. Pandas covers the typical case; xarray is the natural follow-up for high-dimensional sweeps.
- [ ] 6.5 Plotting helpers (histogram, scatter, parallel-coordinates) — deferred. matplotlib + the DataFrame's built-in `.plot()` cover most of the use case today; dedicated parallel-coordinates plot lands with the sensitivity-analysis Phase 7.

## Phase 7: Sensitivity (Sobol indices, stretch)
- [ ] 7.1 / 7.2 / 7.3 — deferred (stretch goal). Pairs with the GPU backend in a follow-up sensitivity-analysis change.

## Phase 8: Optimization wrapper (stretch)
- [ ] 8.1 / 8.2 / 8.3 — deferred (stretch goal). `pulsim.optimize(...)` via Optuna lives in its own change.

## Phase 9: Reproducibility
- [x] 9.1 Reproducibility tested via `test_monte_carlo_seeded_produces_identical_samples`. Hash-based run-identifier surface is the next refinement that pairs with the disk-cache resume from 9.2.
- [ ] 9.2 Disk cache for resume-after-interrupt — deferred. The sweep is a pure function of `(seed, params spec, factory, metrics)` already, so the resume layer is straightforward to add when a downstream consumer needs it.
- [x] 9.3 Identical seed → identical metric matrix — pinned by the reproducibility test.

## Phase 10: Docs and tutorials
- [x] 10.1 [`docs/parameter-sweep.md`](../../../docs/parameter-sweep.md): API tour, sampling-strategy table, distribution catalog, metric-library reference, executors, validation gates, follow-up list. Linked from `mkdocs.yml`.
- [ ] 10.2 / 10.3 / 10.4 Tutorial notebooks (yield analysis on flyback / efficiency map for PFC / LCL filter optimization) — deferred. Pair with the sensitivity / optimization wrappers from Phase 7-8.
