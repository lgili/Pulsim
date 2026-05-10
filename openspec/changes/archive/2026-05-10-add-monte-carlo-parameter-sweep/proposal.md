## Why

Engineering decisions in power electronics aren't single-point: tolerance studies, worst-case analysis, design-of-experiments, sensitivity to component drift, Monte Carlo for yield prediction. PSIM has parametric sweep; PLECS has Monte Carlo and parameter variation. Without these, every "what if" study is a manual loop in Python that wastes time and is hard to reproduce.

This change adds a first-class **parameter sweep** facility that:

- Varies any parameter (or set) over discrete or distribution-sampled values.
- Runs all variations in parallel (multi-CPU, optionally multi-GPU).
- Aggregates results into structured tables with metric extraction (peak, RMS, settling time, etc.).
- Reproducible: same seed → same sequence of samples.

Open-source references:
- **Joblib** for embarrassingly-parallel CPU dispatch.
- **Dask** for distributed sweeps if needed.
- **CuPy / Numba** for GPU-batched runs (advanced).
- **Optuna** / **Hyperopt** for parameter optimization (post-MVP wrapping).

## What Changes

### Sweep Specification
- New `pulsim.sweep(circuit_factory, parameters, metrics, executor='joblib')`:
```python
results = pulsim.sweep(
    circuit_factory=lambda p: pulsim.templates.buck(vin=p.vin, lout=p.lout, ...),
    parameters={
        "vin":   [36, 42, 48, 54, 60],
        "lout":  Distribution.normal(mean=47e-6, std=4.7e-6, n=20),
        "rload": [5, 7.5, 10, 12.5, 15],
    },
    metrics=[
        Metric.steady_state("vout", target=12.0),
        Metric.peak("i_inductor"),
        Metric.efficiency(),
    ],
    executor="joblib",
    n_workers=8,
    seed=42,
)
```
- `parameters` accepts: lists (cartesian sweep), distributions (Monte Carlo sampling), tuples of correlated samples.
- `metrics` accepts a list of `Metric` objects extracted from each simulation result.

### Sampling Strategies
- **Cartesian** — all combinations of discrete lists.
- **Monte Carlo** — i.i.d. samples from `Normal`, `Uniform`, `LogUniform`, `Triangular`, `Beta`, `Custom`.
- **Latin Hypercube** — variance-reduced sampling for high-dim spaces.
- **Sobol / Halton** — quasi-random for design-of-experiments.

### Parallel Execution Backends
- `executor="serial"` — debug, single thread.
- `executor="joblib"` — default, CPU parallelism via process pool.
- `executor="dask"` — distributed across cluster (post-MVP).
- `executor="gpu"` — Numba/CuPy batched (advanced; restricts to PWL state-space mode and identical topology shape).

### Metric Library
- `Metric.steady_state(channel, t_window=None)` — final value averaged over window.
- `Metric.peak(channel)`, `Metric.rms(channel)`, `Metric.thd(channel, fundamental_freq)`.
- `Metric.settling_time(channel, target, tolerance)`.
- `Metric.efficiency()` — `P_out / P_in` from telemetry.
- `Metric.custom(fn)` — user lambda over `SimulationResult`.

### Result Aggregation
- `SweepResult` exposes:
  - `result.parameters` — input matrix.
  - `result.metrics` — output matrix (rows=samples, cols=metrics).
  - `result.to_pandas()` for analysis.
  - `result.percentile(metric, q)` for yield analysis.
  - `result.failed` — list of samples that failed convergence with reason.

### Sensitivity & Optimization (Stretch)
- `pulsim.sensitivity(sweep_result, target_metric)` — Sobol sensitivity indices.
- `pulsim.optimize(circuit_factory, objective, bounds, ...)` — Optuna-backed wrapper.

### Telemetry
- Per-sample wallclock and convergence stats aggregated.
- Failed-sample report with topology trace and failure reasons.

## Impact

- **New capability**: `parameter-sweep`.
- **Affected specs**: `parameter-sweep` (new), `python-bindings`.
- **Affected code**: new `python/pulsim/sweep/` module (mostly Python; minor C++ for state cloning), additions to telemetry export.
- **Performance**: linear scaling on n_workers up to memory bound.

## Success Criteria

1. **Cartesian sweep**: 5×5×5 = 125-sample buck sweep completes in ≤2× single-run wallclock × 125 / n_workers.
2. **Monte Carlo**: 1000-sample Latin Hypercube on 5 parameters produces meaningful percentile bands.
3. **GPU backend**: 10000-sample sweep on PWL buck model runs at ≥10× speedup over CPU joblib.
4. **Reproducibility**: identical `seed` → identical metric matrix.
5. **Tutorial**: yield analysis on a flyback (component tolerances) showing >95% pass rate with explicit margins.
