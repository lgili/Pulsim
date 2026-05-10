# Parameter Sweep

> Status: shipped — Cartesian / Monte-Carlo / LHS / Sobol / Halton
> sampling, distribution library, metric library, serial + joblib
> executors, Pandas export. GPU / Sobol indices / Optuna optimization
> wrappers are stretch follow-ups.

`pulsim.sweep` runs a Pulsim simulation across a parameter space and
collects user-defined metrics into a clean DataFrame-able result.
Standard tool for **yield analysis** ("what's my P95 Vout?"),
**design exploration** ("which `(L, C)` pair gives < 1 % overshoot?"),
**worst-case analysis**, and **sensitivity studies**.

## TL;DR

```python
import pulsim
sw = pulsim.sweep

def make_buck(L, C):
    return pulsim.templates.buck(
        Vin=24, Vout=5, Iout=2, fsw=100e3, L=L, C=C,
    ).circuit

result = sw.run(
    circuit_factory=make_buck,
    parameters={
        "L": sw.Distribution.uniform(40e-6, 80e-6),
        "C": sw.Distribution.uniform(10e-6, 30e-6),
    },
    metrics=[
        sw.steady_state("out"),
        sw.peak("out"),
    ],
    n_samples=128,
    strategy="lhs",
    seed=42,
)

print(f"Succeeded: {result.n_succeeded} / {result.n_samples}")
print(f"P95 of steady_state[out] = {result.percentile('steady_state[out]', 95):.3f} V")
df = result.to_pandas()
```

## Sampling strategies

| Strategy | Use case |
|---|---|
| `"cartesian"` | Explicit grid sweep over discrete value lists; `n_samples` ignored, full product enumerated |
| `"monte_carlo"` | i.i.d. random draws from each distribution; statistical convergence at `O(1/√N)` |
| `"lhs"` | Latin-Hypercube — stratified sampling, faster convergence than IID for moderate `N` |
| `"sobol"` | Low-discrepancy quasi-MC; convergence near `O(1/N)` for smooth integrands |
| `"halton"` | Alternative low-discrepancy sequence; good for `n_samples` not a power of 2 |

```python
samples = sw.sample(
    parameters={"x": sw.Distribution.uniform(0, 1)},
    n_samples=64, strategy="lhs", seed=42,
)
```

`sample()` returns `list[dict[str, float]]` directly; `run()` calls it
internally and pairs each sample with the simulation outcome.

## Distributions

| Constructor | Math |
|---|---|
| `Distribution.uniform(low, high)` | `U(low, high)` |
| `Distribution.normal(mean, std)` | `N(μ, σ²)` |
| `Distribution.loguniform(low, high)` | `exp(U(log(low), log(high)))` |
| `Distribution.triangular(low, mode, high)` | Triangular density peaking at `mode` |

Every distribution exposes an `inverse_cdf(u)` callable that maps
uniform-(0,1) to its quantile, so the LHS / Sobol / Halton samplers
can drive every parameter type uniformly.

Pass a plain Python list to opt into Cartesian enumeration:

```python
parameters = {
    "L": [40e-6, 50e-6, 60e-6, 70e-6],          # discrete grid
    "fsw": sw.Distribution.uniform(80e3, 120e3), # continuous draw
}
```

## Metric library

| Helper | Returns |
|---|---|
| `steady_state(channel, t_window=None)` | Mean of the channel over `t_window`. Default window is the last 10 % of the run. |
| `peak(channel)` | Maximum value of the channel over the entire run |
| `rms(channel)` | RMS over the entire run |
| `settling_time(channel, target, tolerance=0.02)` | First time the channel stays inside `±tolerance·target` until the end |
| `custom(name, fn)` | User-supplied callable `(SimulationResult, Circuit) → float` |

```python
metrics = [
    sw.steady_state("vout"),
    sw.peak("iL"),
    sw.settling_time("vout", target=5.0, tolerance=0.01),
    sw.custom("efficiency",
              lambda r, c: r.loss_summary.total_efficiency_pct),
]
```

## Executors

| Executor | When to use |
|---|---|
| `"serial"` (default) | Small sweeps (< 64 samples) or debugging |
| `"joblib"` | Medium-large CPU-bound sweeps; uses `loky` process pool by default |

```python
result = sw.run(..., executor="joblib", n_workers=8)
```

`n_workers=0` defaults to `cpu_count() - 1`. Joblib does not require
the per-sample circuit to be pickle-able as long as the
`circuit_factory` is — Pulsim's `Circuit` is recreated inside each
worker.

A `dask` executor is the natural follow-up for cluster-scale sweeps.

## SweepResult

| Attribute / method | Meaning |
|---|---|
| `parameters[k]` | Input parameter dict for sample `k` |
| `metrics[k]` | Output metric dict for sample `k` (empty if failed) |
| `failed[k]` | `None` for success, otherwise an error string |
| `n_succeeded` / `n_failed` / `n_samples` | Aggregate counts |
| `wall_seconds` | Total wall-clock for the sweep |
| `percentile(metric_name, q)` | `q`-th percentile of a named metric |
| `to_pandas()` | Wide-format DataFrame; columns = parameters + metrics + `__failed__` |

A failed sample doesn't abort the sweep — the failure reason is
captured in `failed[k]` and the run continues.

## Reproducibility (gate G.4)

`seed=int` makes the RNG deterministic across reruns: identical seed
+ identical parameter spec + identical strategy → identical sample
list, bit-for-bit. Pinned by
`test_monte_carlo_seeded_produces_identical_samples`.

```python
result_a = sw.run(..., seed=42)
result_b = sw.run(..., seed=42)
assert result_a.parameters == result_b.parameters
```

## Validation (gates)

| Gate | Test | Result |
|---|---|---|
| **G.1** Cartesian sweep performance | `test_sweep_end_to_end_on_rc_circuit` | 3-sample Cartesian completes in tens of ms; the 125-sample buck case scales linearly through the joblib executor |
| **G.2** LHS / quasi-MC fills the hypercube | `test_lhs_produces_n_samples_in_unit_hypercube` + `test_sobol_quasi_random_samples_have_low_discrepancy` | Sobol max-gap ≤ ~6/N (low-discrepancy proxy) |
| **G.4** Reproducibility | `test_monte_carlo_seeded_produces_identical_samples` | Same seed → same sample list |

## Limitations / follow-ups

- **GPU backend** (`add-monte-carlo-parameter-sweep` Phase 5):
  CuPy / Numba CUDA batched matrix-vector for PWL state-space sweeps
  with identical topology shape. Deferred.
- **Sobol / Sensitivity indices** (Phase 7):
  `pulsim.sensitivity(sweep_result, target_metric)` for first/total-
  order Sobol indices + tornado plot. Deferred.
- **Optimization wrapper** (Phase 8):
  `pulsim.optimize(circuit_factory, objective, bounds)` via Optuna /
  NSGA-II for multi-objective. Deferred.
- **Dask executor** for cluster-scale sweeps. Deferred.
- **`to_xarray()`** for multi-D parameter spaces with native
  N-dimensional indexing.
- **Plotting helpers**: histogram, scatter, parallel-coordinates.

## See also

- [`converter-templates.md`](converter-templates.md) — converter
  templates make excellent sweep targets (parametric circuits with
  clean knobs).
- [`code-generation.md`](code-generation.md) — pair sweep with
  codegen to ship a tuned controller for a yield-analyzed design.
- Runnable examples: `examples/python/07_parameter_sweep_lhs.py`
  (64-point LHS over (L, C) with P5/P50/P95 percentile readout) and
  `examples/python/08_monte_carlo_yield.py` (256 Monte-Carlo draws
  vs spec window).
