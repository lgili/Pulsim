## ADDED Requirements

### Requirement: Path-Aware Sweep Helper

The Python package SHALL expose
`pulsim.sweep.sweep_path_aware(builder, param_name, values,
t_end, dt, ...)` — a drop-in replacement for the legacy
`pulsim.sweep.sweep(...)` that exploits
`PwlStateSpaceCache.refactor_parametric` to amortise the
`analyze + factorize` cost across every sweep point.

The helper SHALL build a single `PwlStateSpaceCache` upfront,
then iterate `values` calling
`cache.refactor_parametric([param_name], [v])` between
simulation runs. The return value SHALL be a `SweepResult`
with the same shape as the legacy `sweep(...)` so callers can
swap APIs without restructuring downstream code.

If `pool.columns_affected_by_param(param_name)` returns empty
(parameter unknown to the pool, or the user passed a parameter
that drives a topology change rather than a value change), the
helper SHALL emit a `RuntimeWarning` and SHALL transparently
route to the legacy `sweep(...)` codepath, preserving
correctness at the cost of speed.

A multi-parameter overload `sweep_path_aware_nd(builder,
params_dict, t_end, dt, ...)` SHALL accept either a Cartesian
product of values per parameter or a callable that yields
parameter tuples (for Monte Carlo).

#### Scenario: Single-parameter sweep gives 10×+ speedup
- **GIVEN** a buck `CircuitBuilder` with `L_out` registered as
  a builder-time parameter
- **WHEN** `pulsim.sweep.sweep_path_aware(builder, "L_out",
  np.linspace(50e-6, 200e-6, 1000), t_end=10e-3, dt=100e-9)`
  is called
- **AND** the equivalent `pulsim.sweep.sweep(builder, "L_out",
  ...)` (legacy) call is timed for comparison
- **THEN** the path-aware version completes in at least 10×
  less wall time than the legacy version (verified by the
  benchmark suite on the captured Apple Silicon hardware)
- **AND** the per-sweep-point output waveforms match the legacy
  results within $10^{-9}$ on every state-vector entry

#### Scenario: Unknown parameter triggers fallback with warning
- **GIVEN** a builder with no parameter named `"flux_linkage"`
- **WHEN** `sweep_path_aware(builder, "flux_linkage", [1.0, 2.0,
  3.0], t_end=1e-3, dt=1e-6)` is called
- **THEN** a `RuntimeWarning` is emitted naming the unknown
  parameter
- **AND** the call still returns a valid `SweepResult`
  (computed via the legacy `sweep(...)` internally)
- **AND** the wall time is comparable to a direct legacy
  `sweep(...)` call (no speedup, but no regression either)

### Requirement: Path-Aware Monte Carlo Helper

The Python package SHALL expose
`pulsim.sweep.monte_carlo_path_aware(builder,
params_distributions, n_samples, t_end, dt, ...)` — a drop-in
replacement for `pulsim.sweep.monte_carlo(...)` that uses the
parametric-refactor path.

The helper SHALL draw `n_samples` parameter tuples from the
provided distributions, build the cache once, then for each
sample call `cache.refactor_parametric(all_params,
drawn_values)` followed by `run_transient(...)`. The return
value SHALL be a `MonteCarloResult` (existing structure) with
the same shape as the legacy variant.

Numerical equivalence with the legacy variant SHALL hold:
mean / variance / percentile KPIs derived from the two
variants on the same sampling seed SHALL match to at least
5 significant digits.

#### Scenario: 1000-sample Monte Carlo gives 25×+ speedup
- **GIVEN** a buck builder with `R_DS_on ~ Normal(20mΩ, 5mΩ)`
  and `L_out ~ Uniform(95µH, 105µH)` registered as Monte Carlo
  parameters
- **WHEN** `monte_carlo_path_aware(builder, {...},
  n_samples=1000, t_end=10e-3, dt=100e-9, seed=42)` is called
- **AND** the equivalent `monte_carlo(builder, {...}, ...,
  seed=42)` (legacy) is timed
- **THEN** path-aware completes in at least 25× less wall time
- **AND** the mean output voltage across the 1000 samples
  matches the legacy mean to 5 significant digits
- **AND** the 95th-percentile output voltage matches similarly
