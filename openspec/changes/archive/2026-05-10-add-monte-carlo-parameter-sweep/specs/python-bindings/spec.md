## ADDED Requirements

### Requirement: Sweep API in Python
Python bindings SHALL expose `pulsim.sweep(circuit_factory, parameters, metrics, executor, n_workers, seed)` returning a `SweepResult` object.

#### Scenario: Sweep call
- **WHEN** Python code calls `pulsim.sweep(circuit_factory, params_dict, metrics_list, executor="joblib", n_workers=8, seed=42)`
- **THEN** the sweep executes per spec
- **AND** returns a `SweepResult` with `to_pandas()`, `percentile()`, `failed` accessors

### Requirement: Distribution Helper Classes
Python bindings SHALL expose `pulsim.Distribution.{normal, uniform, loguniform, triangular, beta, custom}` for parameter declaration.

#### Scenario: Custom distribution
- **GIVEN** a Python callable `f(rng) -> sample`
- **WHEN** `Distribution.custom(f, n=500)` is supplied as a parameter spec
- **THEN** the sweep draws 500 samples via `f` with the seeded RNG

### Requirement: Sensitivity and Optimization Wrappers
Python bindings SHALL expose `pulsim.sensitivity(sweep_result, target_metric)` and (stretch) `pulsim.optimize(circuit_factory, objective, bounds, ...)`.

#### Scenario: Sensitivity from sweep
- **GIVEN** a Sobol sweep on 5 parameters and a target metric
- **WHEN** `pulsim.sensitivity(result, "efficiency")` is called
- **THEN** first-order and total-order Sobol indices per parameter are returned
- **AND** the values are deterministic given the seed
