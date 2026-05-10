## ADDED Requirements

### Requirement: Parameter Sweep API
The library SHALL provide a parameter sweep facility supporting Cartesian, Monte Carlo, Latin Hypercube, Sobol, and Halton sampling.

#### Scenario: Cartesian sweep
- **GIVEN** a sweep with `parameters = {"vin": [36, 48, 60], "rload": [5, 10]}`
- **WHEN** the sweep runs
- **THEN** 6 simulations execute (3×2 cartesian product)
- **AND** each result is paired with its parameter combination

#### Scenario: Monte Carlo sweep with normal distribution
- **GIVEN** `parameters = {"l_value": Distribution.normal(mean=47e-6, std=4.7e-6, n=1000)}`
- **WHEN** the sweep runs with seed=42
- **THEN** 1000 simulations execute with seeded i.i.d. samples
- **AND** the sample mean is within 1σ/√n of the configured mean

#### Scenario: Latin Hypercube sampling
- **GIVEN** `parameters = LatinHypercube({"l": ..., "c": ..., "r": ...}, n=200, seed=42)`
- **WHEN** the sweep runs
- **THEN** 200 samples cover the joint space with reduced variance vs i.i.d.
- **AND** repeat with seed=42 yields identical sample matrix

### Requirement: Reproducible Sweep Identity
A sweep SHALL be reproducible: identical inputs (seed, parameter spec, circuit factory, metric list) produce identical sample matrices and metric matrices.

#### Scenario: Reproducibility hash
- **GIVEN** two sweep invocations with identical inputs
- **WHEN** both runs complete
- **THEN** the metric matrices are bit-identical
- **AND** the run-identifier hashes match

### Requirement: Parallel Execution Backends
The library SHALL provide `serial`, `joblib`, and `dask` executors with declarative selection.

#### Scenario: Joblib executor scaling
- **GIVEN** a sweep of 100 samples on a 4-core machine
- **WHEN** `executor="joblib", n_workers=4` is specified
- **THEN** wallclock time approaches `single_run_time * 100 / 4` within 30% overhead
- **AND** results are independent of execution order

#### Scenario: Failure isolation
- **GIVEN** a sweep where 5 samples fail convergence
- **WHEN** the sweep completes
- **THEN** the 95 successful samples produce metrics
- **AND** the 5 failures are captured in `SweepResult.failed` with reason and parameters

### Requirement: GPU Backend for PWL Sweeps
The library SHALL support a GPU-batched executor restricted to PWL state-space mode when topology shape is identical across samples.

#### Scenario: GPU sweep on PWL buck
- **GIVEN** 10000 samples of a PWL-mode buck (only parameter values vary, topology shape fixed)
- **WHEN** `executor="gpu"` is specified
- **THEN** the sweep runs in ≤10× the time of an equivalent CPU joblib sweep
- **AND** numerical agreement with CPU is within 1e-6 per metric

#### Scenario: GPU fallback
- **GIVEN** a sweep where parameter values lead to different topology shapes
- **WHEN** `executor="gpu"` is requested
- **THEN** the sweep falls back to CPU joblib with a warning
- **AND** the warning identifies the constraint violation

### Requirement: Metric Extraction Library
The library SHALL provide a metric extraction layer covering steady-state value, peak, RMS, THD, settling time, efficiency, and custom callables.

#### Scenario: Steady-state metric on RC
- **GIVEN** an RC circuit with step input
- **WHEN** `Metric.steady_state("vout", t_window=(0.99 * tstop, tstop))` is applied
- **THEN** the returned value matches analytical steady state within 0.1%

#### Scenario: THD metric on inverter output
- **GIVEN** an inverter producing 50 Hz fundamental with harmonics
- **WHEN** `Metric.thd("v_phase_a", fundamental_freq=50)` is applied
- **THEN** the returned THD matches FFT-based reference within 1%

### Requirement: Sweep Result Aggregation
The `SweepResult` SHALL expose pandas/xarray export, percentile queries, and failed-sample inspection.

#### Scenario: Percentile query
- **GIVEN** a 1000-sample MC result on `efficiency`
- **WHEN** `result.percentile("efficiency", q=5)` is called
- **THEN** the 5th percentile efficiency is returned
- **AND** can be used in yield gating

#### Scenario: Pandas export
- **WHEN** `result.to_pandas()` is called
- **THEN** a DataFrame with one row per sample and columns `param_*` + `metric_*` is returned
