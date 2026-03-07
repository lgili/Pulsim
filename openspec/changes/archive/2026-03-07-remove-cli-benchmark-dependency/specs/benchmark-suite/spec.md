# benchmark-suite

## ADDED Requirements

### Requirement: Python-First Benchmark Execution
The benchmark suite SHALL execute circuit scenarios through Python runtime APIs backed by the v1 kernel and SHALL NOT require an external `pulsim` executable.

#### Scenario: Run benchmark suite without CLI binary
- **WHEN** `benchmark_runner.py` is executed in an environment without a `pulsim` executable
- **THEN** scenarios are executed through Python runtime bindings
- **AND** the run produces standard benchmark artifacts

### Requirement: Backend-Independent Validation Outcomes
The benchmark suite SHALL produce validation outcomes independent of backend type and SHALL NOT mark scenarios as skipped due to missing CLI-only paths.

#### Scenario: Matrix run without CLI binary
- **WHEN** `validation_matrix.py` executes scenarios without an external `pulsim` executable
- **THEN** each scenario returns explicit `passed`, `failed`, or `baseline` status based on validation rules
- **AND** no scenario is skipped solely because execution is via Python runtime

### Requirement: Periodic Scenario Coverage in Matrix
The benchmark suite SHALL execute periodic steady-state scenarios (shooting and harmonic balance) through the Python runtime path.

#### Scenario: Periodic benchmark scenario
- **WHEN** a scenario declares shooting or harmonic balance options
- **THEN** the benchmark runner invokes the corresponding periodic runtime method
- **AND** emits result artifacts and validation status for that scenario

### Requirement: Structured Telemetry Source
Benchmark telemetry SHALL be collected from structured simulation result fields rather than command stdout parsing.

#### Scenario: Collect solver telemetry
- **WHEN** a scenario completes
- **THEN** nonlinear iterations, linear iterations, step counts, rejections, and runtime are read from simulation results
- **AND** telemetry is written to `results.json` in a stable schema

### Requirement: Runtime Parity for ngspice Comparator
The ngspice comparator SHALL use the same Python runtime simulation path as the core benchmark runner.

#### Scenario: Pulsim vs ngspice comparison
- **WHEN** `benchmark_ngspice.py` executes a mapped benchmark pair
- **THEN** Pulsim waveforms are generated through the Python runtime path used by the benchmark suite
- **AND** comparison metrics are computed from those outputs
