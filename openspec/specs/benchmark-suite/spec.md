# benchmark-suite Specification

## Purpose
TBD - created by archiving change remove-cli-benchmark-dependency. Update Purpose after archive.
## Requirements
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

### Requirement: YAML-first benchmark circuits
The system SHALL define benchmark circuits as YAML netlists compliant with the `netlist-yaml` capability and the `pulsim-v1` schema.

#### Scenario: Load a benchmark circuit
- **WHEN** a benchmark circuit is loaded
- **THEN** the YAML netlist is parsed using the `pulsim-v1` schema
- **AND** the circuit is accepted only if it conforms to `netlist-yaml`

### Requirement: Benchmark metadata
The system SHALL allow benchmark metadata to be provided via a `benchmark` block embedded in the YAML netlist or via a sidecar YAML file.

#### Scenario: Resolve benchmark metadata
- **WHEN** a benchmark circuit is loaded
- **THEN** metadata is resolved from the embedded `benchmark` block
- **OR** a sidecar YAML file if the embedded block is absent

### Requirement: Scenario matrix
The system SHALL allow a benchmark run to define multiple scenarios per circuit, each with solver and integrator settings.

#### Scenario: Run multiple scenarios
- **WHEN** a circuit defines multiple scenarios
- **THEN** the benchmark runner executes each scenario with its specified solver and integrator configuration

### Requirement: Benchmark artifacts
The system SHALL emit standardized benchmark artifacts for each run: `results.csv`, `results.json`, and `summary.json`.

#### Scenario: Produce standardized outputs
- **WHEN** a benchmark run completes
- **THEN** `results.csv` contains per-scenario numeric outputs
- **AND** `results.json` contains structured metadata and telemetry
- **AND** `summary.json` contains pass/fail validation results

### Requirement: Telemetry fields
The system SHALL record solver telemetry including nonlinear iterations, linear iterations, step count, residual norms, and wall-clock runtime.

#### Scenario: Record telemetry
- **WHEN** a benchmark scenario completes
- **THEN** telemetry fields are included in `results.json`

### Requirement: Validation types
The system SHALL support the validation types `analytical`, `reference`, and `ngspice` for benchmark comparisons.

#### Scenario: Validate against a reference
- **WHEN** a benchmark uses validation type `reference`
- **THEN** outputs are compared against the stored baseline data

### Requirement: Validation matrix
The system SHALL provide a validation matrix runner that executes solver and integrator combinations across the benchmark corpus.

#### Scenario: Execute validation matrix
- **WHEN** a validation matrix is invoked
- **THEN** the runner executes each solver/integrator combination for the selected circuits

