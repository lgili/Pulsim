# benchmark-suite

## ADDED Requirements

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
