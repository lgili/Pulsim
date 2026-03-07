## ADDED Requirements
### Requirement: AC Sweep Benchmark Corpus Coverage
The benchmark suite SHALL include AC sweep scenarios covering at least one analytical linear circuit and one converter/control-use-case workflow.

#### Scenario: Analytical reference AC sweep case
- **WHEN** AC benchmark suite runs the analytical reference case
- **THEN** magnitude and phase responses are compared against analytical/reference values
- **AND** errors are reported in machine-readable KPI artifacts

#### Scenario: Converter/control AC sweep case
- **WHEN** AC benchmark suite runs a converter/control frequency-response case
- **THEN** benchmark verifies deterministic response generation and structured metric emission
- **AND** unsupported configurations fail with typed diagnostics

### Requirement: AC Sweep KPI Regression Gates
Benchmark gating SHALL enforce non-regression thresholds for AC sweep accuracy and runtime metrics.

#### Scenario: AC sweep regression threshold violation
- **WHEN** required AC sweep KPI thresholds are exceeded versus approved baseline
- **THEN** CI fails the gate deterministically
- **AND** artifacts include baseline/current values and delta metrics

#### Scenario: AC sweep gate pass
- **WHEN** required AC sweep KPIs remain within thresholds
- **THEN** CI reports gate pass
- **AND** machine-readable KPI report is published

### Requirement: AC Sweep Determinism Regression Gate
Benchmark validation SHALL include repeat-run determinism checks for AC sweep outputs on the same machine class.

#### Scenario: Repeat-run deterministic comparison
- **WHEN** the same AC sweep benchmark case executes multiple times with identical configuration
- **THEN** frequency grids are identical and response drift stays within configured tolerance
- **AND** determinism regressions fail the gate
