## ADDED Requirements
### Requirement: Nonlinear Magnetic-Core Benchmark Fixture Catalog
The benchmark suite SHALL include deterministic nonlinear magnetic-core fixture scenarios for saturation, hysteresis, and frequency-dependent core-loss behavior.

#### Scenario: Saturation fixture validation
- **GIVEN** a benchmark fixture with expected saturation behavior
- **WHEN** benchmark execution completes
- **THEN** reported saturation-related observables are compared against declared references/tolerances
- **AND** regressions fail with deterministic diagnostics.

#### Scenario: Hysteresis fixture validation
- **GIVEN** a benchmark fixture with expected hysteresis-loop behavior
- **WHEN** benchmark execution completes
- **THEN** loop-state and cycle-energy metrics are validated against reference envelopes
- **AND** deterministic pass/fail status is emitted.

#### Scenario: Frequency-dependent core-loss fixture validation
- **GIVEN** a fixture sweeping excitation frequency across configured points
- **WHEN** benchmark execution completes
- **THEN** core-loss trend metrics are validated against declared expectations
- **AND** failures include machine-readable error context.

### Requirement: Magnetic-Core KPI Non-Regression Gates
Benchmark gates SHALL enforce magnetic-core fidelity, determinism, and performance thresholds.

#### Scenario: Magnetic KPI gate pass
- **WHEN** benchmark run metrics are within approved thresholds
- **THEN** magnetic-core KPI gate passes
- **AND** machine-readable artifacts include baseline/current/delta fields.

#### Scenario: Magnetic KPI gate fail
- **WHEN** required magnetic-core KPI exceeds configured regression thresholds
- **THEN** CI gate fails deterministically
- **AND** failure report identifies the violated KPI and threshold.

### Requirement: Magnetic Repeat-Run Determinism Checks
The benchmark suite SHALL include repeat-run determinism checks for magnetic-core channels and summaries.

#### Scenario: Repeat-run determinism
- **GIVEN** identical hardware class and identical benchmark configuration
- **WHEN** the same magnetic-core fixture runs multiple times
- **THEN** deterministic fields remain within configured tolerances
- **AND** drift beyond tolerance fails determinism gate.
