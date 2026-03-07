## ADDED Requirements
### Requirement: Post-Processing Benchmark Fixture Coverage
The benchmark suite SHALL include deterministic fixture scenarios that validate waveform post-processing correctness across metric families.

#### Scenario: Spectral fixture with known harmonics
- **WHEN** FFT/THD fixtures with known harmonic content are executed
- **THEN** benchmark artifacts include spectral/THD errors against analytical or frozen-reference expectations
- **AND** errors remain within configured thresholds.

#### Scenario: Power-efficiency fixture
- **WHEN** efficiency/power-factor fixtures are executed
- **THEN** benchmark artifacts include derived metric errors against declared references
- **AND** deterministic pass/fail status is emitted.

### Requirement: Deterministic Expected-Failure Fixtures
Benchmark coverage SHALL include expected-failure fixtures for invalid post-processing contracts.

#### Scenario: Invalid window/sampling expected failure
- **WHEN** a fixture intentionally violates post-processing window/sampling prerequisites
- **THEN** benchmark run validates typed expected-failure diagnostics
- **AND** the case is marked passed only if expected diagnostics match deterministically.

### Requirement: Post-Processing KPI Non-Regression Gates
Benchmark KPI gating SHALL enforce post-processing metric correctness, determinism, and runtime-overhead constraints.

#### Scenario: Correctness gate
- **WHEN** post-processing KPI reports are evaluated
- **THEN** required metrics (for example FFT bin error, THD error, scalar metric error) stay within approved thresholds
- **AND** threshold violations fail the required gate.

#### Scenario: Determinism and overhead gate
- **WHEN** repeat-run and runtime-overhead KPIs are evaluated
- **THEN** determinism drift and overhead regressions stay within approved thresholds
- **AND** regressions beyond thresholds fail CI.

### Requirement: Machine-Readable Post-Processing KPI Reporting
Benchmark artifacts SHALL emit machine-readable post-processing KPI deltas and gate status.

#### Scenario: KPI report publication
- **WHEN** post-processing benchmark gate completes
- **THEN** artifacts include baseline/current values, deltas, and pass/fail status per KPI
- **AND** schema remains stable for automated tooling.
