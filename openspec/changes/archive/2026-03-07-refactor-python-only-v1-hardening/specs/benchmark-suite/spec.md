## MODIFIED Requirements

### Requirement: Runtime Parity for ngspice Comparator
The benchmark comparator SHALL run Pulsim through the same Python runtime path used by the core benchmark runner and SHALL support external SPICE parity backends with LTspice as a primary target.

#### Scenario: Pulsim vs LTspice comparison
- **WHEN** the comparator executes a mapped benchmark pair with LTspice backend configured
- **THEN** Pulsim waveforms are generated through the Python runtime path used by the benchmark suite
- **AND** comparison metrics are computed from mapped Pulsim and LTspice observables

#### Scenario: Pulsim vs ngspice comparison
- **WHEN** the comparator executes a mapped benchmark pair with ngspice backend configured
- **THEN** Pulsim runtime execution path remains identical to the core benchmark runner

### Requirement: Validation types
The system SHALL support the validation types `analytical`, `reference`, `ltspice`, and `ngspice` for benchmark comparisons.

#### Scenario: Validate against LTspice reference
- **WHEN** a benchmark uses validation type `ltspice`
- **THEN** outputs are compared against LTspice-generated reference waveforms using configured observables and tolerances

## ADDED Requirements

### Requirement: External Simulator Path Configuration
Benchmark tooling SHALL accept explicit external simulator executable paths and fail with actionable diagnostics when misconfigured.

#### Scenario: LTspice path missing
- **WHEN** an LTspice parity run is requested without a valid executable path
- **THEN** the run fails with a clear configuration error
- **AND** the failure reason is recorded in benchmark artifacts

### Requirement: Converter Stress Catalog
The benchmark suite SHALL include a declared converter stress catalog covering light, medium, and heavy simulation tiers.

#### Scenario: Run stress catalog
- **WHEN** the stress benchmark command is executed
- **THEN** each catalog case reports convergence status, fallback path, and runtime metrics
- **AND** no case is silently skipped without explicit reason

### Requirement: Deterministic Benchmark Fields
Benchmark artifacts SHALL include deterministic fields suitable for reproducibility checks.

#### Scenario: Reproducibility check
- **WHEN** the same benchmark matrix is run twice on the same hardware class with fixed settings
- **THEN** deterministic fields in artifacts match within configured tolerances
