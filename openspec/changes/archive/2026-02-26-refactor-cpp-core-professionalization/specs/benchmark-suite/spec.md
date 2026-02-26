## ADDED Requirements
### Requirement: Frozen Baseline Artifact Governance
The benchmark suite SHALL maintain versioned frozen baselines for KPI comparison, including environment fingerprint metadata.

#### Scenario: Freeze baseline snapshot
- **WHEN** a baseline freeze is created for a benchmark corpus version
- **THEN** artifacts store KPI values and environment fingerprint fields (compiler, flags, machine class)
- **AND** the baseline snapshot is immutable for regression-gate comparisons

#### Scenario: Baseline provenance check
- **WHEN** KPI gate evaluation runs in CI
- **THEN** the evaluator verifies baseline provenance metadata before comparing thresholds
- **AND** fails deterministically if provenance metadata is missing or inconsistent

### Requirement: KPI Non-Regression Gates
Benchmark and validation pipelines SHALL block merge when required KPI thresholds regress beyond configured limits.

#### Scenario: Regression threshold violation
- **WHEN** any required KPI exceeds allowed regression threshold against frozen baseline
- **THEN** CI marks the gate as failed
- **AND** merge is blocked until KPI compliance is restored or thresholds are explicitly revised

#### Scenario: KPI gate pass
- **WHEN** all required KPIs are within approved thresholds for the selected matrix
- **THEN** CI marks the non-regression gate as passed
- **AND** gate status is published in machine-readable artifacts

### Requirement: Canonical KPI Matrix Coverage
The benchmark suite SHALL enforce KPI coverage for converter, linear, stress, and electrothermal scenario classes across canonical runtime modes.

#### Scenario: Matrix execution with class coverage
- **WHEN** the benchmark matrix is executed for release gating
- **THEN** it includes scenario classes `converter`, `linear`, `stress`, and `electrothermal`
- **AND** reports KPI outputs per class and per runtime mode where applicable

### Requirement: Machine-Readable KPI Delta Reports
Each gated run SHALL emit machine-readable KPI delta reports suitable for automated trend and release decisions.

#### Scenario: Publish KPI delta report
- **WHEN** a gated benchmark run completes
- **THEN** artifacts include baseline, current values, absolute/relative deltas, and pass/fail status per KPI
- **AND** report schema remains stable across minor tool updates
