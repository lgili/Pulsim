## ADDED Requirements
### Requirement: Phase-Gated Solver Refactor Validation
Benchmark and validation tooling SHALL enforce phase-gated KPI checks for solver-core refactor milestones.

#### Scenario: Phase gate blocks regression
- **WHEN** a phase run exceeds configured regression thresholds for required KPIs
- **THEN** the phase is marked failed
- **AND** progression to subsequent implementation phase is blocked in CI

### Requirement: Canonical KPI Reporting
Benchmark artifacts SHALL include canonical KPIs for convergence, accuracy, event fidelity, and runtime efficiency.

#### Scenario: KPI artifact generation
- **WHEN** benchmark/parity/stress suites complete
- **THEN** artifacts include at least convergence success rate, parity RMS error, event-time error, runtime p50, and runtime p95
- **AND** values are emitted in machine-readable JSON summaries

### Requirement: Dual-Mode Coverage Matrix
The benchmark matrix SHALL include both canonical timestep modes (`fixed` and `variable`) across converter-focused scenarios.

#### Scenario: Converter matrix run
- **WHEN** matrix execution is triggered for converter suites
- **THEN** each selected converter case runs in both fixed and variable modes
- **AND** mode-specific KPI results are reported separately

### Requirement: Baseline Freeze and Comparison
Benchmark tooling SHALL support baseline freeze snapshots and automated comparison against the frozen baseline for each phase.

#### Scenario: Baseline comparison report
- **WHEN** a phase benchmark run completes
- **THEN** the report compares current KPIs against the frozen baseline snapshot
- **AND** flags pass/fail status per configured regression threshold
