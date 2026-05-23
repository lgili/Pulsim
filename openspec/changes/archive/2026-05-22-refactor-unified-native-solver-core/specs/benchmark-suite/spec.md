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

### Requirement: Hybrid Path and Electrothermal KPI Gates
Benchmark and stress tooling SHALL track and gate hybrid-path usage plus electrothermal regression metrics for converter-focused phases.

#### Scenario: Hybrid-path KPI emission
- **WHEN** converter-focused benchmark suites complete
- **THEN** reports include at least `state_space_primary_ratio` and `dae_fallback_ratio`
- **AND** required-threshold regressions in these KPIs fail the phase gate

#### Scenario: Electrothermal KPI emission and gating
- **WHEN** electrothermal-enabled benchmark suites complete
- **THEN** reports include at least `loss_energy_balance_error` and `thermal_peak_temperature_delta`
- **AND** required-threshold regressions fail the phase gate
