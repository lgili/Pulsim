## ADDED Requirements
### Requirement: Paired Switching-vs-Averaged Benchmark Coverage
The benchmark suite SHALL include paired scenarios that compare averaged-mode runs against switching-reference runs for supported converter topologies.

#### Scenario: Paired buck benchmark case
- **GIVEN** a supported buck topology with equivalent electrical/control parameters
- **WHEN** switching-reference and averaged-mode scenarios are executed
- **THEN** benchmark artifacts include both runs with comparable observables
- **AND** fidelity metrics are emitted deterministically.

#### Scenario: Expected-failure benchmark for unsupported averaged config
- **GIVEN** an averaged configuration outside supported topology/envelope contracts
- **WHEN** benchmark execution runs
- **THEN** typed expected-failure diagnostics are validated deterministically
- **AND** no regex-based log parsing is required.

### Requirement: Averaged-Mode KPI Gates
Benchmark KPI gating SHALL enforce fidelity and runtime value thresholds for averaged modeling.

#### Scenario: Fidelity gate
- **WHEN** averaged-vs-switching benchmark comparisons are evaluated
- **THEN** configured output/state error metrics remain within approved thresholds
- **AND** gate reports explicit failing metric names when thresholds are exceeded.

#### Scenario: Runtime value gate
- **WHEN** averaged-mode runtime is compared with paired switching-reference runtime
- **THEN** averaged-mode meets minimum speedup thresholds defined in KPI policy
- **AND** regressions beyond tolerance fail the required KPI gate.

### Requirement: Averaged-Mode Determinism Regression Checks
Benchmark coverage SHALL include deterministic repeat-run checks for averaged-mode scenarios.

#### Scenario: Repeat-run deterministic output
- **GIVEN** identical benchmark inputs on the same machine class
- **WHEN** averaged-mode scenario is executed repeatedly
- **THEN** key observables and KPI metrics remain within determinism tolerance bands
- **AND** nondeterministic deviations fail determinism checks.
