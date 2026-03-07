## ADDED Requirements
### Requirement: Component-Level Electrothermal Validation Matrix
The benchmark and validation suite SHALL assert component-level electrothermal outputs, not only aggregate KPIs.

#### Scenario: Electrothermal reference circuit emits component telemetry
- **WHEN** an electrothermal benchmark circuit completes
- **THEN** reports include per-component losses and temperatures for declared components
- **AND** component metrics are compared against baseline tolerances

#### Scenario: Aggregate-to-component consistency checks
- **WHEN** benchmark post-processing computes aggregate electrothermal KPIs
- **THEN** aggregate totals are consistent with reductions over per-component telemetry
- **AND** inconsistencies fail the gate with deterministic diagnostics

#### Scenario: Thermal-port parser contract regression check
- **WHEN** strict-mode benchmark/parser validation runs include invalid thermal-port configurations
- **THEN** deterministic parser diagnostics are emitted and the run fails as expected
