## ADDED Requirements
### Requirement: Deterministic Per-Component Electrothermal Results
The v1 kernel SHALL publish deterministic electrothermal telemetry per non-virtual circuit component in transient simulation results.

#### Scenario: Component coverage and loss fields
- **WHEN** a transient run completes with loss tracking enabled
- **THEN** the result includes one entry per non-virtual component in deterministic order
- **AND** each entry includes deterministic component identity and loss fields (`conduction`, `turn_on`, `turn_off`, `reverse_recovery`, `total_loss`, `total_energy`, `average_power`, `peak_power`)
- **AND** components with zero dissipation are reported with zero-valued loss fields

#### Scenario: Thermal fields for thermal-port-enabled component
- **GIVEN** thermal coupling is enabled and a component thermal port is enabled
- **WHEN** transient simulation completes
- **THEN** the component entry includes `final_temperature`, `peak_temperature`, and `average_temperature`
- **AND** these values are derived from the same accepted-segment/event commit model used for loss accumulation

#### Scenario: Non-thermal-capable component in unified report
- **GIVEN** a component without thermal capability is part of the circuit
- **WHEN** transient simulation completes
- **THEN** the component still appears in the per-component electrothermal report
- **AND** the entry marks thermal as disabled with ambient-derived temperature values

### Requirement: Thermal Parameter Runtime Guardrails
The v1 kernel SHALL validate thermal constants for thermal-enabled components before transient stepping begins.

#### Scenario: Invalid thermal constants
- **WHEN** a thermal-enabled component has non-finite values, `rth <= 0`, or `cth < 0`
- **THEN** simulation fails with deterministic typed diagnostics
- **AND** no partial transient electrothermal report is emitted
