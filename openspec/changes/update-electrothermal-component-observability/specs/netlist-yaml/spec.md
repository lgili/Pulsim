## ADDED Requirements
### Requirement: Thermal-Port YAML Contract
The YAML schema SHALL define `component.thermal` as the thermal-port configuration block with deterministic validation and diagnostics.

#### Scenario: Valid thermal-port configuration
- **WHEN** a component defines `thermal.enabled=true` with valid thermal constants
- **THEN** the parser maps `rth`, `cth`, `temp_init`, `temp_ref`, and `alpha` into runtime thermal-device configuration
- **AND** the thermal port is activated for that component

#### Scenario: Unsupported thermal port activation
- **WHEN** `component.thermal.enabled=true` is declared for a component type without thermal capability
- **THEN** parsing fails with deterministic unsupported-thermal-port diagnostics

### Requirement: Strict and Non-Strict Missing-Parameter Policy
The parser SHALL support explicit strict-mode behavior for missing required thermal constants when a thermal port is enabled.

#### Scenario: Missing thermal constants in strict mode
- **GIVEN** strict parsing is enabled
- **WHEN** `component.thermal.enabled=true` and `rth` or `cth` is missing
- **THEN** parsing fails with field-path diagnostics identifying missing keys

#### Scenario: Missing thermal constants in non-strict mode
- **GIVEN** strict parsing is disabled
- **WHEN** `component.thermal.enabled=true` and `rth` or `cth` is missing
- **THEN** missing values are defaulted from `simulation.thermal.default_rth/default_cth`
- **AND** deterministic warning diagnostics are emitted

### Requirement: Thermal Constant Range Validation
Thermal-port constants SHALL be validated for finite numeric ranges.

#### Scenario: Invalid thermal numeric ranges
- **WHEN** YAML thermal fields contain non-finite values, `rth <= 0`, or `cth < 0`
- **THEN** parsing fails with deterministic range diagnostics
- **AND** diagnostics include the exact YAML field path
