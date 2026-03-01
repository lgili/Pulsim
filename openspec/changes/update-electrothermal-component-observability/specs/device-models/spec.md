## ADDED Requirements
### Requirement: Thermal-Port Capability Declaration and Enforcement
Device models SHALL declare thermal-port capability, and this capability SHALL be enforced by parser/runtime validation.

#### Scenario: Thermal-capable model with thermal port enabled
- **GIVEN** a component type that declares thermal capability
- **WHEN** thermal port configuration is enabled for an instance
- **THEN** the runtime accepts thermal configuration and participates in electrothermal updates

#### Scenario: Non-thermal-capable model with thermal port enabled
- **GIVEN** a component type that declares no thermal capability
- **WHEN** thermal port configuration is enabled for an instance
- **THEN** the configuration is rejected with deterministic diagnostics

### Requirement: Consistent Thermal Parameter Semantics
Thermal-capable device models SHALL apply `rth`, `cth`, `temp_init`, `temp_ref`, and `alpha` with consistent electrothermal semantics.

#### Scenario: Loss-only thermal policy
- **WHEN** global policy is `LossOnly`
- **THEN** losses feed temperature evolution
- **AND** temperature-dependent electrical scaling is not applied

#### Scenario: Loss-with-temperature-scaling policy
- **WHEN** global policy is `LossWithTemperatureScaling`
- **THEN** losses feed temperature evolution
- **AND** bounded temperature-dependent scaling is applied using component thermal parameters
