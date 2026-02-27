## ADDED Requirements

### Requirement: Modular Component Model Library
The system SHALL define each built-in electrical component model in a dedicated component file under a stable component library path, while preserving a compatibility aggregator include for legacy callers.

#### Scenario: Legacy include compatibility after modularization
- **GIVEN** existing code that includes `pulsim/v1/device_base.hpp`
- **WHEN** the project is built after model modularization
- **THEN** all existing built-in component types remain available
- **AND** no caller migration is required for include-path compatibility

#### Scenario: Isolated model evolution per component
- **GIVEN** a change to one component model file
- **WHEN** tests and benchmarks are executed
- **THEN** only that component module and dependent integration paths are impacted
- **AND** unrelated models do not require structural edits in the same file

### Requirement: Controlled Numerical Regularization for Switching Models
The system SHALL support controlled, bounded numerical regularization for switching/nonlinear component models to improve convergence in pathological switching regimes without unbounded physical distortion.

#### Scenario: Automatic regularization in repeated switching-step failure
- **GIVEN** repeated transient failures near switching discontinuities
- **WHEN** recovery policy escalates through configured stages
- **THEN** bounded regularization parameters are applied to eligible component models
- **AND** each escalation is recorded in structured telemetry

#### Scenario: Regularization bounded by policy limits
- **GIVEN** auto-regularization is active
- **WHEN** the solver escalates regularization intensity
- **THEN** configured maximum bounds are never exceeded
- **AND** simulation aborts with typed diagnostics if convergence still fails
