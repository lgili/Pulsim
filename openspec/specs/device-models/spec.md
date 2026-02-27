# device-models Specification

## Purpose
TBD - created by archiving change improve-convergence-algorithms. Update Purpose after archive.
## Requirements
### Requirement: Diode Stamp with Limiting

The diode stamp function SHALL apply voltage limiting before computing current and conductance.

The stamp SHALL:
- Retrieve previous diode voltage from device state
- Apply voltage limiting to new voltage
- Compute I and G using limited voltage
- Store new voltage in device state

#### Scenario: Diode stamp with limiting

- **GIVEN** MNA assembly with voltage limiting enabled
- **WHEN** stamp_diode() is called with V_new from Newton
- **THEN** V_limited = limit_diode_voltage(V_new, V_old)
- **AND** I and G are computed using V_limited
- **AND** V_old is updated to V_limited

### Requirement: MOSFET Stamp with Limiting

The MOSFET stamp function SHALL apply voltage limiting before computing currents.

#### Scenario: MOSFET stamp with Vgs and Vds limiting

- **GIVEN** MNA assembly with voltage limiting enabled
- **WHEN** stamp_mosfet() is called
- **THEN** Vgs_limited = limit_mosfet_vgs(Vgs_new, Vgs_old)
- **AND** Vds_limited = limit_mosfet_vds(Vds_new, Vds_old)
- **AND** drain current is computed using limited voltages

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

