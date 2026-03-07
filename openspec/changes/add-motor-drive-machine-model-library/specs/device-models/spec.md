## ADDED Requirements
### Requirement: Motor Drive Machine Model Library
The system SHALL provide native machine models (PMSM/BLDC/induction) and inverter-drive coupling suitable for FOC and sensorless workflows as a first-class, deterministic feature suitable for production power-electronics workflows.

#### Scenario: Machine model executes with inverter and control loop coupling
- **GIVEN** a valid model configured for native machine models (PMSM/BLDC/induction) and inverter-drive coupling suitable for FOC and sensorless workflows
- **WHEN** the feature is executed
- **THEN** the backend completes with deterministic outputs and typed telemetry
- **AND** results are exportable through existing result channels/APIs.

#### Scenario: Invalid machine parameterization is rejected with typed diagnostics
- **GIVEN** an invalid or unsupported configuration for native machine models (PMSM/BLDC/induction) and inverter-drive coupling suitable for FOC and sensorless workflows
- **WHEN** the run is validated or executed
- **THEN** execution fails fast with structured, deterministic diagnostics
- **AND** partial/ambiguous outputs are not emitted.
