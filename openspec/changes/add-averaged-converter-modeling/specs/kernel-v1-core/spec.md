## ADDED Requirements
### Requirement: Averaged Converter Modeling Mode
The system SHALL provide averaged converter modeling mode with consistent mapping to switching models for fast control-loop design iterations as a first-class, deterministic feature suitable for production power-electronics workflows.

#### Scenario: Averaged model run produces stable low-ripple dynamic response
- **GIVEN** a valid model configured for averaged converter modeling mode with consistent mapping to switching models for fast control-loop design iterations
- **WHEN** the feature is executed
- **THEN** the backend completes with deterministic outputs and typed telemetry
- **AND** results are exportable through existing result channels/APIs.

#### Scenario: Unsupported topology/model-mode combinations are rejected
- **GIVEN** an invalid or unsupported configuration for averaged converter modeling mode with consistent mapping to switching models for fast control-loop design iterations
- **WHEN** the run is validated or executed
- **THEN** execution fails fast with structured, deterministic diagnostics
- **AND** partial/ambiguous outputs are not emitted.
