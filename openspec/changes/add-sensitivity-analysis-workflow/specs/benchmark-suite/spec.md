## ADDED Requirements
### Requirement: Sensitivity Analysis Workflow
The system SHALL provide sensitivity analysis that quantifies output variation against parameter perturbations as a first-class, deterministic feature suitable for production power-electronics workflows.

#### Scenario: Sensitivity run reports ranked parameter influence
- **GIVEN** a valid model configured for sensitivity analysis that quantifies output variation against parameter perturbations
- **WHEN** the feature is executed
- **THEN** the backend completes with deterministic outputs and typed telemetry
- **AND** results are exportable through existing result channels/APIs.

#### Scenario: Unknown parameter targets or malformed sweep ranges are rejected
- **GIVEN** an invalid or unsupported configuration for sensitivity analysis that quantifies output variation against parameter perturbations
- **WHEN** the run is validated or executed
- **THEN** execution fails fast with structured, deterministic diagnostics
- **AND** partial/ambiguous outputs are not emitted.
