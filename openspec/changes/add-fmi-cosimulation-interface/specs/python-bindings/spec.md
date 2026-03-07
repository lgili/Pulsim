## ADDED Requirements
### Requirement: FMI Co-Simulation Interface
The system SHALL provide FMI-compatible co-simulation interfaces for external tool coupling and multi-domain workflows as a first-class, deterministic feature suitable for production power-electronics workflows.

#### Scenario: Pulsim exchanges states/signals with FMI participants in synchronized time
- **GIVEN** a valid model configured for FMI-compatible co-simulation interfaces for external tool coupling and multi-domain workflows
- **WHEN** the feature is executed
- **THEN** the backend completes with deterministic outputs and typed telemetry
- **AND** results are exportable through existing result channels/APIs.

#### Scenario: Incompatible FMI contracts are rejected before runtime coupling
- **GIVEN** an invalid or unsupported configuration for FMI-compatible co-simulation interfaces for external tool coupling and multi-domain workflows
- **WHEN** the run is validated or executed
- **THEN** execution fails fast with structured, deterministic diagnostics
- **AND** partial/ambiguous outputs are not emitted.
