## ADDED Requirements
### Requirement: Real-Time and HIL Execution Mode
The system SHALL provide deterministic fixed-step real-time execution mode for hardware-in-the-loop and controller prototyping as a first-class, deterministic feature suitable for production power-electronics workflows.

#### Scenario: Real-time mode runs with deterministic step deadlines and telemetry
- **GIVEN** a valid model configured for deterministic fixed-step real-time execution mode for hardware-in-the-loop and controller prototyping
- **WHEN** the feature is executed
- **THEN** the backend completes with deterministic outputs and typed telemetry
- **AND** results are exportable through existing result channels/APIs.

#### Scenario: Deadline overruns and unsupported model elements are diagnosed deterministically
- **GIVEN** an invalid or unsupported configuration for deterministic fixed-step real-time execution mode for hardware-in-the-loop and controller prototyping
- **WHEN** the run is validated or executed
- **THEN** execution fails fast with structured, deterministic diagnostics
- **AND** partial/ambiguous outputs are not emitted.
