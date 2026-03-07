## ADDED Requirements
### Requirement: Fault Injection and DFMEA Analysis Workflow
The system SHALL provide fault-injection scenarios (short/open/sensor/control faults) with deterministic event timing and outcome classification as a first-class, deterministic feature suitable for production power-electronics workflows.

#### Scenario: Fault campaign executes and classifies converter outcomes
- **GIVEN** a valid model configured for fault-injection scenarios (short/open/sensor/control faults) with deterministic event timing and outcome classification
- **WHEN** the feature is executed
- **THEN** the backend completes with deterministic outputs and typed telemetry
- **AND** results are exportable through existing result channels/APIs.

#### Scenario: Unsupported fault primitives are rejected with typed diagnostics
- **GIVEN** an invalid or unsupported configuration for fault-injection scenarios (short/open/sensor/control faults) with deterministic event timing and outcome classification
- **WHEN** the run is validated or executed
- **THEN** execution fails fast with structured, deterministic diagnostics
- **AND** partial/ambiguous outputs are not emitted.
