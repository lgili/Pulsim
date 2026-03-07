## ADDED Requirements
### Requirement: Embedded Control Code Generation Pipeline
The system SHALL provide automatic generation of deterministic C code from control diagrams for embedded targets and rapid control prototyping as a first-class, deterministic feature suitable for production power-electronics workflows.

#### Scenario: Control model exports target-ready C artifacts
- **GIVEN** a valid model configured for automatic generation of deterministic C code from control diagrams for embedded targets and rapid control prototyping
- **WHEN** the feature is executed
- **THEN** the backend completes with deterministic outputs and typed telemetry
- **AND** results are exportable through existing result channels/APIs.

#### Scenario: Unsupported control graph patterns fail with actionable diagnostics
- **GIVEN** an invalid or unsupported configuration for automatic generation of deterministic C code from control diagrams for embedded targets and rapid control prototyping
- **WHEN** the run is validated or executed
- **THEN** execution fails fast with structured, deterministic diagnostics
- **AND** partial/ambiguous outputs are not emitted.
