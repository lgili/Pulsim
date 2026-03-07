## ADDED Requirements
### Requirement: Controller Auto-Design Suite
The system SHALL provide controller auto-design/tuning assistants for common converter topologies (PI/PID current and voltage loops) as a first-class, deterministic feature suitable for production power-electronics workflows.

#### Scenario: Auto-design tool synthesizes controller gains from plant targets
- **GIVEN** a valid model configured for controller auto-design/tuning assistants for common converter topologies (PI/PID current and voltage loops)
- **WHEN** the feature is executed
- **THEN** the backend completes with deterministic outputs and typed telemetry
- **AND** results are exportable through existing result channels/APIs.

#### Scenario: Unreachable design constraints are reported with structured diagnostics
- **GIVEN** an invalid or unsupported configuration for controller auto-design/tuning assistants for common converter topologies (PI/PID current and voltage loops)
- **WHEN** the run is validated or executed
- **THEN** execution fails fast with structured, deterministic diagnostics
- **AND** partial/ambiguous outputs are not emitted.
