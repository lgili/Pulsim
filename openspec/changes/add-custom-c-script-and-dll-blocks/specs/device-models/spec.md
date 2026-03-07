## ADDED Requirements
### Requirement: Custom C-Script and DLL Functional Blocks
The system SHALL provide user-defined functional blocks implemented in C code or dynamically loaded libraries with deterministic I/O contracts as a first-class, deterministic feature suitable for production power-electronics workflows.

#### Scenario: Custom block executes with deterministic scheduling and bounded side effects
- **GIVEN** a valid model configured for user-defined functional blocks implemented in C code or dynamically loaded libraries with deterministic I/O contracts
- **WHEN** the feature is executed
- **THEN** the backend completes with deterministic outputs and typed telemetry
- **AND** results are exportable through existing result channels/APIs.

#### Scenario: Invalid ABI/signature or unsafe block configuration is rejected
- **GIVEN** an invalid or unsupported configuration for user-defined functional blocks implemented in C code or dynamically loaded libraries with deterministic I/O contracts
- **WHEN** the run is validated or executed
- **THEN** execution fails fast with structured, deterministic diagnostics
- **AND** partial/ambiguous outputs are not emitted.
