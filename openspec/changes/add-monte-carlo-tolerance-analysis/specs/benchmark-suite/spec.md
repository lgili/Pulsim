## ADDED Requirements
### Requirement: Monte Carlo Tolerance Analysis
The system SHALL provide Monte Carlo analysis over component tolerances and model parameters with reproducible seeded runs as a first-class, deterministic feature suitable for production power-electronics workflows.

#### Scenario: Seeded Monte Carlo run generates stable statistics
- **GIVEN** a valid model configured for Monte Carlo analysis over component tolerances and model parameters with reproducible seeded runs
- **WHEN** the feature is executed
- **THEN** the backend completes with deterministic outputs and typed telemetry
- **AND** results are exportable through existing result channels/APIs.

#### Scenario: Invalid tolerance distribution or sample count is rejected
- **GIVEN** an invalid or unsupported configuration for Monte Carlo analysis over component tolerances and model parameters with reproducible seeded runs
- **WHEN** the run is validated or executed
- **THEN** execution fails fast with structured, deterministic diagnostics
- **AND** partial/ambiguous outputs are not emitted.
