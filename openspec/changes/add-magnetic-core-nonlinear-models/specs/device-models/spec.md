## ADDED Requirements
### Requirement: Nonlinear Magnetic Core and Loss Models
The system SHALL provide nonlinear magnetic core models (saturation, hysteresis, frequency-dependent core loss) for inductors/transformers as a first-class, deterministic feature suitable for production power-electronics workflows.

#### Scenario: Magnetic component simulation includes nonlinear flux and loss behavior
- **GIVEN** a valid model configured for nonlinear magnetic core models (saturation, hysteresis, frequency-dependent core loss) for inductors/transformers
- **WHEN** the feature is executed
- **THEN** the backend completes with deterministic outputs and typed telemetry
- **AND** results are exportable through existing result channels/APIs.

#### Scenario: Invalid core curve/table definitions are rejected deterministically
- **GIVEN** an invalid or unsupported configuration for nonlinear magnetic core models (saturation, hysteresis, frequency-dependent core loss) for inductors/transformers
- **WHEN** the run is validated or executed
- **THEN** execution fails fast with structured, deterministic diagnostics
- **AND** partial/ambiguous outputs are not emitted.
