## ADDED Requirements
### Requirement: Frequency-Domain AC Sweep and Small-Signal Analysis
The system SHALL provide frequency-domain analysis (open-loop/closed-loop transfer functions, impedance sweeps) without requiring manual averaged-model derivation as a first-class, deterministic feature suitable for production power-electronics workflows.

#### Scenario: Open-loop and closed-loop frequency sweep execution
- **GIVEN** a valid model configured for frequency-domain analysis (open-loop/closed-loop transfer functions, impedance sweeps) without requiring manual averaged-model derivation
- **WHEN** the feature is executed
- **THEN** the backend completes with deterministic outputs and typed telemetry
- **AND** results are exportable through existing result channels/APIs.

#### Scenario: Invalid perturbation or sweep configuration is rejected
- **GIVEN** an invalid or unsupported configuration for frequency-domain analysis (open-loop/closed-loop transfer functions, impedance sweeps) without requiring manual averaged-model derivation
- **WHEN** the run is validated or executed
- **THEN** execution fails fast with structured, deterministic diagnostics
- **AND** partial/ambiguous outputs are not emitted.
