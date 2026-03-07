## ADDED Requirements
### Requirement: Waveform Post-Processing and Measurement Tools
The system SHALL provide built-in post-processing for FFT/THD/RMS/efficiency/loop metrics integrated into simulation result workflows as a first-class, deterministic feature suitable for production power-electronics workflows.

#### Scenario: Post-processing computes deterministic spectral and scalar metrics
- **GIVEN** a valid model configured for built-in post-processing for FFT/THD/RMS/efficiency/loop metrics integrated into simulation result workflows
- **WHEN** the feature is executed
- **THEN** the backend completes with deterministic outputs and typed telemetry
- **AND** results are exportable through existing result channels/APIs.

#### Scenario: Invalid windowing or sampling prerequisites are diagnosed
- **GIVEN** an invalid or unsupported configuration for built-in post-processing for FFT/THD/RMS/efficiency/loop metrics integrated into simulation result workflows
- **WHEN** the run is validated or executed
- **THEN** execution fails fast with structured, deterministic diagnostics
- **AND** partial/ambiguous outputs are not emitted.
