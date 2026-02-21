## ADDED Requirements

### Requirement: Converter Power Device Coverage
Device models SHALL cover declared converter power switch classes with stable nonlinear stamping behavior.

#### Scenario: Declared power switch models in one converter case
- **WHEN** a converter uses supported diode, switch, MOSFET, and IGBT models
- **THEN** the simulation converges using v1 model implementations
- **AND** model-specific diagnostics are available for failed convergence

### Requirement: Electro-Thermal Loss Coupling
Device models SHALL support coupling between electrical loss calculations and thermal state updates for declared workflows.

#### Scenario: Thermal feedback enabled
- **WHEN** thermal feedback is enabled for supported devices
- **THEN** per-device losses update thermal states
- **AND** temperature-dependent device parameters are updated according to configured rules

### Requirement: External SPICE Calibration Envelope
Declared converter device-model workflows SHALL define calibration and validation envelopes against external SPICE references.

#### Scenario: LTspice parity check for device waveform
- **WHEN** a supported converter benchmark is compared to LTspice
- **THEN** waveform error metrics are checked against configured thresholds
- **AND** failures report which device-model observables exceeded limits
