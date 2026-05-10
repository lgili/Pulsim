## ADDED Requirements

### Requirement: Three-Phase Source Library
The library SHALL provide balanced, unbalanced, programmable, and harmonic-injected three-phase sources.

#### Scenario: Balanced source generates symmetric voltages
- **GIVEN** `ThreePhaseSource { v_rms: 230, frequency: 50, phase_seq: abc }`
- **WHEN** the simulation runs at 50 Hz steady state
- **THEN** the three line-to-neutral voltages are 230 V RMS, 120° apart, with abc sequence

#### Scenario: Programmable source with sag event
- **GIVEN** `ThreePhaseSourceProgrammable` with a 50% sag event on phase A at t=0.1 s for 100 ms
- **WHEN** the simulation runs over 0.3 s
- **THEN** the phase-A waveform shows the sag for the configured duration
- **AND** phases B and C remain unaffected

#### Scenario: Harmonic source injection
- **GIVEN** `ThreePhaseHarmonicSource` with fundamental + 5th and 7th harmonics
- **WHEN** the FFT of the line voltage is computed
- **THEN** the harmonic magnitudes match the configured values within 1%

### Requirement: Phase-Locked Loop Variants
The library SHALL provide `SrfPll`, `DsogiPll`, and `MafPll` blocks for grid synchronization.

#### Scenario: SrfPll locks on nominal grid
- **GIVEN** an `SrfPll` connected to a balanced 230 V / 50 Hz source
- **WHEN** the simulation runs from rest
- **THEN** the PLL achieves frequency lock within 50 ms
- **AND** steady-state phase error is below 0.5°

#### Scenario: DsogiPll on unbalanced grid
- **GIVEN** a `DsogiPll` connected to a 50%-sag-on-Phase-A source
- **WHEN** the simulation runs to steady state
- **THEN** the PLL frequency estimate remains within ±1% of fundamental
- **AND** no sustained oscillation appears in `theta_estimate`

#### Scenario: PLL re-lock after grid loss
- **GIVEN** an `SrfPll` operating at lock and a 100 ms grid interruption
- **WHEN** the grid is restored
- **THEN** the PLL re-locks within 100 ms with no glitch in downstream control

### Requirement: Symmetrical Components Decomposition
The library SHALL provide a `SymmetricalComponents` block that decomposes three-phase signals into positive, negative, and zero sequences using the Fortescue transform with appropriate delay buffer.

#### Scenario: Pure positive sequence input
- **GIVEN** a balanced abc-sequence input
- **WHEN** decomposition runs after settling
- **THEN** positive-sequence magnitude equals input magnitude
- **AND** negative and zero sequences are below 1% of positive

#### Scenario: Unbalanced input
- **GIVEN** an unbalanced input with known sequence content
- **WHEN** decomposition runs
- **THEN** positive/negative/zero magnitudes match analytical values within 2%

### Requirement: Grid-Following Inverter Template
The library SHALL provide a `grid_following_inverter_template` instantiating a three-phase inverter, LCL filter, dq-decoupled current loop, and SrfPll, with auto-tuned defaults.

#### Scenario: P/Q command tracking
- **GIVEN** a `grid_following_inverter_template` with `P_ref = 10 kW`, `Q_ref = 5 kVAr`
- **WHEN** the simulation reaches steady state
- **THEN** measured P and Q match references within 5%

#### Scenario: Step in P reference
- **GIVEN** a P-reference step from 5 kW to 10 kW
- **WHEN** the simulation runs
- **THEN** P settles within the designed current-loop time constant
- **AND** Q remains within 5% of its reference during the transient

### Requirement: Grid-Forming Inverter Template
The library SHALL provide a `grid_forming_inverter_template` based on a virtual synchronous machine or droop control.

#### Scenario: Voltage regulation under load step
- **GIVEN** a `grid_forming_inverter_template` regulating 230 V / 50 Hz with a 50% load step
- **WHEN** the simulation runs over the step
- **THEN** voltage RMS deviates by no more than 2% during transient
- **AND** frequency deviates by no more than 0.5 Hz during transient

#### Scenario: Droop coefficient effect
- **GIVEN** a grid-forming template with `P-f droop coefficient = 1%/100%`
- **WHEN** load increases by 50%
- **THEN** steady-state frequency drops by approximately 0.5% (0.25 Hz at 50 Hz)

### Requirement: Anti-Islanding Detection Blocks (Informative)
The library SHALL provide `AfdBlock` and `SfsBlock` reference implementations for IEEE 1547 anti-islanding studies, documented as informative (not certified).

#### Scenario: AFD trip on islanding event
- **GIVEN** an AFD block on an inverter and a simulated grid-disconnection event
- **WHEN** the simulation runs through the disconnect
- **THEN** the AFD output trips within the configured detection window
