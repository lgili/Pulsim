## ADDED Requirements

### Requirement: Conduction Loss Calculation
The kernel SHALL calculate conduction losses in power devices.

#### Scenario: Resistive conduction loss
- **WHEN** current flows through a resistive element
- **THEN** Pcond = I^2 * R
- **AND** instantaneous and average losses are computed

#### Scenario: MOSFET conduction loss
- **WHEN** a MOSFET conducts
- **THEN** Pcond = I^2 * Rds_on(Vgs, T)
- **AND** Rds_on depends on gate voltage and temperature

#### Scenario: IGBT conduction loss
- **WHEN** an IGBT conducts
- **THEN** Pcond = Vce_sat * I + Rce * I^2
- **AND** Vce_sat and Rce depend on temperature

#### Scenario: Diode conduction loss
- **WHEN** a diode conducts
- **THEN** Pcond = Vf * I + Rd * I^2
- **AND** Vf and Rd depend on temperature

### Requirement: Switching Loss Calculation
The kernel SHALL calculate switching losses during transitions.

#### Scenario: Turn-on energy
- **WHEN** a switch turns on
- **THEN** Eon is computed from the model or lookup table
- **AND** Eon = f(I, V, Rg, T) from datasheet curves

#### Scenario: Turn-off energy
- **WHEN** a switch turns off
- **THEN** Eoff is computed from the model or lookup table
- **AND** Eoff = f(I, V, Rg, T) from datasheet curves

#### Scenario: Diode reverse recovery loss
- **WHEN** a diode turns off
- **THEN** Err is computed from recovery characteristics
- **AND** Err = f(I, dI/dt, T)

#### Scenario: Switching loss interpolation
- **WHEN** operating point differs from datasheet conditions
- **THEN** losses are interpolated or scaled
- **AND** common scaling: Esw proportional to I and V

### Requirement: Loss Model Parameters
The kernel SHALL support various loss model parameterizations.

#### Scenario: Analytical model
- **WHEN** analytical loss equations are used
- **THEN** parameters are: Rds_on, Qg, Qrr, etc.
- **AND** losses are computed from equations

#### Scenario: Lookup table model
- **WHEN** lookup tables are provided
- **THEN** Eon(I, V), Eoff(I, V), Err(I, dI/dt) tables are used
- **AND** bilinear interpolation is applied

#### Scenario: Curve fitting from datasheet
- **WHEN** datasheet graphs are provided
- **THEN** points are extracted and fitted to polynomial or spline
- **AND** extrapolation is bounded

### Requirement: Loss Integration
The kernel SHALL integrate losses over time.

#### Scenario: Energy accumulation
- **WHEN** simulation runs
- **THEN** total energy loss per device is accumulated
- **AND** Etotal = integral(Ploss * dt)

#### Scenario: Average power loss
- **WHEN** a period is defined (or detected)
- **THEN** Pavg = Etotal / Tperiod
- **AND** useful for thermal steady-state

#### Scenario: Loss breakdown
- **WHEN** loss analysis is requested
- **THEN** losses are broken down: conduction, turn-on, turn-off, reverse recovery
- **AND** per device and system total

### Requirement: Efficiency Calculation
The kernel SHALL calculate converter efficiency.

#### Scenario: Power flow measurement
- **WHEN** input and output power are measurable
- **THEN** Pin and Pout are computed from V*I products
- **AND** averaged over a switching period

#### Scenario: Efficiency computation
- **WHEN** power flows are known
- **THEN** eta = Pout / Pin
- **AND** losses = Pin - Pout

#### Scenario: Loss distribution
- **WHEN** efficiency analysis completes
- **THEN** a pie chart of losses is available
- **AND** identifies dominant loss sources

### Requirement: Loss Output
The kernel SHALL output loss quantities.

#### Scenario: Instantaneous loss waveform
- **WHEN** detailed loss analysis is enabled
- **THEN** Ploss(t) for each device is available
- **AND** includes conduction and switching components

#### Scenario: Per-cycle loss table
- **WHEN** periodic operation is detected
- **THEN** losses per switching cycle are tabulated
- **AND** useful for analyzing load-dependent behavior

#### Scenario: Thermal input preparation
- **WHEN** thermal analysis is enabled
- **THEN** losses are formatted as heat sources
- **AND** injected into the thermal network

### Requirement: Dead-Time and Overlap Handling
The kernel SHALL handle practical switching considerations.

#### Scenario: Dead-time loss
- **WHEN** dead-time is present in half-bridge
- **THEN** body diode conduction during dead-time is captured
- **AND** diode losses are added to the device

#### Scenario: Shoot-through detection
- **WHEN** both switches in a leg are on simultaneously
- **THEN** a warning is issued
- **AND** high current flow is simulated (not infinite)

#### Scenario: Overlap loss
- **WHEN** turn-on and turn-off overlap in time
- **THEN** overlap current is modeled
- **AND** contributes to switching loss
