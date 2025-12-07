## ADDED Requirements

### Requirement: Lumped Thermal Model
The kernel SHALL support lumped-element thermal modeling.

#### Scenario: Thermal node definition
- **WHEN** a thermal model is defined
- **THEN** thermal nodes represent temperatures at key points
- **AND** ambient temperature is the reference (thermal ground)

#### Scenario: Thermal resistance
- **WHEN** thermal resistance Rth is defined between nodes
- **THEN** heat flow Q = (T1 - T2) / Rth
- **AND** units are K/W or C/W

#### Scenario: Thermal capacitance
- **WHEN** thermal capacitance Cth is defined at a node
- **THEN** dT/dt = Q / Cth
- **AND** units are J/K

### Requirement: Foster and Cauer Networks
The kernel SHALL support standard thermal network representations.

#### Scenario: Foster network
- **WHEN** a Foster thermal model is specified
- **THEN** it is a series of parallel RC stages
- **AND** parameters are (Rth_i, tau_i) pairs from datasheet

#### Scenario: Cauer network
- **WHEN** a Cauer thermal model is specified
- **THEN** it is a ladder of series R with shunt C
- **AND** represents physical thermal layers

#### Scenario: Datasheet thermal model
- **WHEN** a device datasheet provides Zth(t) curve
- **THEN** Foster parameters are extracted
- **AND** thermal response matches the curve

### Requirement: Electro-Thermal Coupling
The kernel SHALL couple electrical and thermal simulation.

#### Scenario: Temperature-dependent parameters
- **WHEN** temperature is computed at a device
- **THEN** device parameters are updated: Rds_on(T), Vth(T), etc.
- **AND** the electrical simulation uses updated values

#### Scenario: Loss-to-heat coupling
- **WHEN** power losses are computed in a device
- **THEN** the loss power is injected into the thermal network
- **AND** as a heat source at the device junction node

#### Scenario: Coupling iteration
- **WHEN** electro-thermal coupling is enabled
- **THEN** electrical and thermal are iterated to consistency
- **AND** or solved simultaneously if strongly coupled

### Requirement: Junction Temperature Calculation
The kernel SHALL compute junction temperatures for semiconductors.

#### Scenario: Single device thermal
- **WHEN** a device has thermal model
- **THEN** Tj = Tamb + Ploss * Rth_ja
- **AND** transient response follows Zth(t)

#### Scenario: Multiple devices on heatsink
- **WHEN** devices share a heatsink
- **THEN** thermal network includes cross-coupling
- **AND** Tj of each device depends on all device losses

#### Scenario: Thermal limit monitoring
- **WHEN** Tj exceeds a threshold (e.g., 150C)
- **THEN** a warning is issued
- **AND** optionally simulation stops (thermal failure)

### Requirement: Thermal Sources
The kernel SHALL support thermal boundary conditions.

#### Scenario: Fixed temperature source
- **WHEN** a thermal node is fixed to a temperature
- **THEN** it acts as an ideal heat sink
- **AND** absorbs any heat flow

#### Scenario: Convection boundary
- **WHEN** convection is modeled
- **THEN** Q = h * A * (Tsurface - Tfluid)
- **AND** h is the convection coefficient

#### Scenario: Time-varying ambient
- **WHEN** ambient temperature varies with time
- **THEN** the thermal reference tracks the ambient profile
- **AND** affects all temperatures in the network

### Requirement: Thermal Output
The kernel SHALL output thermal quantities.

#### Scenario: Temperature waveforms
- **WHEN** thermal simulation is enabled
- **THEN** temperature at each thermal node vs time is recorded
- **AND** available in the same format as electrical waveforms

#### Scenario: Thermal impedance extraction
- **WHEN** a step power is applied
- **THEN** the Zth(t) response can be computed
- **AND** compared against datasheet

#### Scenario: Maximum temperature detection
- **WHEN** simulation completes
- **THEN** peak junction temperature is reported
- **AND** time of peak is recorded
