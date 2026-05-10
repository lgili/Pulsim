# motor-models Specification

## Purpose
TBD - created by archiving change add-motor-models. Update Purpose after archive.
## Requirements
### Requirement: Three-Phase PMSM Device
The library SHALL provide a `PmsmDevice` modeling a three-phase permanent-magnet synchronous motor in the dq reference frame.

#### Scenario: PMSM open-circuit
- **GIVEN** a `PmsmDevice` driven externally at `omega_m`
- **WHEN** the electrical terminals are open-circuit
- **THEN** the back-EMF magnitude equals `psi_pm * omega_e` within 5%
- **AND** the back-EMF frequency equals `omega_m * pole_pairs / (2π)` within numerical tolerance

#### Scenario: PMSM locked rotor
- **GIVEN** a `PmsmDevice` with shaft locked
- **WHEN** AC voltage is applied to dq terminals
- **THEN** the dq currents match `(Vd, Vq) / (Rs + jωL)` impedance within 5%

### Requirement: Induction Motor Device
The library SHALL provide an `InductionMotorDevice` modeling a three-phase squirrel-cage induction motor in dq frame.

#### Scenario: IM no-load operation
- **GIVEN** an `InductionMotorDevice` at synchronous speed with no mechanical load
- **WHEN** the simulation reaches steady state
- **THEN** the slip is approximately zero
- **AND** the rotor current magnitude is below 5% of stator nominal

#### Scenario: IM locked-rotor parameter identification
- **GIVEN** an `InductionMotorDevice` with shaft locked
- **WHEN** AC voltage is applied
- **THEN** the input impedance matches the locked-rotor analytical formula within 5%

### Requirement: BLDC Motor with Trapezoidal Back-EMF
The library SHALL provide a `BldcMotorDevice` with trapezoidal back-EMF and hall-sensor commutation signals.

#### Scenario: BLDC commutation events
- **GIVEN** a `BldcMotorDevice` rotating at constant speed
- **WHEN** hall sensors are read
- **THEN** transitions occur at electrical-position multiples of 60° within tolerance
- **AND** the back-EMF waveform exhibits the expected trapezoidal shape

### Requirement: DC Motor Device
The library SHALL provide a `DcMotorDevice` supporting separately-excited and series-excited configurations.

#### Scenario: DC motor speed step response
- **GIVEN** a `DcMotorDevice` with separately excited field and a step in armature voltage
- **WHEN** the simulation runs to steady state
- **THEN** the speed transient matches the first-order analytical response (`τ_mech`) within 5%

### Requirement: Mechanical Subsystem
The library SHALL provide mechanical components — `Shaft`, `GearBox`, `FlywheelLoad`, `ConstantTorqueLoad`, `FanLoad` — that compose with motor devices via mechanical ports.

#### Scenario: Shaft + flywheel step response
- **GIVEN** a shaft connected to a flywheel load with combined inertia `J`
- **WHEN** a step torque `τ` is applied
- **THEN** the angular acceleration equals `τ/J` within 1%
- **AND** the integrated speed is consistent over time

#### Scenario: Gearbox ratio and efficiency
- **GIVEN** a `GearBox { ratio: 5, efficiency: 0.95 }` between motor and load
- **WHEN** the motor delivers torque `τ_m` at speed `ω_m`
- **THEN** the load sees `τ_l = τ_m * 5 * 0.95` and `ω_l = ω_m / 5`

### Requirement: Frame Transformation Blocks
The library SHALL provide `AbcToDq` and `DqToAbc` Park/Clarke transformation blocks, available as both signal-domain (for control) and electrical-domain (for analytical terminals).

#### Scenario: Park transform round-trip
- **GIVEN** an arbitrary three-phase signal `(va, vb, vc)` and rotor angle `theta`
- **WHEN** `AbcToDq` is followed by `DqToAbc` with the same `theta`
- **THEN** the output equals the input within numerical noise

### Requirement: PMSM-FOC Template
The library SHALL provide a `pmsm_foc_template` instantiating current loops, speed loop, and optional position loop with auto-tuned defaults from motor parameters.

#### Scenario: FOC template current decoupling
- **GIVEN** a `pmsm_foc_template` with id and iq references
- **WHEN** simulation runs with step on iq
- **THEN** id remains within 5% of its reference
- **AND** iq settles to its reference within the designed current-loop time constant

#### Scenario: FOC template speed step
- **GIVEN** a `pmsm_foc_template` with speed reference step
- **WHEN** simulation runs
- **THEN** the speed reaches the reference within 20% of the designed bandwidth time constant
- **AND** id, iq stay within compliance limits

### Requirement: Encoder and Hall Sensor Models
The library SHALL provide `EncoderQuadrature`, `HallSensor`, and `Resolver` models suitable for FOC and BLDC commutation.

#### Scenario: Encoder counts shaft revolutions
- **GIVEN** an `EncoderQuadrature { ppr: 1024 }` on a shaft rotating one full revolution
- **WHEN** the encoder counts are read
- **THEN** the count delta equals 4096 (4× quadrature × ppr)

#### Scenario: Hall sensor state at electrical positions
- **GIVEN** a `HallSensor` configured for a 4-pole BLDC
- **WHEN** the rotor passes electrical positions 0°, 60°, 120°, ...
- **THEN** the hall state output transitions at each multiple of 60° electrical

