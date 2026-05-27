## MODIFIED Requirements

### Requirement: Induction Motor Device

The library SHALL provide an `InductionMotorDevice` modeling a three-phase squirrel-cage induction motor with a 5th-order dq state-space representation in a configurable reference frame (stationary or synchronous). The device SHALL expose stator currents, rotor flux components, electrical torque, slip, and mechanical speed as observable channels.

#### Scenario: IM no-load operation

- **GIVEN** an `InductionMotorDevice` at synchronous speed with no mechanical load
- **WHEN** the simulation reaches steady state
- **THEN** the slip is approximately zero within `1e-3`
- **AND** the rotor current magnitude is below 5 % of stator nominal

#### Scenario: IM locked-rotor parameter identification

- **GIVEN** an `InductionMotorDevice` with shaft locked
- **WHEN** AC voltage is applied
- **THEN** the input impedance matches the locked-rotor analytical formula within 5 %

#### Scenario: IM slip-torque steady-state curve

- **GIVEN** an `InductionMotorDevice` driven by a fixed three-phase sinusoidal source at rated voltage and frequency
- **WHEN** the mechanical speed is swept across the full slip range from 0 to 1
- **THEN** the simulated electrical-torque vs slip curve matches the analytical Kloss formula within 5 % over the full operating range
- **AND** the peak torque occurs at the analytical slip-at-pullout `s_p = Rr / sqrt(Rs² + (ω·(Ls+Lr))²)` within 5 %

#### Scenario: V/f open-loop ramp from standstill

- **GIVEN** an `InductionMotorDevice` connected to a three-phase voltage source whose magnitude and frequency ramp proportionally from 0 V / 0 Hz to rated voltage / 50 Hz over 0.5 s
- **WHEN** the simulation runs with a constant load torque equal to 50 % of rated
- **THEN** the mechanical speed reaches the steady-state slip operating point within 1 s of the ramp end
- **AND** the stator current envelope stays within 1.5× rated during the ramp

#### Scenario: Indirect FOC torque step response

- **GIVEN** an `InductionMotorDevice` controlled by an indirect-FOC closed-loop chain (slip-frequency computation + decoupled `id`/`iq` PI loops)
- **WHEN** the `iq` reference is stepped from 0 to rated value
- **THEN** the electrical torque reaches 90 % of the analytical steady-state value within one electrical cycle
- **AND** the rotor flux magnitude remains within ±5 % of the commanded value during the transient

### Requirement: PMSM-FOC Template

The library SHALL provide a closed-loop FOC drive template wiring a `PmsmDevice` to a three-phase VSI, current sensors, abc→dq transform, decoupled PI controllers, dq→abc transform, and SVM modulator.

#### Scenario: PMSM-FOC current step

- **GIVEN** a configured PMSM-FOC template at rated speed
- **WHEN** the `iq_ref` is stepped from 0 to rated
- **THEN** the actual `iq` reaches 90 % of `iq_ref` within one electrical period
- **AND** the simulated electromagnetic torque tracks `1.5 * p * psi_pm * iq` within 5 %
