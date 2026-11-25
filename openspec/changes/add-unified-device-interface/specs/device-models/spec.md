## ADDED Requirements

### Requirement: Built-In Device Kinds Implement The Unified Interface

Every built-in device kind SHALL belong to exactly one category of the
`device-interface` capability: static (resistor, capacitor, inductor,
the voltage and current sources, VCVS, ideal transformer, mutual
coupling), mode (ideal switch, PWL switched diode) or Newton (smooth
diode, Shockley diode, level-1 MOSFET, level-1 IGBT and its tail,
charge-based capacitor, Lauritzen diode, the saturable-inductor family
including the Jiles–Atherton core, MNA PMSM). A Newton kind SHALL carry
no time-integration code of its own: charge and flux SHALL be expressed
as `q(x)`, and internal charges or currents as internal unknowns. A
port SHALL NOT change a kind's device law.

#### Scenario: Ported kind reproduces its own tests
- **WHEN** a kind is moved to the interface
- **THEN** its existing device tests pass unmodified on the fixed and
  the variable-step engine, and harness circuits containing it agree
  with the previous build within 1e-10 relative

#### Scenario: Reverse recovery and tail survive the port
- **GIVEN** the double-pulse circuit with a Lauritzen freewheel diode
  and the IGBT tail regression circuit
- **WHEN** they run after the port
- **THEN** the recovery peak current and the turn-off energy are within
  the existing test tolerances and the two engines agree within 0.1 %

#### Scenario: No kind integrates itself
- **WHEN** the source tree is searched after the migration
- **THEN** no file under `models/` or `device/` contains a trapezoidal
  or BDF2 coefficient, and the per-kind history and refresh headers no
  longer exist

### Requirement: Generic Device Current And Power

`res.i(name)` and `res.p(name)` SHALL be available for every device
kind on every engine, computed by the kernel from the device's stamped
law (`i = f + d/dt q` for a Newton kind, the mode law for a PWL kind,
the companion law for a static kind). They MUST NOT raise
`NotImplementedError` for a supported kind, and MUST NOT differentiate
stored samples numerically. `device_loss_summary` and
`device_thermal_summary` SHALL list Newton devices, with energy obtained
by integrating power over time.

#### Scenario: Newton device current against a sense resistor
- **GIVEN** each of `mosfet_level1`, `igbt_level1`, `lauritzen_diode`,
  `nonlinear_diode`, `nonlinear_capacitor` and a saturable inductor in
  series with a 1 mΩ resistor
- **WHEN** the circuit runs on the fixed and on the variable-step engine
- **THEN** `∫ v·i dt` of the device computed from `res.i` equals the
  value computed with the resistor current within 1e-4 relative

#### Scenario: Existing currents unchanged
- **WHEN** `res.i` is read for resistors, switches, PWL diodes (with
  `V_f`), sources and inductors on the harness circuits
- **THEN** the values equal the previous build within 1e-12 relative

#### Scenario: Loss summary lists Newton devices
- **WHEN** `device_loss_summary` runs on the double-pulse circuit
- **THEN** it contains an entry for the level-1 MOSFET and for the
  Lauritzen diode, and the entries sum, with the passive losses, to the
  source energy within 1e-3 relative on both engines

#### Scenario: Adaptive grid is time-weighted
- **GIVEN** the same circuit on the fixed and the variable-step engine
- **WHEN** average device power is compared
- **THEN** the two agree within 0.5 % (the variable-step value is an
  integral over time, not a mean over samples)
