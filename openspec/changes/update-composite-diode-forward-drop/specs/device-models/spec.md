## ADDED Requirements

### Requirement: Composite Body and Freewheel Diode Forward Drop

The diode inside a composite switching device SHALL take the device's
`V_F` as its conduction drop. This applies to the `<name>_body` diode of
`add_mosfet`, `add_mosfet_with_body_diode` and `add_mosfet_level1`, and
to the `<name>_fwd` diode of `add_igbt_level1`. The diode SHALL be a PWL
switched diode with `V_f = V_F` and `V_th = 0`: it SHALL turn on at
`v ≥ V_F`, conduct `i = g_on·(v − V_F)` while ON, conduct `g_off·v`
while OFF, and turn off when its ON current reaches zero. `V_F` SHALL
default to 0.7 V on all four builders. A `V_F` that is negative or not
finite SHALL be refused with an error naming the device. `add_igbt`
SHALL remain a controlled switch with no diode.

#### Scenario: Body diode carries the load with the gate off
- **GIVEN** `add_mosfet` with `R_on = 1 mΩ`, `V_F = 0.7 V` and the
  default diode conductance 1e3 S, gate off
- **WHEN** 10 A are forced from source to drain
- **THEN** `v_SD = 0.710 V` and the diode dissipates 7.10 W (it read
  0.010 V and 0.10 W before)

#### Scenario: Channel conducts with the gate on
- **WHEN** the same 10 A are forced with the gate on
- **THEN** `v_SD = 0.010 V`, the body diode carries less than 1e-6 A
  and the result is the same as before this requirement

#### Scenario: Turn-on voltage is V_F
- **WHEN** the voltage across an OFF body diode with `V_F = 0.7 V`
  rises
- **THEN** the diode turns ON at 0.7 V and not below

#### Scenario: Level-1 devices
- **WHEN** 10 A are forced through the `_body` diode of
  `add_mosfet_level1` with the gate at 0 V, and through the `_fwd`
  diode of `add_igbt_level1(with_fwd=True)` with the gate off, both
  with the default `V_F`
- **THEN** each diode drops 0.710 V (0.010 V before), and with
  `V_F = 1.0` each drops 1.010 V

#### Scenario: Freewheeling buck
- **GIVEN** a 24 V buck at `D = 0.5`, 100 kHz, 100 µH, 100 µF, 1.2 Ω,
  whose low-side device is an `add_mosfet` held off
- **WHEN** 400 periods run on `pwl`, `trbdf2` and `dsed`
- **THEN** the time-weighted output over the last 100 periods is
  11.640 V within 0.005 V and the body-diode loss is 3.479 W within
  0.01 W on each engine (11.990 V and 0.052 W before)

#### Scenario: Invalid drop refused by name
- **WHEN** `add_mosfet("Q1", ..., V_F=-0.1)` is called
- **THEN** it raises an error naming `Q1` and `V_F`
