## ADDED Requirements

### Requirement: PWL Switched Diode Forward Voltage Drop

The PWL `SwitchedDiode` SHALL accept a forward drop `V_f` (default 0).
While ON it SHALL conduct `i = g_on·(v − V_f)` (`v = v_anode −
v_cathode`); while OFF it SHALL conduct `i = g_off·v` with no offset. It
SHALL turn ON when `v ≥ V_on`, where `V_on = V_th` if `V_f = 0` and
`max(V_th, V_f)` otherwise, and SHALL stay ON while `g_on·(v − V_f) > 0`.
Its terminal power SHALL be `V_f·i + i²/g_on` while ON. A `V_f` that is
negative or not finite SHALL be refused with an error naming the diode.
With `V_f = 0` the stamp and every decision SHALL be bit-identical to
the diode without the parameter.

#### Scenario: Forced current gives the drop plus the resistive part
- **GIVEN** 10 A forced through a diode with `g_on = 20 S`,
  `V_f = 0.7 V`
- **WHEN** the circuit is solved (DC operating point, pwl or trbdf2)
- **THEN** `v_D = 1.2 V` and the conduction power is 12.0 W

#### Scenario: Turn-on at the larger of V_th and V_f
- **GIVEN** an OFF diode with `V_f = 0.7`
- **WHEN** `V_th = 0` and `v` reaches 0.7 V, or `V_th = 0.75` and `v`
  reaches 0.75 V
- **THEN** the diode turns ON there and not below

#### Scenario: Turn-off at zero current
- **GIVEN** an ON diode with `V_f = 0.7`
- **WHEN** `v` falls to `V_f − ε`
- **THEN** the diode turns OFF; at `V_f + ε` it stays ON

#### Scenario: Zero drop is unchanged
- **WHEN** a circuit is built with `V_f` omitted and with `V_f = 0.0`
- **THEN** the states, switch-state record and loss summary are
  bit-identical

#### Scenario: Invalid drop refused by name
- **WHEN** `add_diode("D1", ..., V_f=-0.1)` or `V_f=nan` is called
- **THEN** it raises an error that names `D1` and `V_f`
