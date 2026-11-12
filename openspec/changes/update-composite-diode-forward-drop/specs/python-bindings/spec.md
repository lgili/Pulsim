## ADDED Requirements

### Requirement: Composite Device V_F Keyword Is a Conduction Drop

`CircuitBuilder.add_mosfet` and `add_mosfet_with_body_diode` SHALL
document and apply `V_F` as the body diode's conduction drop (BREAKING:
it was the turn-on threshold). `add_mosfet_level1` and `add_igbt_level1`
SHALL accept `V_F` (keyword, default 0.7) for their body and freewheel
diodes. `components()` SHALL report `V_f = V_F` and `V_th = 0` for the
`<name>_body` and `<name>_fwd` branches, and `res.i`,
`device_loss_summary` and `device_thermal_summary` SHALL include the
`V_F·i` term for them. The YAML device types `mosfet`,
`mosfet_with_body_diode`, `mosfet_level1` and `igbt_level1` SHALL accept
`V_F` with the same meaning.

#### Scenario: Loss summary includes the drop
- **GIVEN** `add_mosfet_with_body_diode("Q", d, s, V_F=0.7,
  g_on_diode=20.0)` with the gate off and 10 A forced source to drain
- **WHEN** `device_loss_summary` is evaluated
- **THEN** `Q_body` reports `V_f = 0.7` and `P_avg = 12.0 W`, and
  `res.i("Q_body")` reads 10 A

#### Scenario: Level-1 keyword
- **WHEN** `add_mosfet_level1("M", d, s, g, K, V_T, V_F=1.0)` is
  called
- **THEN** `components()` lists `M_body` with `V_f = 1.0` and
  `V_th = 0.0`

#### Scenario: YAML
- **WHEN** a netlist declares `type: mosfet_with_body_diode` with
  `V_F: 0.7` and is run with 10 A forced through the body diode
- **THEN** the diode drops 0.710 V

#### Scenario: Zero drop
- **WHEN** `add_mosfet(..., V_F=0.0)` is used in the forced-current
  circuit with the gate off
- **THEN** `v_SD = 0.010 V`
