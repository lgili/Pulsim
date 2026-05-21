## ADDED Requirements

### Requirement: CircuitBuilder.add_mosfet helper

`CircuitBuilder::add_mosfet(name, drain, source, R_on, R_off)` SHALL add a single controlled-switch branch from the drain node to the source node with `g_on = 1/R_on` and `g_off = 1/R_off`.

The defaults `R_on = 1 mΩ`, `R_off = 1 GΩ` MUST apply when the caller omits them.

The helper MUST be functionally equivalent to
`add_switch(name, drain, source, 1/R_on, 1/R_off)`.

#### Scenario: add_mosfet adds one branch with correct conductance

- **GIVEN** a fresh `CircuitBuilder`
- **WHEN** the user calls
  `add_mosfet("Q1", "drain", "source", R_on=2e-3,
              R_off=1e9)`
- **THEN** `num_branches` SHALL be 1
- **AND** the pool's `kind_of(0)` SHALL be `Switch`
- **AND** the pool's `switch_g_on(0)` SHALL equal
  `1.0 / 2e-3` within numerical precision.

### Requirement: CircuitBuilder.add_mosfet_with_body_diode helper

`CircuitBuilder::add_mosfet_with_body_diode(name, drain, source, R_on, R_off, V_F, g_on_diode, g_off_diode)` SHALL add TWO branches in sequence:

1. A controlled switch from drain to source with the same
   semantics as `add_mosfet`.
2. An anti-parallel `SwitchedDiode` from source to drain
   (note the REVERSED direction) with `V_th = V_F`,
   `g_on = g_on_diode`, `g_off = g_off_diode`.

The default `V_F = 0.7 V` MUST apply when omitted, modeling
a typical silicon body diode. The diode's branch ID MUST
equal the switch's branch ID + 1.

#### Scenario: add_mosfet_with_body_diode adds two branches with diode antiparallel

- **GIVEN** a fresh `CircuitBuilder`
- **WHEN** the user calls
  `add_mosfet_with_body_diode("Q1", "drain", "source")`
- **THEN** `num_branches` SHALL be 2
- **AND** the switch branch SHALL go from "drain" → "source"
- **AND** the diode branch SHALL go from "source" → "drain"
  (anti-parallel)
- **AND** the diode's `V_th` MUST equal 0.7 V.

### Requirement: CircuitBuilder.add_igbt helper

`CircuitBuilder::add_igbt(name, collector, emitter, R_on, R_off)` SHALL add a single controlled-switch branch from the collector node to the emitter node with `g_on = 1/R_on` and `g_off = 1/R_off`.

The defaults `R_on = 10 mΩ`, `R_off = 1 GΩ` MUST apply when omitted.

No anti-parallel diode is added — discrete IGBTs typically
don't include one. Users wire a separate `add_diode` call
if their topology requires freewheeling.

#### Scenario: add_igbt uses IGBT-default R_on

- **GIVEN** a fresh `CircuitBuilder`
- **WHEN** the user calls
  `add_igbt("T1", "C", "E")` with all defaults
- **THEN** `num_branches` SHALL be 1
- **AND** `switch_g_on(0)` SHALL equal `1.0 / 10e-3 = 100 S`
  within numerical precision.
