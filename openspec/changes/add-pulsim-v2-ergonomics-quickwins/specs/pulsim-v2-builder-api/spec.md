## ADDED Requirements

### Requirement: MOSFET Body Diode Auto-Helper

The `CircuitBuilder::add_mosfet_level1` method SHALL accept an optional `with_body_diode` parameter. When `true`, the builder SHALL automatically add an anti-parallel `SwitchedDiode` from `source` to `drain` with sensible default parameters (`g_on = 1e3`, `g_off = 1e-9`, `V_th = 0.5`).

#### Scenario: with_body_diode adds an extra branch
- **WHEN** a user calls `builder.add_mosfet_level1("M1", "d", "s", "g", K, V_T, with_body_diode=true)`
- **THEN** the resulting graph SHALL contain 2 branches (one Nonlinear for the MOSFET, one Switch for the body diode)
- **AND** the body diode's anode SHALL be the source node and cathode SHALL be the drain node

#### Scenario: Default value preserves existing API
- **WHEN** a user calls `add_mosfet_level1(...)` without specifying `with_body_diode`
- **THEN** only 1 branch (the MOSFET) SHALL be added — backward-compatible with all existing tests
