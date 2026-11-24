## ADDED Requirements

### Requirement: Switched Diode V_f Key

The YAML `diode` device SHALL accept an optional `V_f` key (default 0),
the forward drop of the PWL switched diode, with the same validation as
`add_diode`.

#### Scenario: V_f key changes the result
- **WHEN** a YAML netlist sets `V_f: 0.7` on a `diode`
- **THEN** the loaded circuit's diode has `V_f = 0.7` in `components()`
  and its forward voltage includes the drop

#### Scenario: Invalid V_f refused
- **WHEN** a YAML `diode` sets `V_f: -1`
- **THEN** loading fails with an error naming the diode
