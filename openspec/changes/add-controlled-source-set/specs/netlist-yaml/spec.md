## ADDED Requirements

### Requirement: Controlled Source Component Types

The YAML loader SHALL accept the component types `vccs` (keys `in_pos`,
`in_neg`, `from`, `to`, `gm`), `ccvs` (`control`, `out_pos`, `out_neg`,
`rm`), `cccs` (`control`, `from`, `to`, `gain`) and `gyrator`
(`p1_pos`, `p1_neg`, `p2_pos`, `p2_neg`, `g`). Validation SHALL be
strict: an unknown key, a missing required key, or a `control` that
does not name a branch with a current unknown SHALL fail with an error
naming the component.

#### Scenario: Gyrator from YAML
- **WHEN** a netlist declares `type: gyrator` with `g: 0.01` and the
  four port nodes
- **THEN** the loaded circuit's `components()` contains a gyrator with
  `g = 0.01`, identical to one built with `add_gyrator`

#### Scenario: Unknown controlling branch
- **WHEN** a `cccs` names `control: Rx` and no such sensable branch
  exists
- **THEN** loading fails with an error naming the `cccs` and `Rx`

### Requirement: Current Source Waveform Component Types

The YAML loader SHALL accept `sine_current_source` (keys `from`, `to`,
`i_dc`, `i_amplitude`, `frequency`, optional `phase`),
`pulse_current_source` (`from`, `to`, `i_initial`, `i_pulsed`,
`t_start`, `pulse_width`, optional `period`, `rise_time`, `fall_time`)
and `pwl_current_source` (`from`, `to`, and either `t` and `i` lists of
equal length or a `file` path, optional `periodic`). Invalid tables
SHALL be refused with the builder's error.

#### Scenario: Table source from lists
- **WHEN** a netlist declares `type: pwl_current_source` with
  `t: [0, 1e-3, 2e-3]` and `i: [0, 1, 0]`
- **THEN** the loaded source produces 0.5 A at 0.5 ms

#### Scenario: Mismatched lists refused
- **WHEN** `t` has three entries and `i` has two
- **THEN** loading fails with an error naming the source
