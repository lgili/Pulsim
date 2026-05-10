## ADDED Requirements

### Requirement: Three-Phase Component Types
The YAML schema SHALL accept new types for three-phase sources, PLLs, frame transforms, sequence decomposition, anti-islanding blocks, and grid-tied inverter templates.

#### Scenario: Programmable three-phase source with events
- **GIVEN** a netlist:
```yaml
- type: three_phase_source_programmable
  name: Vgrid
  nodes_abc: [a, b, c]
  neutral: n
  parameters:
    v_rms: 230
    frequency: 50
    phase_seq: abc
    events:
      - { type: sag, phase: A, ratio: 0.5, t_start: 0.1, t_end: 0.2 }
```
- **WHEN** the parser loads
- **THEN** the source is built with the sag event programmed
- **AND** strict validation checks event ranges (`0 < ratio < 1.5`, `t_end > t_start`)

#### Scenario: Grid-following inverter template
- **GIVEN** `type: grid_following_inverter_template` with `vdc, p_ref, q_ref` parameters
- **WHEN** parser loads
- **THEN** the template expands into inverter + LCL + control sub-netlist
- **AND** auto-tuned defaults produce a stable closed loop

### Requirement: Strict Validation of Three-Phase Configurations
Strict validation SHALL detect inconsistent three-phase configurations.

#### Scenario: Mismatched phase sequence in connected components
- **GIVEN** a three-phase source with `phase_seq: abc` connected to a load expecting `acb`
- **WHEN** strict parsing runs
- **THEN** parsing fails or warns with `phase_sequence_mismatch` per configured policy
- **AND** the diagnostic suggests verifying `phase_seq` on each three-phase component

#### Scenario: Invalid PLL parameters
- **GIVEN** an `srf_pll` with `kp: -10` (negative)
- **WHEN** strict parsing runs
- **THEN** parsing fails with `invalid_parameter` and the field name
