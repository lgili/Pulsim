## ADDED Requirements

### Requirement: Motor and Mechanical Component Types
The YAML schema SHALL accept new component types for motors (`pmsm`, `induction_motor`, `bldc_motor`, `dc_motor`), mechanical (`shaft`, `gearbox`, `flywheel_load`, `constant_torque_load`, `fan_load`), sensors (`encoder_quadrature`, `hall_sensor`, `resolver`), and frame transforms (`abc_to_dq`, `dq_to_abc`).

#### Scenario: PMSM with shaft and load
- **GIVEN** a netlist:
```yaml
- type: pmsm
  name: M1
  electrical_nodes: [a, b, c, n]
  mechanical_port: shaft1
  parameters: { rs: 0.5, ld: 1e-3, lq: 1.5e-3, psi_pm: 0.18, pole_pairs: 4 }
- type: shaft
  name: shaft1
  parameters: { j: 0.001, b_friction: 0.0005 }
- type: flywheel_load
  name: load1
  mechanical_port: shaft1
  parameters: { j_load: 0.005 }
```
- **WHEN** the parser loads
- **THEN** the PMSM, shaft, and flywheel are connected via the shared mechanical port `shaft1`
- **AND** the simulation runs with the combined inertia

### Requirement: Mixed-Port Component Definition
The schema SHALL distinguish electrical from mechanical port lists explicitly via `electrical_nodes` and `mechanical_port` fields.

#### Scenario: Mismatch in port domain
- **GIVEN** a netlist linking `electrical_nodes` from a motor to a `shaft`'s mechanical port (incompatible)
- **WHEN** strict parsing runs
- **THEN** parsing fails with `port_domain_mismatch`
- **AND** the diagnostic indicates which entity expected which port domain

### Requirement: Strict Validation of Motor Parameters
Strict validation SHALL reject motor parameter combinations that are physically inconsistent.

#### Scenario: Negative inductance
- **GIVEN** a `pmsm` with `ld: -1e-3`
- **WHEN** strict parsing runs
- **THEN** parsing fails with `invalid_parameter` and a value/range message

#### Scenario: Saturation table out of range
- **GIVEN** a PMSM with a saturation table that has `ld(id)` decreasing then increasing
- **WHEN** strict parsing runs
- **THEN** parsing fails with `non_monotonic_saturation`
