## ADDED Requirements

### Requirement: Solver Configuration in YAML
The YAML netlist SHALL allow explicit solver configuration under the `simulation` section.

#### Scenario: Configure solver stack
- **WHEN** the netlist defines `simulation.solver` options (linear, nonlinear, preconditioner, fallback order)
- **THEN** the parser SHALL map them to the v1 simulation options
- **AND** invalid values SHALL produce a clear diagnostic

#### Scenario: Strict validation
- **WHEN** strict validation is enabled
- **THEN** unknown solver fields SHALL cause parsing to fail
