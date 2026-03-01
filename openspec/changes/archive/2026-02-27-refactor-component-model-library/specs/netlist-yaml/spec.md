## ADDED Requirements

### Requirement: Model Regularization YAML Surface
The YAML schema SHALL provide a `simulation.model_regularization` configuration block for numerical regularization controls used by nonlinear/switching component models.

#### Scenario: Explicit regularization policy in YAML
- **GIVEN** a netlist with `simulation.model_regularization` overrides
- **WHEN** the parser loads the netlist
- **THEN** the runtime options include the configured regularization policy values
- **AND** invalid ranges are reported as parser errors in strict mode

#### Scenario: Safe defaults when block is omitted
- **GIVEN** a netlist without `simulation.model_regularization`
- **WHEN** the parser loads the netlist
- **THEN** runtime uses conservative defaults
- **AND** behavior remains backward-compatible for existing netlists
