## ADDED Requirements
### Requirement: Canonical Timestep Mode Field
The YAML schema SHALL define a canonical transient mode selector at `simulation.step_mode` with accepted values `fixed` and `variable`.

#### Scenario: Fixed mode selection in YAML
- **WHEN** `simulation.step_mode: fixed` is provided
- **THEN** the parser maps simulation options to fixed-step macro-grid semantics
- **AND** validates required fixed-mode fields

#### Scenario: Variable mode selection in YAML
- **WHEN** `simulation.step_mode: variable` is provided
- **THEN** the parser maps simulation options to adaptive-step semantics
- **AND** validates required variable-mode fields

### Requirement: Mode-Derived Default Profiles
The parser SHALL apply deterministic solver/integrator default profiles derived from `simulation.step_mode`.

#### Scenario: Fixed mode with minimal fields
- **WHEN** fixed mode is provided with only required timing fields
- **THEN** the parser applies fixed-mode default solver policies
- **AND** emits no requirement for expert-only tuning fields

#### Scenario: Variable mode with minimal fields
- **WHEN** variable mode is provided with only required timing fields
- **THEN** the parser applies adaptive-mode default solver policies
- **AND** initializes adaptive controller defaults deterministically

### Requirement: Expert Override Namespace
Advanced solver tuning keys SHALL be accepted under an explicit expert namespace without changing canonical mode semantics.

#### Scenario: Expert override in fixed mode
- **WHEN** YAML includes expert override keys under the documented expert namespace
- **THEN** overrides are applied on top of fixed-mode defaults
- **AND** unsupported expert keys fail with strict diagnostics in strict mode

### Requirement: Legacy Backend-Key Removal Diagnostics
Legacy transient-backend keys in YAML SHALL produce deterministic migration diagnostics in the supported schema.

#### Scenario: Deprecated backend key in strict mode
- **WHEN** YAML contains deprecated backend-selection keys removed by this change
- **THEN** parsing fails with a diagnostic that names the deprecated key
- **AND** suggests `simulation.step_mode` migration mapping
