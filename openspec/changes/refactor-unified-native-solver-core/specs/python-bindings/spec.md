## ADDED Requirements
### Requirement: Canonical Mode-Based Transient Configuration
Python bindings SHALL expose a canonical transient mode selection equivalent to YAML `step_mode` semantics (`fixed` or `variable`).

#### Scenario: Configure fixed mode from Python
- **WHEN** Python code selects fixed mode through the canonical runtime API
- **THEN** the transient run executes with fixed-step macro-grid semantics
- **AND** output sampling follows deterministic fixed-grid behavior

#### Scenario: Configure variable mode from Python
- **WHEN** Python code selects variable mode through the canonical runtime API
- **THEN** the transient run executes with adaptive-step semantics
- **AND** telemetry includes adaptive acceptance/rejection metrics

### Requirement: Hybrid Segment Runtime Semantics in Python
Python bindings SHALL map canonical mode selection to hybrid segment-first runtime behavior without requiring legacy backend selectors.

#### Scenario: Canonical mode uses segment-first runtime
- **WHEN** Python config selects either `fixed` or `variable` canonical mode
- **THEN** runtime attempts state-space segment solve as primary path
- **AND** uses nonlinear DAE fallback deterministically when segment admissibility fails

### Requirement: Expert Override Exposure
Python bindings SHALL provide explicit expert override controls without requiring them for standard transient use.

#### Scenario: Standard use without expert overrides
- **WHEN** Python code configures only canonical mode and timing fields
- **THEN** the simulation runs with deterministic mode-derived default profiles

#### Scenario: Expert override application
- **WHEN** Python code supplies expert override controls
- **THEN** overrides are applied on top of canonical mode defaults
- **AND** invalid expert keys or values produce structured errors

### Requirement: Legacy Transient Configuration Migration Diagnostics
Python bindings SHALL provide deterministic migration diagnostics for removed legacy transient-backend controls.

#### Scenario: Deprecated legacy backend field usage
- **WHEN** Python code attempts to use deprecated backend-specific transient controls removed from supported runtime
- **THEN** bindings raise a structured configuration error
- **AND** include migration guidance to canonical mode-based controls

### Requirement: Electrothermal and Loss Surface Support
Python bindings SHALL expose loss and thermal configuration/results in canonical mode workflows.

#### Scenario: Configure electrothermal options from Python
- **WHEN** Python config enables losses and thermal coupling on a canonical mode run
- **THEN** runtime accepts the configuration without requiring legacy backend controls
- **AND** simulation results include `loss_summary` and `thermal_summary`

#### Scenario: Invalid electrothermal override values
- **WHEN** Python config provides invalid thermal/loss override values
- **THEN** bindings raise structured configuration errors
- **AND** preserve deterministic error messaging
