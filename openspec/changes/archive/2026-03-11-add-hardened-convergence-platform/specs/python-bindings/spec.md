## ADDED Requirements
### Requirement: Convergence Policy Configuration in Python
Python bindings SHALL expose convergence-policy profiles and bounded tuning options for transient execution.

#### Scenario: Select robust convergence profile
- **WHEN** Python code selects profile `robust` for a challenging circuit
- **THEN** runtime receives corresponding policy configuration
- **AND** strict-mode contracts remain explicit and deterministic

#### Scenario: Override bounded policy knobs
- **WHEN** Python code overrides approved policy fields (for example event-burst guard or regularization bounds)
- **THEN** overrides are validated with deterministic errors for invalid ranges
- **AND** accepted values are reflected in runtime options

### Requirement: Structured Convergence Telemetry Exposure
Python result objects SHALL expose structured convergence telemetry produced by the policy engine.

#### Scenario: Inspect convergence diagnostics from Python
- **WHEN** a transient run completes (success or failure)
- **THEN** Python code can access failure classes, recovery stages, and policy actions as typed fields
- **AND** no diagnostic workflow requires string parsing
