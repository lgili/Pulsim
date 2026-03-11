## ADDED Requirements
### Requirement: Convergence Policy Engine
The v1 kernel SHALL provide a convergence policy engine that classifies transient failures and selects context-aware recovery actions instead of relying only on retry ordinals.

#### Scenario: Failure classified as event-burst zero-cross
- **WHEN** repeated failures occur near dense switching boundaries around zero crossing
- **THEN** the failure class is set to `event_burst_zero_cross`
- **AND** recovery policy applies event-aware backoff/guard actions instead of generic retry-only behavior

#### Scenario: Failure classified as control-discrete stiffness
- **WHEN** convergence degradation correlates with discrete control update boundaries
- **THEN** the failure class is set to `control_discrete_stiffness`
- **AND** policy applies control-aware stabilization actions deterministically

### Requirement: Deterministic Strict-Mode Recovery Contract
The v1 kernel SHALL keep strict-mode determinism while still allowing bounded internal numerical stabilization consistent with explicit `allow_fallback` policy.

#### Scenario: Strict mode with fallback disabled
- **WHEN** `allow_fallback=false` is configured
- **THEN** global fallback transitions remain disabled
- **AND** deterministic typed diagnostics are returned on exhaustion
- **AND** bounded internal stabilization follows strict policy limits only

### Requirement: Typed Convergence Diagnostics
The v1 kernel SHALL expose typed convergence diagnostics for each failed or recovered step.

#### Scenario: Recovered step emits structured diagnostics
- **WHEN** a step is recovered after one or more recovery actions
- **THEN** diagnostics include failure class, recovery stage, and policy action identifiers
- **AND** no text parsing is required to consume the recovery path

#### Scenario: Terminal failure emits structured diagnostics
- **WHEN** recovery budget is exhausted
- **THEN** final diagnostics include terminal reason code, last recovery class/stage, and bounded numeric context

### Requirement: Convergence Profile Contract
The v1 kernel SHALL provide explicit convergence profile semantics (`strict`, `balanced`, `robust`) with deterministic behavior boundaries.

#### Scenario: Strict profile preserves deterministic boundaries
- **WHEN** profile `strict` is selected
- **THEN** bounded internal stabilization remains within strict limits
- **AND** global fallback transitions are only permitted when explicitly enabled by policy

#### Scenario: Balanced/robust profiles remain auditable
- **WHEN** profile `balanced` or `robust` applies context-aware recovery
- **THEN** each policy transition is emitted with typed action identifiers
- **AND** resulting behavior remains reproducible under equivalent run fingerprint
