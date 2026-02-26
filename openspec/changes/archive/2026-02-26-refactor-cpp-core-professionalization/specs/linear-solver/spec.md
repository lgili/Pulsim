## ADDED Requirements
### Requirement: Signature-Keyed Linear Solver Reuse Contract
The linear solver service SHALL reuse symbolic, numeric-factorization, and preconditioner assets keyed by deterministic topology signature and solver policy identity.

#### Scenario: Reuse on unchanged signature
- **WHEN** consecutive solves execute with identical topology signature and compatible solver policy
- **THEN** reusable symbolic/factorization/preconditioner assets are reused
- **AND** telemetry records reuse hits for each cache class

#### Scenario: Incompatible policy prevents unsafe reuse
- **WHEN** solver policy or conditioning class changes in a way that invalidates reuse safety
- **THEN** incompatible assets are not reused
- **AND** a deterministic rebuild path is executed

### Requirement: Deterministic Cache Invalidation Reasons
The linear solver service SHALL expose deterministic invalidation reasons for cache rebuilds and fallback transitions.

#### Scenario: Topology-driven invalidation
- **WHEN** switching events produce a new topology signature
- **THEN** incompatible caches are invalidated with reason `topology_changed`
- **AND** rebuild telemetry includes previous and current signature identifiers

#### Scenario: Stability-driven invalidation
- **WHEN** numeric health checks detect conditioning degradation beyond configured thresholds
- **THEN** cache reuse is disabled for that solve with reason `numeric_instability`
- **AND** recovery follows configured deterministic solver fallback policy

### Requirement: Allocation-Bounded Solve Loop
Linear solve hot paths SHALL avoid unbounded dynamic allocation during steady-state reuse windows.

#### Scenario: Iterative steady-state solve sequence
- **WHEN** iterative solves run across a stable segment sequence with cache-compatible signatures
- **THEN** dynamic allocations remain within configured bounded setup/rebuild points
- **AND** no per-iteration heap growth is observed in accepted steady-state loops

### Requirement: Structured Linear Failure Reasons
Linear solver failures SHALL be reported with typed reason codes suitable for nonlinear recovery and KPI tracking.

#### Scenario: Iteration budget exhaustion
- **WHEN** iterative linear solve exceeds configured iteration budget
- **THEN** the solver reports a structured reason such as `iteration_limit`
- **AND** nonlinear recovery receives the reason code without text parsing

#### Scenario: Numerical breakdown
- **WHEN** solver encounters numerical breakdown or singularity
- **THEN** the solver reports a structured reason such as `numerical_breakdown` or `singular_matrix`
- **AND** telemetry captures the terminal solver and fallback chain position
