## ADDED Requirements
### Requirement: Unified Runtime Linear Solver Service
The system SHALL provide a single runtime linear solver service used by all transient timestep modes and nonlinear contexts.

#### Scenario: Shared service across fixed and variable modes
- **WHEN** fixed and variable mode simulations execute on the same circuit class
- **THEN** both modes invoke the same linear solver service interfaces
- **AND** solver selection policy is applied consistently

### Requirement: Deterministic Ordered Solver Selection
The linear solver service SHALL apply deterministic primary and fallback solver ordering.

#### Scenario: Primary failure triggers fallback
- **WHEN** the selected primary solver fails numerical acceptance or iteration limits
- **THEN** fallback solvers are attempted in configured deterministic order
- **AND** telemetry records the transition and terminal solver choice

### Requirement: Topology-Signature Reuse
The linear solver service SHALL support symbolic/pattern reuse keyed by topology signature to reduce repeated setup work.

#### Scenario: Unchanged topology across steps
- **WHEN** consecutive steps have identical topology signature
- **THEN** symbolic analysis metadata is reused
- **AND** full reanalysis is skipped unless numeric safety checks fail

#### Scenario: Topology change invalidates cache
- **WHEN** switch events change topology signature
- **THEN** incompatible cached symbolic data is invalidated deterministically
- **AND** a new compatible cache entry is generated

### Requirement: Preconditioner Lifecycle Control
Iterative solver preconditioners SHALL follow deterministic lifecycle rules for creation, reuse, and invalidation.

#### Scenario: Iterative reuse in stable segment
- **WHEN** iterative solving continues in a stable segment with compatible signature
- **THEN** the preconditioner may be reused
- **AND** reuse is reported via telemetry

#### Scenario: Preconditioner invalidation on instability
- **WHEN** convergence degradation exceeds configured thresholds
- **THEN** the preconditioner is rebuilt or solver falls back according to policy
- **AND** the reason is emitted in telemetry
