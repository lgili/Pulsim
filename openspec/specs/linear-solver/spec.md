# linear-solver Specification

## Purpose
TBD - created by archiving change improve-convergence-algorithms. Update Purpose after archive.
## Requirements
### Requirement: AdvancedLinearSolver Enhancement

The existing AdvancedLinearSolver SHALL be enhanced with the new optimization features.

#### Scenario: AdvancedLinearSolver with KLU

- **GIVEN** AdvancedLinearSolver configured with Backend::Auto
- **WHEN** KLU is available
- **THEN** KLU is used for all solves
- **AND** symbolic caching is enabled by default

#### Scenario: Backward compatibility

- **GIVEN** existing code using AdvancedLinearSolver
- **WHEN** no options are specified
- **THEN** behavior is compatible with previous version
- **AND** new optimizations are applied transparently

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

### Requirement: Health-Driven Linear Solver Policy
Linear solver selection SHALL consider numeric health signals in addition to size/nnz heuristics.

#### Scenario: Conditioning degradation in active solver
- **WHEN** conditioning and residual indicators cross degradation thresholds
- **THEN** solver policy escalates deterministically to a safer candidate solver/preconditioner
- **AND** emits structured policy-transition telemetry

#### Scenario: Healthy regime retains fast solver
- **WHEN** health indicators remain within configured safe bounds
- **THEN** active solver and cache reuse remain on the efficient path
- **AND** no unnecessary solver churn occurs

### Requirement: Structured Linear Failure Taxonomy for Policy Engine
Linear failures SHALL expose typed reason classes consumable by the convergence policy engine.

#### Scenario: Iterative breakdown classification
- **WHEN** iterative solve fails due to breakdown or stagnation
- **THEN** failure reason class is exported in structured form
- **AND** convergence policy can branch recovery by class without parsing error strings

