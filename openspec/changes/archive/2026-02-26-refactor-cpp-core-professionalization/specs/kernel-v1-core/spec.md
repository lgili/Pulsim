## ADDED Requirements
### Requirement: Layered Core Boundary Enforcement
The v1 kernel SHALL enforce one-way dependency boundaries across core layers (`domain-model`, `equation-services`, `solve-services`, `runtime-orchestrator`, `adapters`) to reduce coupling and refactor blast radius.

#### Scenario: Forbidden cross-layer dependency
- **WHEN** a higher-risk include/import path introduces a dependency from a lower layer to a higher layer
- **THEN** boundary checks fail in CI
- **AND** the change is rejected until the dependency graph is restored

#### Scenario: Runtime orchestration stays policy-only
- **WHEN** transient execution is run in supported modes
- **THEN** orchestration coordinates services through layer contracts only
- **AND** low-level equation/solve logic remains outside orchestrator modules

### Requirement: Stable Extension Contracts
The v1 kernel SHALL provide explicit contracts and registries for devices, solvers, and integrators so new feature classes can be added without editing orchestrator internals.

#### Scenario: Add new device through registry
- **WHEN** a new device implementation satisfies the documented extension contract
- **THEN** the device is discovered/registered through the extension registry
- **AND** simulation executes without mandatory edits in runtime orchestration files

#### Scenario: Reject incompatible extension deterministically
- **WHEN** an extension violates required capabilities, metadata, or validation hooks
- **THEN** the kernel rejects registration with a deterministic structured diagnostic
- **AND** partial registration side effects are rolled back

### Requirement: Deterministic Failure Taxonomy and Boundary Guards
The v1 kernel SHALL standardize failure reason taxonomy and enforce finite-value, bounds, and dimensional guards at service boundaries.

#### Scenario: Non-finite value at service boundary
- **WHEN** NaN/Inf or invalid dimensional input reaches a protected boundary
- **THEN** the solve is aborted with a typed deterministic failure reason
- **AND** diagnostics include the failing subsystem and guard category

#### Scenario: Hard nonlinear failure containment
- **WHEN** retry/recovery budgets are exhausted in transient or DC contexts
- **THEN** the kernel returns a deterministic terminal failure code
- **AND** emits final residual and recovery-stage telemetry without crashing

### Requirement: Hot-Path Allocation Discipline
The v1 kernel SHALL enforce allocation-bounded steady-state stepping in hot loops, with deterministic cache reuse/invalidation across topology transitions.

#### Scenario: Stable topology steady-state stepping
- **WHEN** repeated accepted steps run under unchanged topology signature
- **THEN** the hot stepping path performs no unplanned dynamic allocations
- **AND** reusable solver/integration caches are reused

#### Scenario: Topology transition cache invalidation
- **WHEN** a switch/event changes topology signature
- **THEN** incompatible cache entries are invalidated deterministically before next solve
- **AND** new cache state is rebuilt under the active signature

### Requirement: Core Safety Tooling Gates
Core module changes SHALL pass sanitizer and static-analysis gates before merge.

#### Scenario: Changed core module in pull request
- **WHEN** a pull request modifies kernel core files in managed modules
- **THEN** ASan/UBSan and configured static-analysis jobs are executed
- **AND** merge is blocked on findings above configured severity thresholds

### Requirement: Modern C++ Interface Safety Contracts
Core service interfaces SHALL use modern C++ non-owning views and constrained extension contracts where applicable.

#### Scenario: Non-owning hot-path interfaces
- **WHEN** a core service exposes read-only sequence/string inputs in hot paths
- **THEN** interfaces use non-owning views (for example span-like/string-view semantics)
- **AND** avoid unnecessary ownership transfer or deep copies

#### Scenario: Constrained extension templates
- **WHEN** extension integration uses template-based contracts
- **THEN** compile-time constraints validate required operations/capabilities
- **AND** incompatible implementations fail with deterministic compile-time diagnostics
