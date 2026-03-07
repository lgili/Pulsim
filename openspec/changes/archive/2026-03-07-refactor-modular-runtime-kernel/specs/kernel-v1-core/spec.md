## ADDED Requirements
### Requirement: Runtime Module Lifecycle Contracts
The v1 kernel SHALL execute transient runtime concerns through explicit module lifecycle contracts so each concern can evolve independently without mandatory edits in central orchestrator logic.

#### Scenario: Deterministic lifecycle execution order
- **GIVEN** a simulation run with multiple active runtime modules
- **WHEN** transient execution starts and steps are processed
- **THEN** modules are invoked through deterministic lifecycle hooks in resolved dependency order
- **AND** repeated runs with identical inputs produce the same module invocation order

#### Scenario: Isolated module evolution
- **GIVEN** one runtime module implementation changes
- **WHEN** integration tests and benchmarks are executed
- **THEN** unrelated modules do not require structural edits
- **AND** regressions are localized to module-specific tests or declared integration boundaries

### Requirement: Deterministic Module Dependency Resolution
The v1 kernel SHALL resolve module dependencies/capabilities deterministically at run initialization and reject incompatible module graphs with typed diagnostics.

#### Scenario: Missing required capability
- **GIVEN** an enabled module declares a required capability not provided by any active module
- **WHEN** run initialization validates module dependencies
- **THEN** the run fails fast before stepping
- **AND** emits deterministic diagnostics identifying the missing capability and module

#### Scenario: Cyclic dependency rejection
- **GIVEN** module declarations form a dependency cycle
- **WHEN** module dependency resolution executes
- **THEN** the cycle is rejected deterministically
- **AND** the diagnostic includes the conflicting module set

### Requirement: Module-Owned Channel and Telemetry Registration
The v1 kernel SHALL require modules that emit channels or telemetry to register ownership and metadata through a shared module-output contract.

#### Scenario: Module channel registration
- **GIVEN** an active module that emits virtual channels
- **WHEN** channel registration is performed
- **THEN** channel names and metadata are declared before steady-state sampling
- **AND** ownership is traceable to the emitting module

#### Scenario: Summary reduction consistency under module ownership
- **GIVEN** modules emit canonical thermal/loss channels
- **WHEN** summaries are finalized
- **THEN** summary values remain deterministic reductions of module-emitted channels
- **AND** consistency checks fail with typed diagnostics on mismatch

## MODIFIED Requirements
### Requirement: Layered Core Boundary Enforcement
The v1 kernel SHALL enforce one-way dependency boundaries across core layers (`domain-model`, `equation-services`, `solve-services`, `runtime-modules`, `runtime-orchestrator`, `adapters`) to reduce coupling and refactor blast radius.

#### Scenario: Forbidden cross-layer dependency
- **WHEN** a dependency is introduced from a lower layer to a higher layer
- **THEN** boundary checks fail in CI
- **AND** the change is rejected until dependency direction is restored

#### Scenario: Runtime orchestration stays policy-only
- **WHEN** transient execution is run in supported modes
- **THEN** orchestration coordinates execution through runtime module and service contracts only
- **AND** module-internal physics/analysis logic remains outside orchestrator units

### Requirement: Stable Extension Contracts
The v1 kernel SHALL provide explicit contracts and registries for devices, solvers, integrators, and runtime modules so new feature classes can be added without editing orchestrator internals.

#### Scenario: Add extension through registry contract
- **WHEN** a new extension satisfies the documented contract and metadata requirements
- **THEN** it is discoverable/registered through extension registries
- **AND** simulation executes without mandatory edits in central orchestration files

#### Scenario: Reject incompatible extension deterministically
- **WHEN** an extension violates capabilities, metadata, or validation hooks
- **THEN** registration is rejected with deterministic structured diagnostics
- **AND** partial registration side effects are rolled back
