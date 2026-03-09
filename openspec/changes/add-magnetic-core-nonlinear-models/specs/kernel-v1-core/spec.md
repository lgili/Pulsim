## ADDED Requirements
### Requirement: Nonlinear Magnetic-Core Runtime Execution
The v1 kernel SHALL execute nonlinear magnetic-core state updates for supported magnetic components with deterministic accepted-step semantics.

#### Scenario: Nonlinear magnetic state updates on accepted steps
- **GIVEN** a supported component configured with nonlinear magnetic-core model
- **WHEN** transient simulation accepts integration steps
- **THEN** magnetic-core internal state is updated only on accepted steps
- **AND** rejected steps do not commit persistent magnetic state changes.

#### Scenario: Unsupported runtime configuration fails deterministically
- **GIVEN** a magnetic-core runtime configuration that violates supported contracts
- **WHEN** simulation preflight/execution runs
- **THEN** the kernel fails with typed deterministic diagnostics
- **AND** failure class is machine-readable for tooling.

### Requirement: Canonical Magnetic-Core Telemetry Channels
The v1 kernel SHALL expose canonical magnetic-core state/loss channels and metadata for supported workflows.

#### Scenario: Magnetic channels include metadata
- **GIVEN** a simulation with nonlinear magnetic-core telemetry enabled
- **WHEN** results are exported
- **THEN** magnetic channels are present with deterministic ordering
- **AND** each channel includes structured metadata (domain, unit, source component, quantity).

#### Scenario: Summary consistency with channels
- **GIVEN** exported magnetic-core channels and summary fields
- **WHEN** summary reductions are computed
- **THEN** summary values are consistent with channel reductions within declared tolerance
- **AND** inconsistency triggers deterministic diagnostics.

### Requirement: Core-Loss Coupling into Loss and Thermal Surfaces
The v1 kernel SHALL integrate magnetic core-loss terms into canonical loss and electrothermal surfaces when corresponding options are enabled.

#### Scenario: Loss-enabled run exports core-loss contribution
- **GIVEN** magnetic core-loss modeling and global loss pipeline are enabled
- **WHEN** simulation runs
- **THEN** core-loss contribution is reflected in per-component loss surfaces and summaries
- **AND** contribution accounting is deterministic.

#### Scenario: Thermal-enabled run includes core-loss heating contribution
- **GIVEN** magnetic core-loss and electrothermal coupling are enabled
- **WHEN** simulation runs
- **THEN** thermal evolution includes magnetic core-loss heating contribution
- **AND** thermal summary/channel consistency is preserved.
