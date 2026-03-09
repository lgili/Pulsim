## ADDED Requirements
### Requirement: Nonlinear Magnetic Core Model Families
Device models SHALL provide canonical nonlinear magnetic-core model families for supported inductor/transformer workflows, including saturation and hysteresis-capable behavior.

#### Scenario: Supported nonlinear magnetic family executes deterministically
- **GIVEN** a supported magnetic component with a valid nonlinear core model-family configuration
- **WHEN** transient simulation runs
- **THEN** the component uses nonlinear core behavior instead of implicit linear fallback
- **AND** repeated identical runs produce deterministic magnetic state trajectories.

#### Scenario: Unsupported model family is rejected
- **GIVEN** a magnetic component configured with an unsupported nonlinear core model family
- **WHEN** parser/runtime validation runs
- **THEN** execution fails with a deterministic typed diagnostic
- **AND** no partial ambiguous magnetic result is emitted.

### Requirement: Hysteresis Memory-State Determinism
Hysteresis-capable magnetic-core models SHALL define explicit memory-state initialization and deterministic update ordering.

#### Scenario: Explicit hysteresis initialization
- **GIVEN** a hysteresis-capable model with explicit initial-state parameters
- **WHEN** simulation starts
- **THEN** internal hysteresis state is initialized deterministically from configuration
- **AND** first-cycle trajectories are reproducible across repeated runs.

#### Scenario: Deterministic loop-state update
- **GIVEN** two runs with identical inputs and timestep acceptance sequence
- **WHEN** hysteresis loop state is updated during simulation
- **THEN** internal hysteresis state evolution is identical within declared tolerance
- **AND** derived loss/state outputs remain deterministic.

### Requirement: Frequency-Dependent Core-Loss Modeling
Nonlinear magnetic-core models SHALL support deterministic frequency-dependent core-loss evaluation for supported operating regions.

#### Scenario: Frequency sweep changes core loss trend
- **GIVEN** a valid nonlinear magnetic-core model with frequency-dependent loss parameters
- **WHEN** excitation frequency changes across declared operating points
- **THEN** computed core-loss terms follow configured trend semantics deterministically
- **AND** outputs stay bounded by model policy and validation limits.

#### Scenario: Invalid loss parameterization fails fast
- **GIVEN** missing or inconsistent frequency-loss parameters
- **WHEN** strict validation or runtime preflight runs
- **THEN** configuration is rejected with deterministic diagnostics
- **AND** diagnostics include actionable parameter context.

### Requirement: Core-Loss Integration with Electrothermal Pipeline
When electrothermal coupling is enabled, magnetic core-loss contributions SHALL be integrated into per-component loss/thermal updates with summary-channel consistency.

#### Scenario: Core-loss contributes to thermal rise
- **GIVEN** nonlinear magnetic-core loss is enabled and electrothermal coupling is active
- **WHEN** simulation runs under non-zero core-loss conditions
- **THEN** core-loss power contributes to thermal evolution for the owning component
- **AND** summary reductions remain consistent with exported time-series channels.

#### Scenario: Electrothermal-disabled run keeps deterministic loss-only behavior
- **GIVEN** nonlinear magnetic-core loss is enabled and electrothermal coupling is disabled
- **WHEN** simulation runs
- **THEN** core-loss channels are still computed deterministically
- **AND** no thermal-state update is applied from magnetic losses.
