## ADDED Requirements
### Requirement: Frequency-Domain Analysis Modes in v1 Kernel
The v1 kernel SHALL provide first-class frequency-domain analysis modes for open-loop transfer, closed-loop transfer, and impedance sweeps.

#### Scenario: Open-loop transfer sweep
- **GIVEN** a valid circuit and frequency-analysis configuration for open-loop mode
- **WHEN** the frequency sweep executes
- **THEN** the kernel returns deterministic complex transfer response over the configured frequency grid
- **AND** results are available through structured runtime outputs

#### Scenario: Impedance sweep
- **GIVEN** a valid port definition for impedance analysis
- **WHEN** the sweep executes in input or output impedance mode
- **THEN** the kernel returns deterministic complex impedance response over the configured frequency grid
- **AND** units/quantity metadata are included in result structures

### Requirement: Deterministic Operating-Point Anchoring for AC Sweep
The v1 kernel SHALL support explicit anchoring modes (`dc`, `periodic`, `averaged`, `auto`) for frequency-domain analysis and SHALL expose the selected mode in telemetry.

#### Scenario: Auto anchoring for switching converter
- **GIVEN** a switching converter configured with anchoring mode `auto`
- **WHEN** the sweep starts
- **THEN** the kernel deterministically selects a supported anchor strategy
- **AND** selected anchor mode is reported in structured telemetry

#### Scenario: Unsupported anchor/mode combination
- **GIVEN** an analysis request that cannot be anchored by available strategies
- **WHEN** validation or execution runs
- **THEN** execution fails with a deterministic typed diagnostic
- **AND** no partial ambiguous frequency-response payload is emitted

### Requirement: Canonical Sweep Grid and Perturbation Contract
The v1 kernel SHALL execute sweeps using deterministic frequency-grid generation and perturbation rules based on configured sweep parameters.

#### Scenario: Repeat-run determinism on identical config
- **GIVEN** identical circuit, options, and machine class
- **WHEN** the same frequency sweep is executed repeatedly
- **THEN** frequency-grid points are identical
- **AND** numeric response drift remains within configured determinism tolerance

#### Scenario: Invalid sweep bounds
- **GIVEN** invalid sweep settings (for example non-positive frequency bounds or empty point count)
- **WHEN** validation is performed
- **THEN** the kernel rejects configuration with deterministic field-level diagnostics

### Requirement: Structured Frequency-Domain Result Contract
The v1 kernel SHALL provide structured AC sweep results including frequency axis, complex response data, magnitude/phase data, and derived stability metrics when defined.

#### Scenario: Derived margin metrics available
- **GIVEN** a sweep response that crosses gain/phase criteria
- **WHEN** post-processing metrics are computed
- **THEN** crossover frequency and gain/phase margins are included in result structures
- **AND** undefined metrics are marked explicitly with deterministic reason tags

#### Scenario: Result export through runtime surfaces
- **WHEN** a frequency-domain run completes successfully
- **THEN** outputs are available through backend/Python result surfaces without console-text parsing
- **AND** metadata is sufficient for frontend routing and plotting

### Requirement: Frequency-Domain Failure Taxonomy
The v1 kernel SHALL expose deterministic typed failure reasons for AC sweep execution failures.

#### Scenario: Singular linearization or unresolved response extraction
- **WHEN** numeric conditions prevent valid response extraction at one or more sweep points
- **THEN** execution returns typed deterministic diagnostics identifying failure class and context
- **AND** benchmark tooling can classify the failure without regex parsing

### Requirement: AC Sweep Allocation and Runtime Discipline
AC sweep execution SHALL maintain allocation-bounded hot-loop behavior after warm-up and SHALL expose performance telemetry for CI regression gates.

#### Scenario: Stable repeated sweep execution
- **WHEN** a sweep is repeated with unchanged topology and configuration
- **THEN** reusable analysis structures are reused deterministically
- **AND** runtime telemetry includes metrics needed for non-regression gating
