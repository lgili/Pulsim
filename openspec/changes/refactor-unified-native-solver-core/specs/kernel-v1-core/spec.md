## ADDED Requirements
### Requirement: Unified Native Transient Core
The v1 kernel SHALL execute transient simulation through a single native mathematical core with shared equation assembly, nonlinear solve, linear solve, event scheduling, and recovery services.

#### Scenario: Fixed and variable modes share the same solve services
- **WHEN** two simulations are run with identical circuit topology and different timestep modes (`fixed` and `variable`)
- **THEN** both runs execute through the same residual/Jacobian assembly service
- **AND** both runs use the same nonlinear and linear service interfaces

#### Scenario: No alternate supported backend routing
- **WHEN** transient simulation is executed in the supported runtime path
- **THEN** the solver does not route through an alternate backend-specific transient engine
- **AND** telemetry identifies the native core as the selected runtime path

### Requirement: Dual-Mode User Semantics
The v1 kernel SHALL support two canonical timestep semantics: deterministic fixed-step execution and adaptive variable-step execution.

#### Scenario: Fixed-step deterministic output grid
- **WHEN** a simulation is configured in fixed mode
- **THEN** output samples are committed on the user-defined timestep grid
- **AND** internal substeps (if required) do not alter output-grid determinism

#### Scenario: Variable-step adaptive execution
- **WHEN** a simulation is configured in variable mode
- **THEN** the solver adapts timestep using error and convergence feedback
- **AND** step acceptance obeys configured accuracy constraints

### Requirement: Deterministic Recovery Ladder
The v1 kernel SHALL apply a deterministic, bounded convergence-recovery ladder for failed transient steps.

#### Scenario: Ordered escalation on failed step
- **WHEN** a transient step fails acceptance
- **THEN** the kernel applies recovery actions in configured order (dt backoff, globalization escalation, stiffness profile, transient regularization)
- **AND** records each escalation stage in fallback telemetry

#### Scenario: Retry budget exhaustion
- **WHEN** a step exceeds configured retry budget
- **THEN** the simulation terminates with a deterministic failure reason code
- **AND** includes last stage diagnostics and residual metadata

### Requirement: Event-Segmented Switched-Converter Integration
The v1 kernel SHALL segment integration intervals at switching-relevant boundaries to improve switched-converter fidelity.

#### Scenario: Earliest boundary segmentation
- **WHEN** multiple candidate boundaries exist within the current step window (PWM boundary, threshold crossing, explicit breakpoint)
- **THEN** the kernel targets the earliest boundary as the current segment end
- **AND** advances subsequent events in following segments deterministically

#### Scenario: Event timestamp refinement
- **WHEN** a threshold crossing is detected inside a segment
- **THEN** the event timestamp is refined within configured tolerance
- **AND** the event is emitted with consistent state values and transition metadata

### Requirement: Shared DC/Transient Nonlinear Services
The v1 kernel SHALL reuse nonlinear globalization and convergence-checking services across DC and transient contexts.

#### Scenario: Common convergence policy application
- **WHEN** nonlinear convergence criteria are evaluated in DC and transient solves
- **THEN** both contexts use the same weighted-error and residual policy definitions
- **AND** report comparable convergence telemetry fields

### Requirement: Legacy Transient Path Decommissioning
The v1 kernel SHALL remove supported execution dependence on legacy duplicated transient pathways.

#### Scenario: Removed legacy path request
- **WHEN** runtime configuration requests a decommissioned transient pathway
- **THEN** the kernel returns a deterministic configuration diagnostic
- **AND** includes migration guidance to supported mode-based configuration
