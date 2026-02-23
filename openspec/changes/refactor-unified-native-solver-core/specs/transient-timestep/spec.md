## ADDED Requirements
### Requirement: Fixed-Step Macro Grid with Internal Substeps
The transient timestep subsystem SHALL support fixed-step macro-grid execution with optional internal substeps for event alignment and convergence recovery.

#### Scenario: Internal substep during fixed-mode switching edge
- **GIVEN** fixed mode with macro timestep `dt_user`
- **WHEN** a switching boundary falls inside the current macro interval
- **THEN** the solver may insert internal substeps to land on the boundary
- **AND** committed output samples remain aligned to the macro grid

#### Scenario: Fixed-mode convergence retry
- **GIVEN** fixed mode and a failed nonlinear solve on a macro interval
- **WHEN** retry policy is triggered
- **THEN** retries use bounded internal substeps and recovery stages
- **AND** deterministic failure is reported if retry budget is exhausted

### Requirement: Variable-Step Controller Coupling LTE and Nonlinear Health
The variable-step controller SHALL combine local truncation error signals with nonlinear convergence health when selecting next timestep.

#### Scenario: Good LTE but poor Newton behavior
- **GIVEN** LTE is below target tolerance
- **WHEN** nonlinear iterations exceed configured health threshold
- **THEN** the controller reduces or caps timestep growth
- **AND** prioritizes nonlinear stability over aggressive expansion

#### Scenario: Good LTE and healthy nonlinear convergence
- **GIVEN** LTE is below tolerance and nonlinear metrics are healthy
- **WHEN** next timestep is computed
- **THEN** timestep may increase within configured growth limits

### Requirement: Event-Aware Step Clipping
Timestep selection SHALL clip candidate steps to the earliest pending event boundary before solve execution.

#### Scenario: Multiple boundary candidates
- **GIVEN** candidate events at times `t1 < t2 < t3` inside the allowed step window
- **WHEN** timestep is chosen
- **THEN** the chosen step endpoint is clipped to `t1`
- **AND** later events are handled in subsequent steps

### Requirement: Segment-Type Integrator Policy
The transient controller SHALL support segment-type integrator policies that can choose different internal integration profiles near events versus smooth regions.

#### Scenario: Event-adjacent segment
- **WHEN** the current segment is event-adjacent
- **THEN** the controller can select a stiff-safe conservative integrator profile
- **AND** applies stricter acceptance safeguards

#### Scenario: Smooth segment
- **WHEN** the current segment is classified as smooth
- **THEN** the controller can select a higher-efficiency profile under the same accuracy constraints
