## ADDED Requirements
### Requirement: Datasheet-Grade Semiconductor Loss Evaluation
The v1 kernel SHALL support backend-resident semiconductor loss characterization with both scalar and datasheet-grade evaluation paths.

#### Scenario: Multidimensional switching-energy evaluation
- **GIVEN** a component configured with datasheet switching surfaces for `Eon`, `Eoff`, or `Err`
- **WHEN** a switching event is committed during transient execution
- **THEN** the kernel evaluates energy using the configured operating variables (at minimum current, blocking voltage, and junction temperature)
- **AND** the event contribution is included in per-component and aggregate loss telemetry

#### Scenario: Backward-compatible scalar loss evaluation
- **GIVEN** a component configured with scalar `eon/eoff/err` fields only
- **WHEN** switching events occur
- **THEN** the kernel uses scalar event energies
- **AND** runtime behavior remains backward compatible with existing scalar workflows

### Requirement: Switching Event Coverage for Native and Forced Semiconductor Paths
The v1 kernel SHALL account switching events for both native switching devices and externally forced semiconductor targets.

#### Scenario: PWM-forced semiconductor transition
- **GIVEN** a `pwm_generator` drives a semiconductor via target-component forcing
- **WHEN** the forced logical state toggles on an accepted step/event boundary
- **THEN** the kernel records deterministic on/off switching events for that semiconductor
- **AND** configured switching-loss models are applied

#### Scenario: Diode reverse-recovery transition
- **GIVEN** diode reverse-recovery characterization is configured
- **WHEN** conduction transitions from forward to blocking with reverse-recovery condition met
- **THEN** reverse-recovery energy is accounted in component and aggregate loss telemetry

### Requirement: Multi-Stage Electrothermal Network Integration
The v1 kernel SHALL support `single_rc`, `foster`, and `cauer` thermal-network models for thermal-enabled components.

#### Scenario: Foster network thermal update
- **GIVEN** a thermal-enabled component configured with a Foster network
- **WHEN** accepted transient segments provide dissipated power input
- **THEN** junction temperature is advanced using that network model deterministically
- **AND** emitted thermal traces reflect the configured multi-stage dynamics

#### Scenario: Cauer network thermal update
- **GIVEN** a thermal-enabled component configured with a Cauer network
- **WHEN** accepted transient segments are integrated
- **THEN** thermal state is advanced through the ladder network deterministically
- **AND** resulting temperatures remain finite under valid input ranges

### Requirement: Canonical Per-Sample Electrothermal Channel Export
The v1 kernel SHALL export canonical per-component electrothermal time-series channels aligned to the transient time base.

#### Scenario: Loss and thermal channels emitted with time alignment
- **WHEN** transient simulation runs with losses and thermal enabled
- **THEN** channel families `Pcond(<X>)`, `Psw_on(<X>)`, `Psw_off(<X>)`, `Prr(<X>)`, `Ploss(<X>)`, and `T(<X>)` are emitted for eligible components
- **AND** each channel length equals `len(result.time)`
- **AND** channel metadata includes deterministic domain, source component, quantity identity, and unit

#### Scenario: Summary consistency with channel reductions
- **WHEN** channel and summary telemetry are available in the same run
- **THEN** summary fields are deterministic reductions of corresponding channel values
- **AND** mismatch beyond numerical tolerance is reported as a deterministic runtime/test failure

### Requirement: Electrothermal Hot-Path Allocation Discipline
The v1 kernel SHALL maintain allocation-bounded hot-loop behavior when rich electrothermal modeling is enabled.

#### Scenario: Steady transient stepping with warm caches
- **WHEN** accepted steps proceed without topology/schema changes
- **THEN** electrothermal loss and thermal services avoid unplanned dynamic allocations in hot loops
- **AND** prevalidated interpolation/network structures are reused deterministically
