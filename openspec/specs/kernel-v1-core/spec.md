# kernel-v1-core Specification

## Purpose
TBD - created by archiving change unify-v1-core. Update Purpose after archive.
## Requirements
### Requirement: Single v1 Core Engine
The system SHALL use `pulsim/v1` as the sole simulation kernel for DC and transient analysis.

#### Scenario: Python or CLI invokes simulation
- **WHEN** a simulation is executed via Python or CLI
- **THEN** the execution path uses `pulsim/v1` classes and algorithms

### Requirement: Robust DC Operating Point
The system SHALL compute DC operating points using convergence aids (Gmin, source stepping, pseudo-transient) with a configurable strategy order.

#### Scenario: Nonlinear converter with difficult DC
- **WHEN** Newton fails with direct solve
- **THEN** the solver attempts Gmin, source stepping, and pseudo-transient in order until convergence or exhaustion

### Requirement: Adaptive Transient Simulation
The system SHALL support adaptive timesteps using LTE estimation and PI control, with BDF order control when enabled.

#### Scenario: Switching transient at high frequency
- **WHEN** LTE exceeds tolerance or Newton fails
- **THEN** the timestep is reduced and the step is retried

### Requirement: Event Handling for Switches
The system SHALL detect switch events and refine event times via bisection to record accurate transitions.

#### Scenario: Switch threshold crossing
- **WHEN** a control waveform crosses the threshold within a step
- **THEN** the simulator bisects the interval to locate the event time

### Requirement: Loss Accumulation
The system SHALL compute conduction and switching losses and expose per-device loss summaries.

#### Scenario: MOSFET switching
- **WHEN** a MOSFET turns on or off
- **THEN** the switching loss is accumulated for that device and included in the result

