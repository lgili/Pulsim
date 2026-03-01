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

### Requirement: Advanced Linear Solver Stack
The v1 kernel SHALL provide both direct and iterative linear solvers with runtime selection and robust fallback.

#### Scenario: Large sparse circuit prefers iterative solver
- **WHEN** a circuit exceeds the configured size/nnz thresholds
- **THEN** the solver selects an iterative method (GMRES/BiCGSTAB/CG)
- **AND** applies a preconditioner (ILU0/Jacobi) if configured

#### Scenario: Iterative solve fails
- **WHEN** an iterative solve fails to converge within limits
- **THEN** the solver SHALL fall back to a direct method (KLU/Eigen SparseLU)
- **AND** record the fallback in solver telemetry

### Requirement: Nonlinear Solver Acceleration
The v1 kernel SHALL support nonlinear acceleration strategies beyond basic Newton iteration.

#### Scenario: Difficult nonlinear circuit
- **WHEN** Newton stalls or oscillates
- **THEN** the solver SHALL apply an acceleration method (Anderson or Broyden)
- **AND** may switch to Newton-Krylov with the same tolerances

#### Scenario: Aggressive steps increase residual
- **WHEN** a Newton step increases residual error
- **THEN** the solver SHALL apply line search or trust-region damping
- **AND** retry the step within configured limits

### Requirement: Solver Auto-Selection and Fallback Order
The v1 kernel SHALL allow a configurable solver selection order with deterministic fallback.

#### Scenario: User-defined solver order
- **WHEN** the configuration specifies a solver order
- **THEN** the kernel SHALL try solvers in that order
- **AND** stop at the first successful strategy

#### Scenario: Deterministic fallback
- **WHEN** multiple solvers are enabled
- **THEN** the fallback order SHALL be deterministic for reproducible results

### Requirement: Stiffness-Aware Transient Integration
The v1 kernel SHALL detect stiffness indicators and adapt integration order and timestep accordingly.

#### Scenario: Stiff switching transient
- **WHEN** stiffness is detected (e.g., repeated step rejection or large Jacobian condition changes)
- **THEN** the solver SHALL reduce timestep and/or lower BDF order
- **AND** continue with stability-focused settings until recovery

### Requirement: Solver Telemetry
The v1 kernel SHALL expose solver telemetry for debugging and regression tracking.

#### Scenario: Telemetry capture
- **WHEN** a simulation completes
- **THEN** the result SHALL include counts of nonlinear iterations, linear iterations, and fallback events
- **AND** the selected solver policies SHALL be reported in a structured form

### Requirement: Primary and Fallback Solver Order
The v1 kernel SHALL support separate primary and fallback solver orders for deterministic selection.

#### Scenario: Primary order succeeds
- **WHEN** the primary solver order succeeds
- **THEN** fallback order SHALL NOT be used

#### Scenario: Primary order fails
- **WHEN** the primary solver order fails
- **THEN** the fallback order SHALL be attempted in deterministic order

### Requirement: SPD‑Safe Conjugate Gradient
CG SHALL only be used when the linear system is symmetric positive definite (SPD).

#### Scenario: Non‑SPD matrix
- **WHEN** the matrix is not SPD
- **THEN** CG SHALL be rejected and a fallback solver SHALL be selected

### Requirement: Jacobian‑Free Newton–Krylov
The v1 kernel SHALL support JFNK with Jacobian‑vector products and iterative linear solvers.

#### Scenario: JFNK enabled
- **WHEN** JFNK is enabled
- **THEN** the solver SHALL compute J·v without assembling the full Jacobian
- **AND** use an iterative Krylov method

### Requirement: Stiff‑Stable Integrators
The v1 kernel SHALL provide TR‑BDF2 and Rosenbrock‑W/SDIRK integrators for stiff systems.

#### Scenario: TR‑BDF2 selection
- **WHEN** TR‑BDF2 is selected
- **THEN** the integrator SHALL remain stable on stiff switching transients

#### Scenario: Rosenbrock selection
- **WHEN** Rosenbrock‑W/SDIRK is selected
- **THEN** the integrator SHALL maintain stability for stiff DAEs

### Requirement: Periodic Steady‑State Solvers
The v1 kernel SHALL provide periodic steady‑state solvers for switching converters.

#### Scenario: Shooting method
- **WHEN** the shooting method is enabled
- **THEN** the solver SHALL converge to a periodic steady‑state waveform

#### Scenario: Harmonic balance
- **WHEN** harmonic balance is enabled
- **THEN** the solver SHALL compute steady‑state frequency‑domain solution

