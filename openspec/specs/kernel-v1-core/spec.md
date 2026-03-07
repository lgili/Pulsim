# kernel-v1-core Specification

## Purpose
TBD - created by archiving change unify-v1-core. Update Purpose after archive.
## Requirements
### Requirement: Single v1 Core Engine
The system SHALL use `pulsim/v1` as the sole simulation kernel for DC and transient analysis across all supported runtime entrypoints.

#### Scenario: Python runtime invokes simulation
- **WHEN** a simulation is executed through supported Python APIs
- **THEN** the execution path uses `pulsim/v1` classes and algorithms
- **AND** no alternate legacy kernel path is used

#### Scenario: Internal tooling invokes simulation
- **WHEN** benchmarks or internal helpers run simulations
- **THEN** they use the same `pulsim/v1` runtime path as Python-facing flows

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

### Requirement: Mixed-Domain Execution Scheduler

The v1 kernel SHALL support deterministic mixed-domain execution for electrical devices, behavioral control blocks, and virtual instrumentation/routing blocks.

#### Scenario: Electrical-control coupling in one timestep
- **GIVEN** a circuit containing electrical devices and control blocks
- **WHEN** an accepted timestep is processed
- **THEN** electrical solve, control update, and event-state updates execute in deterministic order
- **AND** resulting signals are consistent and reproducible across runs

### Requirement: Event-Driven Stateful Device Transitions

The v1 kernel SHALL support event-driven state transitions for latching and trip-based components (`THYRISTOR`, `TRIAC`, `FUSE`, `CIRCUIT_BREAKER`, `RELAY`).

#### Scenario: Stateful transition with event localization
- **GIVEN** a component whose state change depends on threshold crossing
- **WHEN** crossing occurs within a step
- **THEN** event localization refines transition timing
- **AND** state change is applied without non-deterministic ordering

### Requirement: Virtual Instrumentation Graph

The v1 kernel SHALL support virtual instrumentation components (`VOLTAGE_PROBE`, `CURRENT_PROBE`, `POWER_PROBE`, `ELECTRICAL_SCOPE`, `THERMAL_SCOPE`) that do not directly stamp the MNA system.

#### Scenario: Probe and scope extraction
- **GIVEN** a circuit with probes and scopes bound to electrical/thermal signals
- **WHEN** simulation runs
- **THEN** configured channel signals are captured and emitted in result data
- **AND** instrumentation components do not alter electrical matrix topology

### Requirement: Signal Routing Blocks

The v1 kernel SHALL support virtual signal routing components (`SIGNAL_MUX`, `SIGNAL_DEMUX`) for deterministic channel mapping.

#### Scenario: Mux/demux channel routing
- **GIVEN** signal routing blocks with explicit channel ordering
- **WHEN** upstream signals update
- **THEN** mux/demux outputs reflect configured mapping deterministically

### Requirement: SUNDIALS Transient Backend
The v1 kernel SHALL provide a SUNDIALS transient backend with selectable solver family support for IDA, CVODE, and ARKODE when compiled with SUNDIALS.

#### Scenario: IDA backend selected for DAE transient
- **WHEN** transient backend mode is configured to SUNDIALS with solver family `IDA`
- **THEN** the simulator SHALL execute the transient using SUNDIALS IDA callbacks
- **AND** the result SHALL include backend telemetry identifying IDA as the active solver family

#### Scenario: SUNDIALS unavailable at build time
- **WHEN** backend mode requests SUNDIALS but the binary was compiled without SUNDIALS
- **THEN** the simulator SHALL fail deterministically with an explicit backend-unavailable diagnostic
- **AND** no undefined behavior or silent fallback SHALL occur

### Requirement: Deterministic Native-to-SUNDIALS Escalation
The v1 kernel SHALL support deterministic escalation from native transient integration to SUNDIALS based on configured retry thresholds.

#### Scenario: Native retries exhausted in auto mode
- **WHEN** backend mode is `Auto` and native transient retries exceed configured threshold
- **THEN** the simulator SHALL reinitialize and continue using configured SUNDIALS solver family
- **AND** fallback trace SHALL record backend escalation with deterministic reason code and action text

#### Scenario: Auto mode succeeds without escalation
- **WHEN** native transient integration converges within configured thresholds
- **THEN** SUNDIALS SHALL NOT be invoked
- **AND** telemetry SHALL report native backend as final execution path

### Requirement: SUNDIALS Backend Telemetry
The v1 kernel SHALL expose structured SUNDIALS telemetry counters in simulation results.

#### Scenario: Successful SUNDIALS run
- **WHEN** a transient run completes with SUNDIALS backend
- **THEN** telemetry SHALL include backend name, solver family, nonlinear iteration counters, and backend recovery/reinitialization counters
- **AND** telemetry SHALL be accessible alongside existing linear/nonlinear telemetry fields

#### Scenario: Failed SUNDIALS run
- **WHEN** SUNDIALS transient execution fails
- **THEN** result status/message SHALL include mapped solver failure reason
- **AND** fallback trace SHALL include the final backend failure event

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

### Requirement: Legacy Feature Migration Gate
Legacy functionality SHALL only be removed after equivalent v1 behavior exists and is validated.

#### Scenario: Legacy-only capability identified
- **WHEN** a capability exists only in legacy code
- **THEN** it is classified as `migrate`, `drop`, or `defer` in a migration matrix
- **AND** any `migrate` item is ported and tested in v1 before legacy deletion

### Requirement: Converter Component Support Matrix
The v1 kernel SHALL maintain a declared support matrix for converter-critical components and analyses.

#### Scenario: Supported converter workflow
- **WHEN** a converter benchmark uses components listed in the declared support matrix
- **THEN** the v1 runtime executes the case without falling back to legacy implementations
- **AND** emits structured solver telemetry for the run

### Requirement: Electro-Thermal Coupled Simulation
The v1 kernel SHALL support coupled electrical and thermal simulation for declared converter workflows.

#### Scenario: Coupled run enabled
- **WHEN** electro-thermal coupling is enabled for a converter case
- **THEN** electrical losses feed thermal states during simulation
- **AND** temperature-dependent model effects are applied according to configured coupling policy

### Requirement: Stress Convergence and Determinism Envelope
The v1 kernel SHALL pass tiered stress simulations with deterministic outcomes for fixed configurations.

#### Scenario: Repeated stress execution
- **WHEN** the same stress case is run multiple times on the same hardware class with fixed settings
- **THEN** status and key deterministic metrics (step count, solver path, error metrics) remain reproducible

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

### Requirement: Hybrid Segment-First Solve Path
The v1 kernel SHALL execute switched-converter transients using an event-driven hybrid policy: state-space segment solve as primary path and shared nonlinear DAE solve as deterministic fallback.

#### Scenario: Segment model solved on primary path
- **WHEN** the current interval is classified as segment-linear under the active topology signature
- **THEN** the kernel advances state through the segment solve path without invoking nonlinear fallback
- **AND** telemetry records the segment as `state_space_primary`

#### Scenario: Deterministic fallback on non-admissible segment
- **WHEN** segment admissibility checks fail due to nonlinearity, conditioning, or policy guardrails
- **THEN** the kernel executes the shared nonlinear DAE fallback for that interval
- **AND** telemetry records the fallback reason code and topology signature

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

### Requirement: Integrated Loss and Electrothermal Commit Model
The v1 kernel SHALL integrate switching/conduction losses and thermal state updates into the accepted-step/event commit path.

#### Scenario: Switching loss commit on event transition
- **WHEN** a switching event is committed
- **THEN** the kernel accumulates switching energy for the device on that event
- **AND** rejected attempts do not contribute duplicate switching loss

#### Scenario: Electrothermal update on accepted segment
- **WHEN** a segment step is accepted
- **THEN** conduction losses are integrated for that segment and thermal RC states are advanced
- **AND** optional temperature-to-electrical parameter feedback is applied using deterministic bounded rules

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

### Requirement: Deterministic Per-Component Electrothermal Results
The v1 kernel SHALL publish deterministic electrothermal telemetry per non-virtual circuit component in transient simulation results.

#### Scenario: Component coverage and loss fields
- **WHEN** a transient run completes with loss tracking enabled
- **THEN** the result includes one entry per non-virtual component in deterministic order
- **AND** each entry includes deterministic component identity and loss fields (`conduction`, `turn_on`, `turn_off`, `reverse_recovery`, `total_loss`, `total_energy`, `average_power`, `peak_power`)
- **AND** components with zero dissipation are reported with zero-valued loss fields

#### Scenario: Thermal fields for thermal-port-enabled component
- **GIVEN** thermal coupling is enabled and a component thermal port is enabled
- **WHEN** transient simulation completes
- **THEN** the component entry includes `final_temperature`, `peak_temperature`, and `average_temperature`
- **AND** these values are derived from the same accepted-segment/event commit model used for loss accumulation

#### Scenario: Non-thermal-capable component in unified report
- **GIVEN** a component without thermal capability is part of the circuit
- **WHEN** transient simulation completes
- **THEN** the component still appears in the per-component electrothermal report
- **AND** the entry marks thermal as disabled with ambient-derived temperature values

### Requirement: Thermal Parameter Runtime Guardrails
The v1 kernel SHALL validate thermal constants for thermal-enabled components before transient stepping begins.

#### Scenario: Invalid thermal constants
- **WHEN** a thermal-enabled component has non-finite values, `rth <= 0`, or `cth < 0`
- **THEN** simulation fails with deterministic typed diagnostics
- **AND** no partial transient electrothermal report is emitted

