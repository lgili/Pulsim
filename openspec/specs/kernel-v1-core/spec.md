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

### Requirement: Canonical Waveform Post-Processing Pipeline in v1 Kernel
The v1 kernel SHALL provide a backend-owned waveform post-processing pipeline that operates on canonical simulation/frequency result surfaces.

#### Scenario: Deterministic post-processing execution
- **GIVEN** valid post-processing job definitions and resolved source signals
- **WHEN** post-processing executes
- **THEN** the kernel computes requested metrics deterministically
- **AND** output ordering and key naming are stable for equivalent inputs.

#### Scenario: Unsupported source contract
- **WHEN** a post-processing job references unsupported or missing source data
- **THEN** execution fails with typed deterministic diagnostics
- **AND** no partial ambiguous result object is emitted for that failed job.

### Requirement: Deterministic Windowing and Sampling Semantics
The v1 kernel SHALL enforce explicit deterministic semantics for analysis windows and sampling prerequisites.

#### Scenario: Valid window specification
- **WHEN** a job specifies a valid window (time/index/cycle mode) with sufficient samples
- **THEN** the kernel resolves deterministic sample bounds
- **AND** all derived metrics use those exact bounds.

#### Scenario: Invalid or insufficient window
- **WHEN** a job window is invalid, empty, or undersampled for requested analysis
- **THEN** the job fails with typed diagnostics (`invalid_window` or `insufficient_samples` equivalent)
- **AND** failure reason includes deterministic context fields.

### Requirement: Spectral and Harmonic Metric Engine
The v1 kernel SHALL support FFT/harmonic analysis and THD computation with explicit configuration and deterministic behavior.

#### Scenario: FFT/THD on known harmonic waveform
- **GIVEN** a waveform with known harmonic composition
- **WHEN** FFT and THD jobs execute
- **THEN** fundamental, harmonic bins, and THD are computed within configured tolerances
- **AND** spectral outputs include deterministic frequency-axis metadata.

#### Scenario: Undefined THD reason code
- **WHEN** THD cannot be defined due to zero/undefined fundamental component
- **THEN** the kernel emits deterministic undefined-metric reason codes
- **AND** does not silently report arbitrary fallback values.

### Requirement: Time-Domain and Power Metric Engine
The v1 kernel SHALL provide deterministic time-domain and power metrics for post-processing workflows.

#### Scenario: Time-domain metric set
- **WHEN** a job requests metrics such as RMS, mean, min/max, peak-to-peak, crest factor, or ripple
- **THEN** kernel outputs canonical metric keys and values with units
- **AND** metric definitions remain stable across releases unless explicitly versioned.

#### Scenario: Efficiency and power-factor metric set
- **WHEN** input/output power signal bindings are valid
- **THEN** kernel computes average power, derived efficiency, and power-factor metrics deterministically
- **AND** invalid bindings fail with typed diagnostics.

### Requirement: Post-Processing Runtime Discipline
Post-processing execution SHALL maintain allocation-aware behavior and expose telemetry for non-regression gating.

#### Scenario: Repeated post-processing with equivalent jobs
- **WHEN** equivalent post-processing jobs are executed repeatedly on the same machine class
- **THEN** runtime telemetry indicates stable execution cost and deterministic outputs
- **AND** regressions beyond configured KPI thresholds fail CI gates.

### Requirement: C_BLOCK Type in SignalEvaluator
The `SignalEvaluator` SHALL recognise `"C_BLOCK"` as a valid signal-domain
component type, include it in `SIGNAL_TYPES`, and evaluate it in topological order
each timestep.  During `build()`, the evaluator MUST initialise a `CBlockLibrary`
or `PythonCBlock` controller for each `C_BLOCK` component and store it in the
internal controller registry.  During `step(t)`, the evaluator MUST call
`ctl.step(t, dt, inputs)` where `dt` is the elapsed time since the previous
accepted step.  The evaluator MUST call `ctl.reset()` for all C-Block controllers
in its `reset()` method.

#### Scenario: C_BLOCK correctly transforms signal in a simple chain
- **GIVEN** a `SignalEvaluator` built from a circuit dict with:
  `CONSTANT(value=4.0) → C_BLOCK(gain×2) → PWM_GENERATOR`
- **WHEN** `evaluator.step(t=1e-3)` is called
- **THEN** the PWM component receives duty `8.0` (clamped to `1.0` by the evaluator)
- **AND** no exceptions are raised

#### Scenario: `dt` passed to C-Block reflects elapsed accepted timestep
- **GIVEN** a `PythonCBlock` that records the `dt` argument it receives
- **AND** an evaluator whose `step` is called at `t=0`, `t=1e-6`, `t=3e-6`
- **WHEN** the three calls are made
- **THEN** `dt` values recorded are `0.0`, `1e-6`, `2e-6` respectively

#### Scenario: `build()` fails with AlgebraicLoopError for self-referencing C_BLOCK
- **GIVEN** a circuit dict where the single output of a `C_BLOCK` feeds back into
  its only input with no delay
- **WHEN** `evaluator.build()` is called
- **THEN** `AlgebraicLoopError` is raised
- **AND** the C_BLOCK's id appears in `exc.cycle_ids`

#### Scenario: `reset()` reinitialises all C-Block controllers
- **GIVEN** a `SignalEvaluator` with two `C_BLOCK` components, both stateful
- **WHEN** `evaluator.reset()` is called after 50 steps
- **THEN** both C-Block controllers have their state reset
- **AND** the next `step()` call produces the same output as the very first step

### Requirement: C_BLOCK Compile-at-Build Support in SignalEvaluator
The `SignalEvaluator` MUST support automatic compilation of C-Block source files.
When a `C_BLOCK` component specifies `source` (a path to a `.c` file) instead of
`lib_path`, the evaluator MUST call `compile_cblock(source, ...)` during `build()`
and load the resulting shared library.  If compilation fails, `build()` SHALL raise
`CBlockCompileError` immediately.  The compiled library path MUST be cached and
SHALL NOT be recompiled on subsequent `build()` calls unless the source modification
time has changed.

#### Scenario: Source-based C_BLOCK compiles and runs during build
- **GIVEN** a circuit dict with `C_BLOCK` specifying `source: "gain.c"` and the
  file exists with valid C code; and a C compiler is available
- **WHEN** `evaluator.build()` is called
- **THEN** `gain.c` is compiled automatically
- **AND** `evaluator.step(t)` executes the compiled block without errors

#### Scenario: Source-based C_BLOCK compilation failure surfaces in build
- **GIVEN** a circuit dict with `C_BLOCK` specifying `source: "broken.c"` that
  has a syntax error
- **WHEN** `evaluator.build()` is called
- **THEN** `CBlockCompileError` is raised with the compiler stderr

#### Scenario: Source compilation is skipped if cached library is fresh
- **GIVEN** a `C_BLOCK` with a previously compiled library that is newer than the
  source file
- **WHEN** `evaluator.build()` is called a second time
- **THEN** the compiler subprocess is NOT invoked (cache hit)

### Requirement: C_BLOCK Multi-Output Vector and SIGNAL_DEMUX Integration
When a `C_BLOCK` has `n_outputs > 1`, the evaluator MUST store the full output
`list[float]` as the component state.  A downstream `SIGNAL_DEMUX` wired from
the C_BLOCK's output MUST receive the vector and distribute individual scalar
values by pin index.  This MUST happen in a single evaluation round (no extra DAG
passes).

#### Scenario: 3-output C_BLOCK distributes values via SIGNAL_DEMUX
- **GIVEN** a `C_BLOCK` with `n_outputs=3` returning `[1.1, 2.2, 3.3]`
  wired to a `SIGNAL_DEMUX` with pins `OUT1`, `OUT2`, `OUT3`
- **WHEN** the evaluator runs one step
- **THEN** the demux output pins carry values `1.1`, `2.2`, `3.3` respectively
- **AND** downstream consumers of `OUT2` see exactly `2.2`

### Requirement: Convergence Policy Engine
The v1 kernel SHALL provide a convergence policy engine that classifies transient failures and selects context-aware recovery actions instead of relying only on retry ordinals.

#### Scenario: Failure classified as event-burst zero-cross
- **WHEN** repeated failures occur near dense switching boundaries around zero crossing
- **THEN** the failure class is set to `event_burst_zero_cross`
- **AND** recovery policy applies event-aware backoff/guard actions instead of generic retry-only behavior

#### Scenario: Failure classified as control-discrete stiffness
- **WHEN** convergence degradation correlates with discrete control update boundaries
- **THEN** the failure class is set to `control_discrete_stiffness`
- **AND** policy applies control-aware stabilization actions deterministically

### Requirement: Deterministic Strict-Mode Recovery Contract
The v1 kernel SHALL keep strict-mode determinism while still allowing bounded internal numerical stabilization consistent with explicit `allow_fallback` policy.

#### Scenario: Strict mode with fallback disabled
- **WHEN** `allow_fallback=false` is configured
- **THEN** global fallback transitions remain disabled
- **AND** deterministic typed diagnostics are returned on exhaustion
- **AND** bounded internal stabilization follows strict policy limits only

### Requirement: Typed Convergence Diagnostics
The v1 kernel SHALL expose typed convergence diagnostics for each failed or recovered step.

#### Scenario: Recovered step emits structured diagnostics
- **WHEN** a step is recovered after one or more recovery actions
- **THEN** diagnostics include failure class, recovery stage, and policy action identifiers
- **AND** no text parsing is required to consume the recovery path

#### Scenario: Terminal failure emits structured diagnostics
- **WHEN** recovery budget is exhausted
- **THEN** final diagnostics include terminal reason code, last recovery class/stage, and bounded numeric context

### Requirement: Convergence Profile Contract
The v1 kernel SHALL provide explicit convergence profile semantics (`strict`, `balanced`, `robust`) with deterministic behavior boundaries.

#### Scenario: Strict profile preserves deterministic boundaries
- **WHEN** profile `strict` is selected
- **THEN** bounded internal stabilization remains within strict limits
- **AND** global fallback transitions are only permitted when explicitly enabled by policy

#### Scenario: Balanced/robust profiles remain auditable
- **WHEN** profile `balanced` or `robust` applies context-aware recovery
- **THEN** each policy transition is emitted with typed action identifiers
- **AND** resulting behavior remains reproducible under equivalent run fingerprint

