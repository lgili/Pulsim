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

### Requirement: Layered Core Boundary Enforcement
The v1 kernel SHALL enforce one-way dependency boundaries across core layers (`domain-model`, `equation-services`, `solve-services`, `runtime-orchestrator`, `adapters`) to reduce coupling and refactor blast radius.

#### Scenario: Forbidden cross-layer dependency
- **WHEN** a higher-risk include/import path introduces a dependency from a lower layer to a higher layer
- **THEN** boundary checks fail in CI
- **AND** the change is rejected until the dependency graph is restored

#### Scenario: Runtime orchestration stays policy-only
- **WHEN** transient execution is run in supported modes
- **THEN** orchestration coordinates services through layer contracts only
- **AND** low-level equation/solve logic remains outside orchestrator modules

### Requirement: Stable Extension Contracts
The v1 kernel SHALL provide explicit contracts and registries for devices, solvers, and integrators so new feature classes can be added without editing orchestrator internals.

#### Scenario: Add new device through registry
- **WHEN** a new device implementation satisfies the documented extension contract
- **THEN** the device is discovered/registered through the extension registry
- **AND** simulation executes without mandatory edits in runtime orchestration files

#### Scenario: Reject incompatible extension deterministically
- **WHEN** an extension violates required capabilities, metadata, or validation hooks
- **THEN** the kernel rejects registration with a deterministic structured diagnostic
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

### Requirement: Single Robustness Policy Owner
The kernel SHALL provide `RobustnessProfile` as the single source of truth for robust default configuration of Newton, linear solver, integrator, recovery, and fallback knobs.

#### Scenario: Single declaration site
- **WHEN** the codebase is grepped for "robust default" / `apply_robust_*` / `_tune_*_for_robust`
- **THEN** only one definition site (in `robustness_profile.hpp/cpp`) is found
- **AND** all callers route through this single owner

#### Scenario: Tier resolution
- **GIVEN** `RobustnessProfile::for_circuit(circuit, RobustnessTier::Aggressive)`
- **WHEN** the factory runs
- **THEN** the resulting profile has knobs derived from circuit analysis (switching count, nonlinear count) and the tier
- **AND** identical inputs produce identical profile (deterministic)

### Requirement: Robustness Profile Telemetry
`BackendTelemetry` SHALL include `robustness_profile` reflecting the resolved tier, key knob values, and a reproducibility hash.

#### Scenario: Telemetry capture
- **WHEN** a simulation completes
- **THEN** `BackendTelemetry.robustness_profile` exposes `tier`, `newton_max_iter`, `linear_solver_order`, `integrator`, `max_step_retries`, `gmin_initial`, `gmin_max`
- **AND** a hash combining these into a single identifier appears

#### Scenario: Profile diff in verbose mode
- **GIVEN** verbosity-enabled output and a non-default profile
- **WHEN** the result message is composed
- **THEN** the diff vs the default profile is included as a structured list

### Requirement: runtime_circuit.hpp Implementation Split
The `core/include/pulsim/v1/runtime_circuit.hpp` header SHALL be split such that method bodies move to a corresponding `.cpp`, with explicit template instantiation where applicable.

#### Scenario: Header trimmed
- **WHEN** the project is built
- **THEN** `runtime_circuit.hpp` contains declarations and public templates only
- **AND** method bodies live in `core/src/v1/runtime_circuit.cpp`
- **AND** the header is below 1000 lines

#### Scenario: Explicit instantiation
- **GIVEN** any client TU that includes `runtime_circuit.hpp`
- **WHEN** the TU is compiled
- **THEN** `extern template` declarations prevent re-instantiation of common specializations
- **AND** total client compile time is reduced ≥40% on a representative file

### Requirement: Bloated Header Audit
Headers exceeding 1500 lines (`high_performance.hpp`, `integration.hpp`) SHALL be audited and trimmed by moving inline implementations to `.cpp`, removing dead code, or splitting by concern.

#### Scenario: Header below threshold post-trim
- **WHEN** the audit completes
- **THEN** no kernel header in `core/include/pulsim/v1/` exceeds 1500 lines
- **AND** dead utilities removed from headers are documented in the change history

### Requirement: Build-Time Regression Alerts
CI SHALL track clean-build wallclock per platform and alert when growth exceeds 10% across two consecutive runs without an associated justification.

#### Scenario: Build-time regression
- **WHEN** a PR causes clean-build wallclock to grow >10% on any platform
- **THEN** the CI emits an alert linking the PR
- **AND** the PR description must include justification or a fix

### Requirement: Three-Phase Transform Control Blocks
The runtime SHALL provide virtual control blocks for Clarke and Park transformations and their inverses, plus a phase-locked-loop block that produces a synchronization angle θ.

#### Scenario: Clarke transform of a balanced three-phase sine source
- **GIVEN** three sine voltage sources at 0°, 120°, 240° with equal amplitude V_pk
- **WHEN** a `clarke_transform` block is connected to the three source nodes [a, b, c]
- **THEN** the block emits channel values `<name>.alpha`, `<name>.beta`, `<name>.gamma`
- **AND** `alpha` is a 60 Hz sine of amplitude V_pk (amplitude-invariant convention)
- **AND** `beta` is the 90°-shifted sine of the same amplitude
- **AND** `gamma` is approximately zero for a balanced source

#### Scenario: Park transform takes θ from a channel
- **GIVEN** a `park_transform` block with metadata `theta_from_channel: PLL.theta`
- **WHEN** the simulation runs and the PLL has produced a value `PLL.theta` for the current step
- **THEN** the Park block reads θ from `virtual_signal_state_["PLL.theta"]`
- **AND** emits `<name>.d` and `<name>.q` according to the standard Park rotation matrix

### Requirement: PLL Block
The runtime SHALL provide a `pll` virtual block that locks to a single-phase sinusoidal input via a PI loop on the q-axis projection and emits the recovered phase angle.

#### Scenario: PLL locks to a 60 Hz sine
- **GIVEN** a `pll` block with `kp` and `ki` configured for 60 Hz nominal
- **WHEN** the input is a 60 Hz sine
- **THEN** the PLL's `theta` channel converges to the input's actual phase (within ±1° after settling)
- **AND** the `lock_error` channel drops below 0.01 V_pk in steady state

### Requirement: Space-Vector Modulation Block
The runtime SHALL provide an `svm` virtual block that takes a stationary-frame reference (α, β) plus a DC bus voltage and emits three half-bridge duties.

#### Scenario: SVM produces sinusoidal duties from rotating reference
- **GIVEN** an `svm` block with `alpha_from_channel` / `beta_from_channel` referencing a rotating reference vector at 60 Hz
- **WHEN** the simulation runs for several fundamental periods
- **THEN** each of `<name>.d_a`, `<name>.d_b`, `<name>.d_c` is a sinusoid at 60 Hz with the appropriate 120°-shifted phase
- **AND** the duties are clamped to [0, 1]

### Requirement: Circuit Component Introspection
The Circuit type SHALL expose a deterministic enumeration of every physical component added to it, returning each component's name, canonical kind, terminal node indices in pin order, and a parameter map.

#### Scenario: Enumerate components on a code-built circuit
- **WHEN** code calls `circuit.components()` after invoking `add_resistor`, `add_capacitor`, and `add_voltage_source`
- **THEN** the returned sequence contains one descriptor per `add_*` call, in insertion order
- **AND** each descriptor exposes `name`, `kind`, `nodes`, and `params`

#### Scenario: Enumerate components on a YAML-loaded circuit
- **WHEN** code loads a netlist via `YamlParser.load(...)` and calls `components()`
- **THEN** the descriptor list mirrors the YAML `components:` array in order
- **AND** node IDs match the integer IDs resolved by the parser

#### Scenario: Determinism across runs
- **WHEN** the same circuit is built twice with identical `add_*` calls
- **THEN** both `components()` sequences are byte-identical (same names, kinds, nodes, params)

#### Scenario: Read-only accessor
- **WHEN** consumer code retains a reference to the components sequence
- **THEN** subsequent `add_*` calls do not mutate previously returned descriptors

### Requirement: Node Position Hint
The Circuit type SHALL expose a node-position-hint helper that classifies each node into a role used by downstream layout consumers.

#### Scenario: Ground node classified as ground
- **WHEN** a consumer queries the role of `circuit.ground()`
- **THEN** the helper returns the `ground` role

#### Scenario: Voltage-source-fed node classified as source positive
- **WHEN** a node is the positive terminal of a voltage source
- **THEN** the helper returns the `source_pos` role

#### Scenario: Resistive load node classified as load
- **WHEN** a node is connected to a resistor whose other terminal is ground
- **THEN** the helper returns the `load` role

#### Scenario: Internal node has no specific role
- **WHEN** a node has no role-defining connection
- **THEN** the helper returns the `internal` role

