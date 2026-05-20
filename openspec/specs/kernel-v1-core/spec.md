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
The system SHALL compute DC operating points using convergence aids
(Direct, source stepping, Gmin stepping, pseudo-transient, and
homotopy continuation) with a configurable strategy order.

Homotopy continuation SHALL be the fifth and final fallback in the
`DCStrategy::Auto` ladder, invoked only when the first four
strategies have all failed. The homotopy ladder SHALL step a
parameter λ from 0 (all nonlinear devices replaced by their linear
off-state conductance) to 1 (full nonlinear behaviour) in
configurable increments (5 by default, 10 in `Preset::HighFidelity`).

#### Scenario: Nonlinear converter with difficult DC

- **WHEN** Newton fails with direct solve
- **THEN** the solver attempts source stepping, Gmin, pseudo-transient,
  and homotopy in order until convergence or exhaustion

#### Scenario: Cold-start multilevel NPC requires homotopy

- **GIVEN** a 3-level NPC at cold start where the first four
  strategies all fail to converge
- **WHEN** `DCStrategy::Auto` reaches the homotopy step
- **THEN** the homotopy ladder runs from λ=0 to λ=1 in 5 increments
- **AND** the DC operating point is established
- **AND** `homotopy_ladder_completed` is reported true in telemetry

#### Scenario: Simple RC bypasses homotopy

- **GIVEN** a passive RC circuit where Direct converges immediately
- **WHEN** `DCStrategy::Auto` runs
- **THEN** homotopy is never invoked
- **AND** `homotopy_steps == 0` is reported in telemetry

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

### Requirement: Numerical Preset Profiles

The system SHALL provide a `Preset` enumeration with four named
profiles — `Auto`, `Fast`, `Robust`, `HighFidelity` — and a factory
method `SimulationOptions::from_preset(Preset, dt, tstop)` that
materialises a fully-tuned `SimulationOptions` instance from a
single named choice.

Each preset SHALL select a coherent combination of integrator,
linear solver, timestep controller, DC strategy, stiffness policy,
and Newton tuning. Users SHALL NOT be required to set any additional
field beyond the preset, `dt`, and `tstop` to obtain a valid
simulation configuration for the preset's target use case.

#### Scenario: Preset materialises a complete configuration

- **GIVEN** a user calls `SimulationOptions::from_preset(Preset::Auto,
  1e-6, 1e-3)`
- **WHEN** the returned options are inspected
- **THEN** every numerical sub-config is populated with the Auto
  profile's defaults
- **AND** the simulation can be executed without setting any other
  field

#### Scenario: Preset.Fast targets pure-switching topologies

- **GIVEN** the user picks `Preset::Fast` for a buck converter
- **WHEN** the resulting options are used in `Simulator::run_transient`
- **THEN** the PWL Ideal switching path is enabled
- **AND** the integrator is `Trapezoidal`
- **AND** the timestep mode is `Fixed`
- **AND** the linear solver auto-selects to a direct (KLU-equivalent)
  path

#### Scenario: Preset.Robust targets motor-drive / mixed-domain circuits

- **GIVEN** the user picks `Preset::Robust`
- **WHEN** the resulting options are materialised
- **THEN** the integrator is `TRBDF2`
- **AND** stiffness detection + automatic integrator switch-over is
  enabled
- **AND** at least 12 step-retries are allowed before failing
- **AND** the DC strategy is `Auto` with the full fallback ladder

#### Scenario: Preset.HighFidelity targets parity-validation runs

- **GIVEN** the user picks `Preset::HighFidelity`
- **WHEN** the resulting options are materialised
- **THEN** the LTE tolerance is at least 10x tighter than `Preset::Robust`
- **AND** step-doubling LTE estimation is used (not Richardson)
- **AND** the maximum timestep is at least 10x smaller than the
  caller-supplied `dt`

#### Scenario: Explicit override wins over preset

- **GIVEN** the user calls `from_preset(Preset::Robust, dt, tstop)`
  and then sets `opts.integrator = Integrator::BDF1`
- **WHEN** the simulation runs
- **THEN** the integrator used is `BDF1`, overriding the preset's
  `TRBDF2`

### Requirement: AdvancedOptions Namespace

The system SHALL aggregate all power-user numerical configuration
fields under a single `AdvancedOptions advanced{};` field on
`SimulationOptions`, replacing the today's flat 28-field surface.

The `AdvancedOptions` aggregate SHALL contain at least:
`newton`, `timestep`, `lte`, `bdf_order`, `dc`, `stiffness`,
`fallback`, `formulation`, `linear_solver`.

#### Scenario: Advanced fields reachable via nested namespace

- **GIVEN** a user has a `SimulationOptions opts`
- **WHEN** they write `opts.advanced.newton.max_iterations = 100`
- **THEN** the field is set on the underlying NewtonOptions struct
- **AND** the simulation honours the value

#### Scenario: Deprecated top-level aliases still work for one release

- **GIVEN** legacy user code that sets `opts.newton_options.max_iterations
  = 100`
- **WHEN** the field is accessed
- **THEN** the value is forwarded to `opts.advanced.newton.max_iterations`
- **AND** a `[[deprecated]]` warning is logged once per process

### Requirement: Damped Newton with Armijo Line Search

The system SHALL implement Armijo backtracking line search inside
the Newton-Raphson iteration loop. The line search SHALL trigger
when the trial residual norm does not satisfy the Armijo descent
condition with `σ = 1e-4` and SHALL halve the step length until the
condition is satisfied or `α_min = 2^-8` is reached.

The line search SHALL be enabled by default and SHALL be configurable
via `opts.advanced.newton.line_search.{enable, sigma, alpha_min}`.

#### Scenario: Line search recovers from oversized Newton step

- **GIVEN** a Newton iteration produces a step direction that, taken
  with full length, increases the residual norm
- **WHEN** the line search loop runs
- **THEN** the step length is halved until the residual decreases
- **AND** the iteration proceeds without rejecting the whole step

#### Scenario: Multilevel NPC cold start converges with line search

- **GIVEN** a 3-level NPC converter at cold start (zero state)
- **WHEN** the simulation runs with line search enabled
- **THEN** Newton converges within 20 iterations on the first
  timestep

#### Scenario: Line search disabled preserves prior behavior

- **GIVEN** `opts.advanced.newton.line_search.enable = false`
- **WHEN** Newton runs
- **THEN** the residual-norm check is skipped and the iteration
  matches the pre-v0.11 behavior

#### Scenario: Line search telemetry reports backtracks

- **GIVEN** a Newton run that triggered backtracking on at least one
  iteration
- **WHEN** the user inspects `result.backend_telemetry`
- **THEN** the `line_search_backtracks` counter is greater than zero

### Requirement: Simultaneous Switching Event Detection

The system SHALL detect multiple switching events that fall within
`ε = 1e-12 · dt` of each other and SHALL apply them atomically as a
single group, performing exactly one Newton solve per group rather
than one per event.

The grouping SHALL preserve the original event-time ordering for
crossings outside the `ε` window.

#### Scenario: 3φ inverter same-edge commutation triggers one Newton

- **GIVEN** a 3φ VSI where all six MOSFET gates commutate at the
  identical PWM edge
- **WHEN** the PWL engine detects the crossings within a single
  timestep
- **THEN** all six switch-state changes are applied atomically
- **AND** exactly one Newton solve is performed for the commutation
  instant

#### Scenario: MMC arm commutation converges

- **GIVEN** a 9-submodule MMC half-arm with all submodules turning
  on at the same phase angle
- **WHEN** the simulation runs through the commutation instant
- **THEN** the simulation converges within the step's retry budget
- **AND** the simultaneous-event grouping is reported in
  `result.backend_telemetry.simultaneous_event_groups`

#### Scenario: Single event preserves prior behavior

- **GIVEN** a circuit where exactly one switch crosses threshold per
  timestep
- **WHEN** the PWL engine runs
- **THEN** the behavior is bit-identical to the pre-v0.11 single-event
  path

### Requirement: Iterative Refinement on Direct Linear Solve

The system SHALL automatically apply one round of iterative refinement
after the direct linear solve back-substitution when the residual
norm exceeds `10 · ε_machine · ||b||`.

This refinement SHALL be invisible to the user (no API surface) and
SHALL be reported only via the `linear_refinement_steps` telemetry
counter.

#### Scenario: Ill-conditioned floating-cap network triggers refinement

- **GIVEN** a multi-level flying-cap topology where the MNA matrix
  has a condition number above 10^10
- **WHEN** the direct linear solve runs
- **THEN** iterative refinement triggers automatically
- **AND** the final residual is below `10 · ε_machine · ||b||`
- **AND** `linear_refinement_steps >= 1` is reported

#### Scenario: Well-conditioned circuit incurs no refinement overhead

- **GIVEN** a simple RC circuit with condition number below 10^4
- **WHEN** the direct linear solve runs
- **THEN** `linear_refinement_steps == 0`
- **AND** the solve completes without iterative refinement

### Requirement: Multilevel Converter Convergence Gate

The system SHALL converge from a cold-start operating point to
steady state on the following four reference multilevel converters
within the wall-clock and accuracy gates defined by their
benchmark YAML files:

- 3-level Neutral-Point-Clamped (NPC)
- T-type 3-level
- 5-level Flying-Cap
- 9-submodule three-phase Modular Multilevel Converter (MMC)

Convergence SHALL be achieved using `Preset::Auto` with no
per-circuit manual tuning.

#### Scenario: 3-level NPC matches PLECS within 0.5% RMS

- **GIVEN** the `benchmarks/multilevel/3level_npc.yaml` reference
  circuit
- **WHEN** the simulation runs with `Preset::Auto` for 100 ms
- **THEN** Pulsim's `V(out_a)`, `I(load_a)`, and `V(cap_neutral)`
  traces match the PLECS golden CSV within 0.5% RMS error

#### Scenario: 9-submodule MMC matches PSIM within 1% RMS

- **GIVEN** the `benchmarks/multilevel/mmc_9sub.yaml` reference
  circuit
- **WHEN** the simulation runs with `Preset::Auto` for 40 ms
- **THEN** Pulsim's output-phase voltage, output-phase current, and
  per-submodule cap voltages match the PSIM golden CSV within 1% RMS
  error

### Requirement: Numerical Layer Directory Organization

The system SHALL organize all user-facing numerical primitives —
integrator, Newton, line search, linear solver, DC strategy,
timestep controller, stiffness policy, formulation, event detector,
homotopy, iterative refinement, advanced options aggregate, and
preset — under a single header directory
`core/include/pulsim/v1/numerical/`.

For one release after this reorganization, old include paths
(`integration.hpp`, `convergence_aids.hpp`,
`transient_services.hpp`, relevant slices of `high_performance.hpp`
and `simulation.hpp`) SHALL continue to compile via forwarder stubs
that emit a `#pragma message` deprecation warning.

#### Scenario: New code reaches numerical primitives via the new path

- **GIVEN** new code in the kernel or tests
- **WHEN** it needs `Integrator`, `NewtonOptions`, `LinearSolverKind`,
  or any other numerical primitive
- **THEN** the canonical include is
  `#include "pulsim/v1/numerical/<primitive>.hpp"`

#### Scenario: Legacy include paths still compile

- **GIVEN** existing user or test code that does
  `#include "pulsim/v1/integration.hpp"`
- **WHEN** the file is compiled
- **THEN** the include succeeds
- **AND** the compiler emits a `#pragma message` indicating the
  deprecated path

### Requirement: Thermal-Service Dispatch Order

The `DefaultThermalService::commit_accepted_segment` SHALL execute the
following ordered sub-steps per accepted simulation step:

1. Integrate `T_i(t)` via the Euler update
   `T_i ← T_i + dt·(P_i·R_th_i − (T_i − T_amb_i))/τ_i` for every device
   with `has_thermal_model == true`.
2. Compute `scale_i = clamp(1 + α_i·(T_i − T_ref_i), 0.05, 4)` for the
   same devices.
3. Push `scale_i` into the stamp via
   `circuit_.set_device_temperature_scales(scale_i)`.
4. Push `T_i` into the device-internal `T_j_` via `set_T_j_init(T_i)`
   for every device that exposes the setter.
5. Mirror `T_i`, `peak_T_i`, `avg_T_i` into the
   `thermal_summary.device_temperatures[i]` accumulator.

Steps (1)-(3) preserve the existing closed loop on the stamp side.
Step (4) is the new sub-step this proposal adds. Step (5) is unchanged
and SHALL continue to reflect the post-step `T_i` value.

#### Scenario: Dispatch order is deterministic across runs

- **GIVEN** the same circuit + options run twice in succession
- **WHEN** the dispatch executes
- **THEN** the per-step `T_i`, `scale_i`, and device-internal `T_j_`
  values SHALL match between the two runs bit-for-bit (no random
  ordering, no map iteration order leak).

