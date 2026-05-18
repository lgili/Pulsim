## ADDED Requirements

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

## MODIFIED Requirements

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
