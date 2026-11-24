# transient-timestep Specification

## Purpose
TBD - created by archiving change improve-convergence-algorithms. Update Purpose after archive.
## Requirements
### Requirement: Richardson LTE Estimation

The system SHALL implement Richardson extrapolation for Local Truncation Error (LTE) estimation as an alternative to step-doubling.

Richardson LTE estimation SHALL:
- Use stored solution history (last 3 solutions)
- Compute LTE estimate without additional matrix solves
- Provide comparable accuracy to step-doubling
- Reduce computational cost by approximately 3x

#### Scenario: LTE computation with sufficient history

- **GIVEN** a transient simulation with at least 3 completed timesteps
- **WHEN** LTE is computed using Richardson extrapolation
- **THEN** the estimate uses polynomial extrapolation from history
- **AND** no additional Newton solves are performed
- **AND** the estimate is within 10% of step-doubling estimate

#### Scenario: LTE computation with insufficient history

- **GIVEN** a transient simulation in the first 2 timesteps
- **WHEN** LTE is computed
- **THEN** the system falls back to conservative estimate
- **AND** uses smaller timesteps until history is available

#### Scenario: Richardson vs step-doubling accuracy

- **GIVEN** a BDF2 integration with tolerance 1e-6
- **WHEN** both Richardson and step-doubling LTE are computed
- **THEN** Richardson estimate is within factor of 2 of step-doubling
- **AND** resulting timestep control produces similar accuracy

### Requirement: Timestep Controller

The system SHALL implement an adaptive timestep controller that
combines LTE-based and Newton-iteration-based feedback. The
controller SHALL adjust the timestep based on the LTE estimate
versus tolerance, reduce the timestep when Newton iterations exceed
the target, increase the timestep when Newton iterations fall below
the minimum threshold, and apply smoothing to prevent timestep
oscillation.

The controller's tuning fields SHALL live under
`opts.advanced.timestep.*` (per the AdvancedOptions namespace
reorganization). The legacy boolean
`SimulationOptions::adaptive_timestep` SHALL be deprecated; the
canonical selector is `opts.step_mode = TransientStepMode::Variable
| TransientStepMode::Fixed`.

#### Scenario: LTE-based timestep reduction

- **GIVEN** a transient simulation where the LTE estimate exceeds
  the tolerance
- **WHEN** the timestep controller runs
- **THEN** the next timestep is reduced

#### Scenario: Newton-feedback timestep growth

- **GIVEN** a transient simulation where Newton consistently
  converges in fewer than the target iteration count
- **WHEN** the timestep controller runs
- **THEN** the next timestep grows by a bounded factor

#### Scenario: Step mode is the canonical enable switch

- **GIVEN** a user sets `opts.step_mode = TransientStepMode::Fixed`
- **WHEN** the simulator runs
- **THEN** the timestep controller is bypassed and the user's `dt`
  is used unchanged

#### Scenario: Legacy adaptive_timestep emits deprecation warning

- **GIVEN** legacy user code that sets `opts.adaptive_timestep = true`
- **WHEN** the field is accessed
- **THEN** the equivalent `opts.step_mode = TransientStepMode::Variable`
  is applied
- **AND** a deprecation warning is logged once per process

### Requirement: Event-Driven Timestep Control

The system SHALL detect switching events and adjust timesteps to hit event times precisely.

Event detection SHALL:
- Monitor control signals of all switches
- Detect threshold crossings between timesteps
- Use binary search to find exact crossing time
- Force timestep to land at event time

#### Scenario: Switch transition detection

- **GIVEN** a MOSFET switch with Vgs threshold = 5V
- **WHEN** Vgs changes from 4V to 6V between t1 and t2
- **THEN** a switch event is detected
- **AND** the crossing time is computed via bisection

#### Scenario: Timestep adjustment for event

- **GIVEN** current time = 10us and switch event at 12us
- **WHEN** the controller computes next timestep
- **THEN** the timestep is adjusted to land exactly at 12us
- **AND** the switch state is updated at the event time

#### Scenario: Multiple events in timestep

- **GIVEN** two switch events detected at 12us and 15us
- **WHEN** the controller computes next timestep
- **THEN** the timestep targets the earlier event (12us)
- **AND** subsequent step handles the 15us event

#### Scenario: PWM source breakpoints

- **GIVEN** a PWM source with period 10us and 50% duty cycle
- **WHEN** transient simulation starts
- **THEN** breakpoints are scheduled at 0, 5us, 10us, 15us, ...
- **AND** timesteps are adjusted to hit these breakpoints
- **AND** PWM edges are captured accurately

### Requirement: Solution History Management

The system SHALL maintain a history of recent solutions for LTE estimation and event detection.

#### Scenario: History buffer management

- **GIVEN** a transient simulation in progress
- **WHEN** each timestep completes
- **THEN** the solution vector is stored in a ring buffer
- **AND** buffer maintains last 3 solutions
- **AND** older solutions are discarded

#### Scenario: History with variable timesteps

- **GIVEN** timesteps [1us, 0.5us, 2us] with corresponding solutions
- **WHEN** LTE is computed
- **THEN** the computation accounts for non-uniform timesteps
- **AND** polynomial extrapolation uses correct time coefficients

### Requirement: Transient Options Extension

The system SHALL extend `TransientOptions` with new timestep control parameters.

New options SHALL include:
- `timestep_method`: Richardson or StepDoubling (default: Richardson)
- `event_detection`: Enable event detection (default: true)
- `event_tolerance`: Tolerance for event time bisection (default: 1e-9)
- `target_newton_iters`: Target Newton iterations (default: 5)
- `timestep_increase_factor`: Max increase per step (default: 1.5)
- `timestep_decrease_factor`: Decrease factor on slow convergence (default: 0.5)

#### Scenario: Use step-doubling for compatibility

- **GIVEN** TransientOptions with `timestep_method = StepDoubling`
- **WHEN** transient simulation runs
- **THEN** LTE is computed using step-doubling
- **AND** behavior matches previous implementation

#### Scenario: Disable event detection

- **GIVEN** TransientOptions with `event_detection = false`
- **WHEN** transient simulation runs
- **THEN** switch events are not detected
- **AND** timesteps are controlled only by LTE and Newton feedback
- **AND** simulation runs faster but may miss sharp transitions

#### Scenario: Aggressive timestep increase

- **GIVEN** TransientOptions with `timestep_increase_factor = 2.0`
- **WHEN** LTE and Newton iterations allow increase
- **THEN** timestep may double between steps
- **AND** simulation completes faster with acceptable accuracy loss

### Requirement: Deprecated Integrator Enum Values

The `Integrator` enum SHALL mark `BDF3`, `BDF4`, `BDF5`, `Gear`, and
`SDIRK2` as deprecated for one release cycle (v0.11), then SHALL
remove them in v0.12. Users explicitly selecting any of these values
SHALL receive a compile-time `[[deprecated]]` warning (C++) and a
YAML parser `PULSIM_YAML_W_DEPRECATED_FIELD` warning.

**Rationale**:
- `BDF3 / BDF4 / BDF5`: not A-stable for stiff switching topologies;
  zero benchmark coverage; oscillations on diode commutation make them
  unsafe defaults. Replacement: `TRBDF2` (order 2, A-stable) or
  `RosenbrockW`.
- `Gear`: literal alias for `BDF2` (zero behavioural difference).
  Replacement: `BDF2`.
- `SDIRK2`: research-grade implementation with zero benchmark coverage
  and no production validation. Replacement: `TRBDF2`.

#### Scenario: Deprecated integrator emits warning in v0.11

- **GIVEN** a user selects `Integrator::BDF5` (or `BDF3`, `BDF4`,
  `Gear`, `SDIRK2`) in v0.11
- **WHEN** the code compiles
- **THEN** the C++ compiler emits a `[[deprecated]]` warning with the
  suggested replacement
- **AND** the integrator still works at runtime (deprecation cycle)

#### Scenario: Deprecated integrator removed in v0.12

- **GIVEN** the same code in v0.12 (after the removal cycle)
- **WHEN** the code compiles
- **THEN** the enum value no longer exists and the compile fails
- **AND** the migration message points the user at the replacement

### Requirement: Kernel-Owned Nonlinear Refresh

The transient drivers and the DC operating point SHALL stamp every
static Newton device (smooth diode, Shockley diode, level-1 MOSFET,
level-1 IGBT) without the caller supplying a refresh callback: when the
pool holds such a device and no callback is given, the kernel SHALL
build the standard chain itself. A transient run that is asked to
disable the refresh on a circuit with Newton devices SHALL raise
`UnsupportedError`; it SHALL NOT simulate those devices as open
circuits. The DC operating point MAY keep an explicit opt-out that
returns the linear network.

#### Scenario: Raw driver stamps the diode by default
- **GIVEN** 10 V through 1 kΩ into a Shockley diode to ground with
  100 nF across it
- **WHEN** the bound `run_transient` is called with its default
  arguments at `dt = 1 µs` for 1 ms
- **THEN** the node settles at 0.594 V within 0.01 V (it read 10.000 V
  before this requirement)

#### Scenario: Disabling the refresh is refused
- **WHEN** `simulate(b, 1e-3, dt=1e-6, engine='pwl',
  enable_nonlinear_refresh=False)` is called on that circuit
- **THEN** it raises `UnsupportedError` naming `engine == 'pwl'` and the
  diode, as `engine='trbdf2'` already does

#### Scenario: A circuit without Newton devices is unchanged
- **WHEN** a linear RC is run with `enable_nonlinear_refresh` set to
  `None`, `True` or `False`
- **THEN** the three state trajectories are bit-identical

### Requirement: Native Schedules Supply Exact Edges

Every native schedule SHALL be an edge oracle: each C++ switching
schedule that exposes `next_edge_after` — `SwitchSchedule`,
`NativePwm2Switch` and `NativeMultiMaskPwm` — SHALL be recognised by every transient driver, called without re-entering
Python, and used as the edge oracle by the variable-step engines, so
that each edge is a step boundary and no pulse is located by sampling.

#### Scenario: A narrow window is not stepped over
- **GIVEN** a half bridge on a 1 H / 10 Ω load driven by
  `NativeMultiMaskPwm(T = 1 ms, [0.5, 0.501, 1.0], [hi, lo, hi])`
- **WHEN** 20 periods run on `engine='trbdf2'`
- **THEN** `n_gate_events == 40` and each edge time is within 1e-12 s
  of `k·T + 0.5·T` or `k·T + 0.501·T` (the run saw 0 events before
  this requirement)

#### Scenario: Schedules already recognised are unchanged
- **WHEN** the same half bridge is driven by `NativePwm2Switch` at
  `D = 0.37`, `T = 100 µs`, on a 1 mH / 10 Ω load for 10 periods
- **THEN** the run reports 20 gate events and the same step and solve
  counts as before the change

### Requirement: Fixed-Step Truncation Estimate

The fixed-step engine SHALL estimate, at every step, the local
truncation error of each linear capacitor voltage and inductor current
as `(dt/12)·|f_{k+1} − 2·f_k + f_{k−1}|`, with `f` the derivative the
trapezoidal companion stores (`i_C/C`, `v_L/L`), and SHALL record per
stored sample the largest estimate since the previous sample, relative
to the running peak magnitude of its variable, together with the
worst value of the run, its branch and its time. An estimate SHALL be
formed only when its three derivative values were produced under one
stamped switch mask with no restart step and no source jump among
them. The estimate SHALL NOT change any state: a run with
`resolution_diagnostics` on and off yields bit-identical states.

#### Scenario: RC stepped far above its time constant
- **GIVEN** a 10 V step into an RC run at `dt = 10·τ`
- **WHEN** the result is read
- **THEN** the worst relative truncation is above 0.1 (the node is
  1.87 V off and overshoots to 11.11 V, and nothing is reported
  without this change)

#### Scenario: RC resolved
- **GIVEN** the same RC at `dt = τ/100`
- **WHEN** the result is read
- **THEN** the worst relative truncation is below 1e-5

#### Scenario: A well-resolved converter
- **GIVEN** a 24 V buck (100 µH, 47 µF, 1 Ω, 100 kHz) in continuous
  conduction at T/100
- **WHEN** it runs for 40 periods
- **THEN** the worst relative truncation is below 1e-3 and no estimate
  was formed across a gate edge

#### Scenario: Diagnostics do not change results
- **GIVEN** any fixed-step circuit, with or without Newton devices
- **WHEN** it runs with `resolution_diagnostics` on and off
- **THEN** the two state trajectories are bit-identical

### Requirement: Fixed-Step Nyquist Ringing Detector

The fixed-step engine SHALL record every node voltage or branch
current whose step-to-step difference changes sign on four or more
consecutive steps while each difference exceeds 1 % of that entry's
magnitude at the two samples it joins. A record SHALL carry the
entry, the time the run of alternating steps started, its length and
its largest difference.

#### Scenario: Under-resolved RC rings
- **GIVEN** the RC at `dt = 10·τ`
- **WHEN** it runs
- **THEN** a ringing record names the capacitor node with a run of at
  least 4 alternating steps

#### Scenario: Idle interval of a DCM buck
- **GIVEN** a buck in discontinuous conduction at T/100 on the engine
  without the discontinuity restart rule
- **WHEN** it runs
- **THEN** a ringing record names node `sw` (its samples alternate
  0.50 V / 26.15 V around the true 13.31 V)

#### Scenario: Smooth run is silent
- **GIVEN** an ideal LC oscillation resolved with 100 steps per period
- **WHEN** it runs for 1000 periods
- **THEN** no ringing record is produced

### Requirement: Diode Commutation Signals With Forward Drop

Every engine SHALL decide a PWL diode's state through the single
`SwitchedDiode` law (`decide`, `crossing_signal`), so that no engine
lets an ON diode conduct backwards: the pwl event iteration, DC
operating point and seeding through `DiodeEventState::update_from_state`;
the pwl sub-step watcher on `v − V_on`; TR-BDF2 crossing localisation,
held and forbidden checks on `v − V_f` while ON and `v − V_on` while OFF;
the DSED turn-ON predicate on `v − V_on`, the turn-OFF predicate on
`v − V_f`, and the DSED settle through `decide`. When the DSED settle
cannot converge, its error SHALL suggest a hysteresis band
`V_th = V_f + 1e-3`.

#### Scenario: Resistive half-wave turns on and off at V_f
- **GIVEN** 10 V peak, 50 Hz into a diode (`g_on = 20`, `V_f = 0.7`) and
  10 Ω
- **WHEN** simulated on pwl, trbdf2 and auto
- **THEN** conduction starts near 223 µs and ends near 9.777 ms, the
  output averages 2.8268 V, and the diode current never falls below
  −(1e-7 A + 1e-4·max|i_D|): OFF leakage `g_off·|v|` (1e-8 A on pwl)
  plus a variable-step landing just past the current zero (−1.6e-7 A on
  trbdf2 and auto), far above the −g_on·V_f = −14 A a decision site
  that ignored `V_f` would allow

#### Scenario: DSED needs a band
- **GIVEN** a circuit whose diode turns on at zero current
- **WHEN** it runs on `engine='dsed'` with `V_th = V_f + 1e-3`
- **THEN** it matches pwl
- **AND** with `V_th ≤ V_f` it either settles or raises an error that
  names the band advice

### Requirement: Control Tick Instants

Each control tick source SHALL have its own period: a source with period
`T` SHALL fire at `t_start + k·T` for every integer `k ≥ 0` such that
`t_start + k·T < t_end`, and at no other time. The instant SHALL be
computed from the integer `k`, not accumulated. The number of ticks of a
source SHALL therefore be `ceil((t_end − t_start)/T)` on every engine
that accepts it. Sources due at the same instant SHALL fire in
registration order, and the state passed to a tick SHALL be the solution
at that instant.

#### Scenario: Tick count is the same on both engines
- **GIVEN** an observer with `controller_period = 100 µs` on a 2 ms run
- **WHEN** it runs on `engine='pwl'` at `dt = 1 µs` and on
  `engine='trbdf2'`
- **THEN** the observer is called 20 times on each (it was called 2001
  times on `pwl`)

#### Scenario: First tick at the start
- **WHEN** a 2 kHz closed loop runs for 20 ms
- **THEN** it ticks 40 times, the first at `t_start` with the initial
  state and the last at 19.5 ms

#### Scenario: Coincident ticks keep registration order
- **GIVEN** three sources registered in the order 1 kHz, 2 kHz, 4 kHz
- **WHEN** they run for 10 ms
- **THEN** they fire 10, 20 and 40 times, and at every multiple of 1 ms
  the three fire in the order registered

### Requirement: Exact Multi-Rate Ticks on the Variable-Step Engine

The variable-step engine SHALL land a step boundary on every tick
instant of every source and SHALL fire, at that boundary, only the
sources due there. A tick SHALL fire before the gate functions are
sampled for the step that follows it. A run with a single source SHALL
be bit-identical to the engine before per-source periods existed.

#### Scenario: A slow loop is not driven at the fast rate
- **GIVEN** a buck with a PI loop at 2 kHz and a passive observer with
  `controller_period = 50 µs`, 20 ms, no `dt`
- **WHEN** it runs on the variable-step engine
- **THEN** the PI updates 40 times and the observer is called 400
  times; the time-weighted V_out over the last 5 ms is within 0.01 V of
  the run without the observer (13.033 V). Before, the PI updated 400
  times and V_out read 12.091 V

#### Scenario: Ticks land on their instants
- **WHEN** the tick times of that run are compared with `k·T`
- **THEN** each is within `1e-12·T`

### Requirement: Scheduled Ticks on the Fixed-Step Engine

The fixed-step engine SHALL deliver tick `k` of a source at the first
grid point at or after `t_start + k·T`, counting `k` as an integer, so
that no tick is lost or duplicated. It SHALL refuse a period shorter
than the step. When the period is not an integer multiple of the step it
SHALL warn once, stating that ticks are late by less than one step, and
SHALL record the largest lateness in the result.

#### Scenario: No tick is lost
- **GIVEN** two bucks with loops at 20 kHz and 2 kHz, 10 ms,
  `dt = 1 µs`
- **WHEN** the run completes on `engine='pwl'`
- **THEN** the loops ticked 200 and 20 times (the 20 kHz loop ticked
  198 times before)

#### Scenario: Period shorter than the step
- **WHEN** a source with `T = 0.5·dt` is passed to `engine='pwl'`
- **THEN** the call raises an error naming the period and the step

#### Scenario: Period that is not a multiple of the step
- **WHEN** a source with `T = 2.5 µs` runs at `dt = 1 µs` for 1 ms
- **THEN** it ticks 400 times, one warning is emitted, and the recorded
  largest lateness is below 1 µs

### Requirement: DSED Auto Integrator Prefers the Exact Stepper

The DSED engine with `integrator='auto'` SHALL advance a mode with the
closed-form (exact) propagator whenever that mode has one, regardless of
the mode's stiffness. It SHALL fall back to the stiffness detector's
choice between the explicit and the BDF2 integrator only for a mode
without an exact stepper. The result SHALL report the number of steps
taken by each of the three paths.

#### Scenario: Fast RC mode on the default integrator
- **GIVEN** 100 V through 10 Ω onto a node with 1 nF to ground and a
  1 mΩ switch to ground, gated at 100 kHz, 50 % duty, 4 periods
- **WHEN** `simulate(b, 4e-5, engine='dsed', switch_fn=fn)` runs with
  the default integrator
- **THEN** in each interval with the switch open the node voltage
  follows `100 − 99.99·exp(−(t − t_off)/10 ns)` within 1e-5 V, it stays
  within 0 V and 100.001 V over the whole run, and
  `n_exact_steps > 0` with `n_bdf2_steps == 0` (it ranged from −99.98 V
  to 196.08 V before)

#### Scenario: A mode without an exact stepper still runs
- **WHEN** a circuit whose current mode has no exact stepper is run
  with `integrator='auto'`
- **THEN** the run completes on the numeric integrators and agrees with
  the fixed-step engine at a converged step within 0.1 % of the
  observable's span

### Requirement: DSED BDF2 Bootstrap Is L-Stable

The first step of every DSED BDF2 segment SHALL be a backward-Euler
step, `(I − h·A)·y₁ = y₀ + h·b(t + h)`. It SHALL NOT be a
Crank–Nicolson step. The integrator SHALL remain second order on
non-stiff modes.

#### Scenario: Stiff start does not overshoot the source
- **GIVEN** `y' = (100 − y)/τ` from `y₀ = 0`
- **WHEN** one bootstrap step is taken with `h/τ = 100`
- **THEN** `y₁ = 99.0099` within 1e-9 relative (it was 196.0784), and
  the following BDF2 step gives 100.4731

#### Scenario: Explicit BDF2 on the fast RC mode
- **WHEN** the circuit of the previous requirement runs with
  `integrator='bdf2'`
- **THEN** the first sample after each turn-off is
  `100·(h/τ)/(1 + h/τ)` within 0.01 V, `h` being the step the run took
  (90.909 V at 0.1 µs), and the node voltage stays between 0 V and
  102.8 V (it reached 166.67 V and −99.98 V before)

#### Scenario: Second order is kept
- **WHEN** a mode with `h/τ = 0.01` is integrated at `h` and at `h/2`
- **THEN** the global error falls by a factor of 4 within 10 %

