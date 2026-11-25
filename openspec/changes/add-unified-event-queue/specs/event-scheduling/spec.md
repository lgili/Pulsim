## ADDED Requirements

### Requirement: Single Queue For Time Events

A run SHALL have exactly one event queue that knows every scheduled
instant at which the drive, the sources, the observers or the recording
change: gate edges, control ticks, source breakpoints, output samples
and the end of the run. Every engine SHALL obtain those instants from
the queue and from nowhere else. Periodic event times SHALL be computed
from an integer counter (`t0 + k·T`), never by accumulation.

#### Scenario: Variable-step engine lands on every scheduled instant
- **GIVEN** a buck with an analytic PWM schedule and a 10 kHz control
  tick
- **WHEN** it runs on the variable-step engine
- **THEN** a step ends within 1e-14 s of every gate edge and of every
  tick, and at `t_end`

#### Scenario: No drift over a long run
- **WHEN** a 100 kHz schedule runs for 10⁷ periods
- **THEN** the last edge time equals `t0 + k·T` computed directly, to
  the last bit

#### Scenario: Same events on every engine
- **GIVEN** the same circuit, schedule and tick rate
- **WHEN** it runs on the fixed, the variable-step and the event-driven
  engine
- **THEN** the three runs consume the same sequence of
  `(kind, ordinal)` time events

### Requirement: Deterministic Order Of Simultaneous Events

Events at the same instant SHALL be processed in this order: output
sample (left limit), control ticks in registration order, gate edges,
source breakpoints, settle of the self-commutated devices, then the
right-limit record. The order SHALL NOT depend on the engine or on the
order in which sources were registered across kinds.

#### Scenario: Controller sees the pre-edge state and sets the coming duty
- **GIVEN** a tick and a carrier edge at the same instant
- **WHEN** the tick handler writes a new duty
- **THEN** the handler received the state before the edge and the
  carrier period that starts at that instant uses the new duty, on
  every engine

#### Scenario: Left and right limits are both recorded
- **WHEN** a gate edge fires at a recorded instant
- **THEN** the result holds the state before the edge under the
  pre-edge mask and the state after the settle under the post-edge mask

### Requirement: Declarative Gate Sources With Analytic Edges

A gate drive SHALL be a gate source that declares the switch indices it
owns, writes only those bits, and returns the first edge strictly after
a given time. The built-in PWM helpers and native schedules SHALL be
gate sources with exact edges. The number of gate events consumed in a
run SHALL equal the analytic edge count of the sources, on every
engine.

#### Scenario: n_gate_events equals the analytic edge count on every engine
- **GIVEN** a synchronous buck at 100 kHz with dead time (four edges
  per period) simulated for 200 periods
- **WHEN** it runs on the fixed, the variable-step and the event-driven
  engine
- **THEN** `n_gate_events` is 800 on each

#### Scenario: Multi-mask native schedule is exact everywhere
- **WHEN** a `NativeMultiMaskPwm` drives a full bridge on the
  variable-step engine
- **THEN** no bisection is performed and every edge is landed within
  1e-14 s

#### Scenario: No evaluation between edges
- **GIVEN** an analytic gate source on the fixed engine
- **WHEN** the run takes 1000 steps per switching period
- **THEN** the source is evaluated at most once per edge, not once per
  step

### Requirement: Gate Composition By Switch Index

Gate sources SHALL be composed by letting each write its own bits into
one mask whose width is the circuit's number of switches. Two sources
that claim the same switch SHALL be refused with an error naming the
switch and both sources. Composition SHALL be performed once, before
any engine code, and SHALL behave identically on every engine.

#### Scenario: Two closed loops and a user schedule
- **GIVEN** two closed loops that each own one switch and a user
  schedule that owns a third
- **WHEN** the circuit runs with and without `dt`
- **THEN** both runs complete, each switch follows its own source, and
  the fixed and variable-step results agree within the variable-step
  tolerance

#### Scenario: Overlapping ownership refused
- **WHEN** a closed loop and a user schedule both drive switch `S1`
- **THEN** an error naming `S1` and both sources is raised before the
  run starts

### Requirement: Opaque Callables Are Sampled With A Declared Resolution

A gate drive given as an opaque callable SHALL be wrapped in an adapter
that locates its edges to a declared time resolution and reports that
resolution in the result. A pulse observed to be narrower than the
resolution SHALL produce a diagnostic; it MUST NOT pass without one. The
adapter SHALL be the same on every engine.

#### Scenario: Edge located to the declared resolution on every engine
- **GIVEN** a Python `switch_fn` with an edge at a known time and a
  declared resolution of 10 ns
- **WHEN** it runs on the variable-step and the event-driven engine
- **THEN** the edge is applied within 10 ns of the true time on both

#### Scenario: Dead time survives an opaque callable on the event-driven engine
- **GIVEN** a synchronous buck with 100 ns dead time driven by a plain
  Python callable
- **WHEN** it runs on the event-driven engine
- **THEN** the body diodes conduct during the dead time and the output
  is 11.49 V within the existing test tolerance

#### Scenario: Under-resolved pulse is reported
- **WHEN** a sampled gate shows a pulse narrower than its resolution
- **THEN** the result carries a diagnostic naming the switch, the pulse
  width and the resolution

### Requirement: Per-Rate Control Ticks

Every observer, closed loop and control block SHALL register its own
rate `{period, offset}`. Its handler SHALL fire once at every
`t0 + offset + k·period` (`k ≥ 0`) that is earlier than `t_end`, that
is `ceil((t_end − t0 − offset)/period)` times, on every engine,
independently of the other rates and of the step size, and SHALL
receive the scheduled tick time.

#### Scenario: Two rates
- **GIVEN** a 2 kHz PI loop and a 20 kHz monitor over 20 ms
- **WHEN** the circuit runs on the fixed and the variable-step engine
- **THEN** the PI fires 40 times and the monitor 400 times on both, and
  the output voltage agrees between the engines within the
  variable-step tolerance

#### Scenario: No lost ticks on the fixed engine
- **GIVEN** a 10 kHz loop over 20 ms at `dt = 2 µs`
- **WHEN** it runs on the fixed engine
- **THEN** the handler fires 200 times

#### Scenario: Python runs at ticks only
- **GIVEN** a closed loop built by `bind_pi_to_switch`
- **WHEN** it runs for 200 switching periods at 1000 steps per period
- **THEN** the Python tick function is called once per control period
  and no other Python callable is invoked by the run

### Requirement: Source Breakpoints Are Time Events

A source whose waveform has corners (pulse, PWM) SHALL declare its next
breakpoint, and a variable-step engine SHALL end a step on it. Smooth
sources SHALL declare none.

#### Scenario: Pulse-driven RC converges with the tolerance
- **GIVEN** an RC driven by a 10 V pulse source
- **WHEN** it runs on the variable-step engine at `rtol` 1e-4, 1e-6 and
  1e-8
- **THEN** the error against the analytic response decreases with
  `rtol` and is below 1.8e-4 V at 1e-6

### Requirement: One State-Event Locator And One Mode Decision

State events SHALL be located by one bracketing root finder that every
engine uses with its own trial evaluator (a state event is a
self-commutated device's crossing signal or a device's declared event
function), and mode decisions SHALL be taken by one settle routine that
calls the device's `decide`.
No engine SHALL contain its own copy of either.

#### Scenario: Conduction instants agree across engines
- **GIVEN** 10 V peak, 50 Hz into a diode (`g_on = 20`, `V_f = 0.7`)
  and 10 Ω
- **WHEN** it runs on the variable-step and the event-driven engine
- **THEN** conduction starts at 223 µs and ends at 9.777 ms within 1 µs
  on both

#### Scenario: Mask cycle is reported once, the same way
- **GIVEN** a circuit whose diodes cannot settle
- **WHEN** it runs on the fixed engine and in the DC operating point
- **THEN** both report the cycle through the same record (time,
  iterations, cycle detected)

### Requirement: Event Scheduling Allocates Nothing In The Step Loop

The queue, the gate sources and the locator SHALL perform no heap
allocation after a run has started, and a step with no pending event
SHALL cost one time comparison.

#### Scenario: Zero allocations with events
- **GIVEN** a three-phase inverter with dead time and a control tick
- **WHEN** 10 000 steps run after warm-up under the allocation counter
- **THEN** the number of heap allocations is zero

#### Scenario: Fixed-step cost does not rise
- **WHEN** the buck and the VSI of the audit are timed on the fixed
  engine before and after the change
- **THEN** solves per step are equal and ns/step is not above the
  previous value (best of five, same machine)

### Requirement: Output Samples Are Queue Events

Recorded output samples SHALL be time events of the queue with the
lowest priority at an instant; a sample SHALL record the state and
SHALL NOT change a mode, arm a restart or alter the steps or segments
taken between other events.

#### Scenario: Output grid does not change the run
- **GIVEN** the DCM buck on the exact strategy over 100 periods
- **WHEN** it is simulated with `dt_out = T/7` and with `dt_out = T`
- **THEN** both runs have the same number of non-sample events and
  agree at the common sample times within 1e-12

### Requirement: One Mode Selection Per Event Instant

When the complementarity mode selection is in effect the kernel SHALL
call it exactly once per event instant, after the instant's state
events, ticks, gate edges and source breakpoints are applied and
before its samples, and SHALL call the re-initialisation service
after it.

#### Scenario: Gate edge and diode crossing together
- **GIVEN** the boost (12 V, 100 µH, 47 µF, 500 Ω, D = 0.5) in DCM
  with a gate edge placed at a located diode turn-off
- **WHEN** the instant is handled
- **THEN** one selection is made for both, and the mode after the
  instant is the one that selection returned
