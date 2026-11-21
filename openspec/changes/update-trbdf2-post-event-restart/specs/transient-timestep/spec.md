## ADDED Requirements

### Requirement: TR-BDF2 Derivative Refresh After A Gate Event

The variable-step engine SHALL, after every gate-edge landing, commit
the last probe of the zero-time diode cascade (a trapezoidal solve of
length `h_probe` under the post-event mask) as a recorded step, so that
the derivative slots of its companion history (inductor voltage,
capacitor current, and the stored derivative of each stateful device)
hold post-edge values; the probe sample SHALL be recorded with its
switch state, so that loss and charge post-processing integrate across
the edge with the right mask. After a gate edge settled at a step
boundary, and when the probe would reach `t_end`, it SHALL instead
replace only the derivative slots with the probe's values, leaving the
states untouched. It SHALL take one probe for this purpose when the
cascade did not run or did not settle. The first step after a gate edge SHALL therefore be sized by
an error estimate of the post-edge circuit. Diode events and the
initial settle SHALL keep the committed slots: a diode turn-off at the
zero crossing of an inductor's current leaves that inductor on a
g_off-only path, whose probe derivative is a picosecond mode the
trapezoidal stage does not damp. `TrBdf2Options::refresh_after_event =
false` SHALL reproduce the previous step sequence exactly.

#### Scenario: Step after a gate edge on a buck
- **GIVEN** a 24 V buck (100 µH, 47 µF, 1 Ω, 100 kHz, D = 0.5) run for
  400 periods at the default tolerances
- **WHEN** the accepted steps are read
- **THEN** the first step after every gate edge is longer than 1e-8 s
  (the rising-edge step is 8.26e-13 s for the whole run without this
  change)
- **AND** the run takes at most 8 accepted steps per period (18
  without it) with an output error not above twice the previous one

#### Scenario: Refreshed slot equals the post-event derivative
- **GIVEN** an RL behind a switch that closes at `te`
- **WHEN** the edge is landed at `t_end` and the slots are refreshed
- **THEN** the stored inductor voltage equals `V − R·i(te)` to 1e-8
  relative (it is 0 V, the open-switch value, without the refresh)

#### Scenario: Diode events are not refreshed
- **GIVEN** a 325 V diode bridge with a 1 mH line inductor and no gate
- **WHEN** it runs with this change
- **THEN** its step sequence and result are those of the engine
  without this change

#### Scenario: Stateful device beside a diode
- **GIVEN** a circuit with a stateful device, a PWL diode and a gate
  edge
- **WHEN** the edge is landed
- **THEN** the device's derivative slot is refreshed from the probe,
  every probe sets the device step and stage, and the run agrees with
  the fixed engine

### Requirement: TR-BDF2 Post-Edge Step Memory

The variable-step engine SHALL remember, per gate-edge parity, the
step its controller proposes after the first accepted step that
follows the edge, bounded by the step ceiling, and SHALL restart from
it at the next edge of the same parity. The memory SHALL be able to
grow as well as shrink. A step clamped to an upcoming edge SHALL NOT
be remembered, and diode events SHALL have no memory.

#### Scenario: The memory recovers
- **GIVEN** a periodic PWM circuit in which one post-edge step was
  rejected down to 1e-12 s
- **WHEN** the following periods run
- **THEN** the remembered step for that parity grows back by the
  controller's factor each period until the error estimate bounds it

#### Scenario: Diode events keep the controller's step
- **GIVEN** a flyback in discontinuous conduction
- **WHEN** it runs with this change
- **THEN** its number of diode events does not exceed the number
  before the change
