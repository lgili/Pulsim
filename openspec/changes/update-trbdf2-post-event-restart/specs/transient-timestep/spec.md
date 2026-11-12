## ADDED Requirements

### Requirement: TR-BDF2 Derivative Refresh After An Event

The variable-step engine SHALL, after every gate-edge landing, every
diode-event landing and the initial settle, replace the derivative
slots of its companion history (inductor voltage, capacitor current,
and the stored derivative of each stateful device) with the values of
a solve taken under the post-event mask at the event time, leaving the
states untouched. It SHALL use the last probe of the zero-time cascade
when one ran and SHALL take one probe for this purpose otherwise. The
first step after an event SHALL therefore be sized by an error
estimate of the post-event circuit.

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
- **WHEN** the edge is landed and the slots are refreshed
- **THEN** the stored inductor voltage equals `V − R·i(te)` to 1e-9
  relative (it is 0 V, the open-switch value, without the refresh)

#### Scenario: Discontinuous conduction
- **GIVEN** the same buck with 50 Ω and D = 0.3
- **WHEN** it runs for 400 periods
- **THEN** no accepted step is shorter than 1e-11 s (runs of twelve
  steps between 4e-14 and 1e-12 s occur without this change)
- **AND** the inductor-current error stays at or below 1e-9 A

#### Scenario: Stateful device beside a diode
- **GIVEN** a circuit with a saturable inductor, a PWL diode and a
  gate edge
- **WHEN** the edge is landed
- **THEN** the saturable inductor's stored voltage is the probe's
  `v_from − v_to` and every probe sets the device step and stage

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
