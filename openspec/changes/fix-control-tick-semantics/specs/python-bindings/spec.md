## ADDED Requirements

### Requirement: Engine-Independent Control Composition

`simulate()` SHALL resolve `closed_loops`, `step_observer`,
`controller_period` and `switch_fn` once, before choosing an engine,
into one plan: a combined gate function, a list of tick sources with
their periods, and a list of every-step observers. Both the fixed-step
and the variable-step paths SHALL consume that plan. Gate functions
SHALL be merged with `make_combined_switch_fn`. Any number of closed
loops SHALL be accepted with or without a user `switch_fn`.

#### Scenario: Two closed loops without dt
- **GIVEN** two bucks sharing a source, each with a `ClosedLoop`
  (20 kHz and 2 kHz)
- **WHEN** `simulate(b, 10e-3, closed_loops=[la, lb])` is called
- **THEN** it completes on the variable-step engine, the loops tick 200
  and 20 times (it raised `TypeError` before)

#### Scenario: A loop next to a user gate function
- **WHEN** one closed loop drives one switch and a user `switch_fn`
  drives another, with no `dt`
- **THEN** the run completes and both switches follow their drivers

### Requirement: Controller Period Honoured or Refused

`simulate(controller_period=T)` SHALL schedule `step_observer` at the
tick instants of period `T` on every engine that accepts observers, and
SHALL raise on the engines that do not. It SHALL NOT be ignored. A
`step_observer` without a period SHALL remain an every-step callback on
the fixed-step engine and SHALL remain refused on the variable-step
engine. `result.control_tick_counts` SHALL list the ticks fired per
source in registration order.

#### Scenario: Honoured on the fixed-step engine
- **WHEN** `simulate(b, 2e-3, dt=1e-6, step_observer=obs,
  controller_period=1e-4)` runs
- **THEN** `obs` is called 20 times and
  `result.control_tick_counts == [20]`

#### Scenario: Every-step observer is unchanged
- **WHEN** the same call is made without `controller_period`
- **THEN** `obs` is called once per step, as before

#### Scenario: Refused where observers are not supported
- **WHEN** `controller_period` and `step_observer` are passed with
  `engine='dsed'`
- **THEN** the call raises and names `step_observer`
