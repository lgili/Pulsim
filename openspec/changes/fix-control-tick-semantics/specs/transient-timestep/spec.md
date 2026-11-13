## ADDED Requirements

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
