## ADDED Requirements

### Requirement: Native Default Switch Schedule

When `simulate()` is called without a `switch_fn`, it SHALL drive the
engine with a C++ constant schedule holding the same default mask it
used before, and SHALL NOT install a Python callable in the step loop.
The module SHALL expose `make_constant_switch_fn(mask)` returning a
`SwitchSchedule`. Warnings about undriven controlled switches SHALL be
unchanged.

#### Scenario: No Python call per step
- **GIVEN** a passive RLC circuit and no `switch_fn`
- **WHEN** `simulate(builder, t_end=..., dt=...)` runs
- **THEN** the step loop makes no Python call and the states equal the
  goldens

#### Scenario: Same mask as before
- **GIVEN** a circuit with controlled switches and no `switch_fn`
- **WHEN** it is run on the fixed engine and on `engine="trbdf2"`
- **THEN** the switch-state record is the one the Python default
  produced, and the same warning is emitted

### Requirement: Zero-Copy Sample Times

`SimulationResult.times` SHALL be a read-only one-dimensional NumPy
array that views the result's own buffer, with the result as its base.
Its values SHALL equal the sample times previously returned as a list.

#### Scenario: Access cost independent of length
- **GIVEN** a result with 100 000 samples
- **WHEN** `res.times` is read twice
- **THEN** no copy of the sample times is made and
  `res.times.base` is the result object

#### Scenario: Read-only view
- **WHEN** code assigns to `res.times[0]`
- **THEN** NumPy raises, and `res.times.copy()` is writable

### Requirement: Run Counters In The Result

`SimulationResult.counters` SHALL return a mapping from counter name to
integer for the run that produced the result. For a fixed-step run it
SHALL contain `switch_fn_calls`, `segment_searches` and
`rollback_snapshots`.

#### Scenario: Counters readable from Python
- **WHEN** `res = simulate(builder, t_end=1e-3, dt=1e-6)` completes
- **THEN** `res.counters["switch_fn_calls"]` is an `int` and equals the
  number of solve times of the run plus a constant that does not depend
  on the number of steps
