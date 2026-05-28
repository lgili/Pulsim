# Delta Spec — `dsed-engine` capability

## ADDED Requirements

### Requirement: DSED engine selectable at simulate-time
The Python facade `pulsim.simulate(...)` SHALL accept an `engine`
keyword argument with values `'pwl'` (default, v1.4.0 behaviour)
or `'dsed'`. The C++ entry point SHALL expose a corresponding
`SimulationEngine::Pwl|Dsed` enum.

#### Scenario: backward compatibility
- **GIVEN** a Python script that calls `pp.simulate(builder, t_end, dt=1e-7)`
  without specifying `engine=`
- **WHEN** the user upgrades to v2.0.0
- **THEN** the simulation runs identically to v1.4.0 (engine defaults to `'pwl'`)
- **AND** the result struct contains an `event_log` field that is `[]` (empty)

#### Scenario: DSED opt-in
- **GIVEN** a Python script calling `pp.simulate(builder, t_end=1e-3, engine='dsed', rtol=1e-6, atol=1e-9)`
- **WHEN** v2.0.0 is installed
- **THEN** the new DSED scheduler runs instead of fixed-step trap
- **AND** `result.event_log` contains the chronological event list
- **AND** `result.times` reflect the actual variable-step grid (not the user-passed `dt_hint`)

#### Scenario: invalid engine name
- **GIVEN** a user mistypes `engine='dsdee'`
- **WHEN** `pp.simulate(...)` is called
- **THEN** a `ValueError` is raised with the message listing valid options

---

### Requirement: Event prediction for built-in event types
The DSED engine SHALL provide built-in event predicates for:
gate edges (from the user's `switch_fn`), diode forward-bias
threshold crossings, diode reverse-block current zero-crossings,
inductor current zero-crossings (DCM detection), and arbitrary
voltage-threshold crossings between two named nodes.

#### Scenario: gate edge prediction
- **GIVEN** a buck converter with `switch_fn(t)` toggling at `t = 5e-6` seconds (a known gate edge)
- **WHEN** DSED is invoked with current time `t = 4.5e-6`
- **THEN** the event predictor returns `t* = 5.0e-6` as the next event
- **AND** the scheduler advances exactly to `t = 5e-6` (within machine epsilon) before handling the gate edge

#### Scenario: diode forward turn-on
- **GIVEN** a buck converter with the freewheel diode anode/cathode currently below the 0.7 V turn-on threshold
- **AND** the inductor current is forcing the anode voltage to rise
- **WHEN** DSED predicts the next event
- **THEN** the predictor returns the time when `V_a - V_c - 0.7 = 0` is satisfied (via Newton root-finding)
- **AND** the scheduler advances to that exact time before flipping the diode's mask bit

#### Scenario: simultaneous events at same time
- **GIVEN** a gate edge and a diode turn-on both occur at `t = 1e-6` within numerical tolerance
- **WHEN** the DSED scheduler reaches `t = 1e-6`
- **THEN** both events are processed in deterministic priority order (gate edges first by default)
- **AND** the mask update reflects both transitions
- **AND** `partial_refactor` is called with the union of changed columns

---

### Requirement: Path-based partial refactor at event handler
The DSED engine SHALL invoke
`PulsimSparseLuSolverT::partial_refactor(new_J, changed_cols)`
(Pulsim v1.4.0's existing API) on every mask transition triggered
by an event, when the changed-column union path length is below
`MAX_PATH_LENGTH_RATIO`. The engine SHALL fall back to a fresh
`analyze + factorize` for new masks or oversized path unions.

#### Scenario: single-bit gate transition
- **GIVEN** the DSED scheduler reaches a gate event toggling one switch bit
- **WHEN** the cache already has a segment for the new mask
- **AND** the changed-column path length is below the threshold
- **THEN** `partial_refactor` is called on the cached segment
- **AND** `cache.metrics.rank1_hits` is incremented
- **AND** the wall-clock time of the event-handler step is dominated by the path-based update

#### Scenario: multi-bit event (gate + diode at same time)
- **GIVEN** a DSED step where both a gate edge and a diode commutation fire simultaneously, changing 2 switch bits
- **WHEN** the event handler resolves both events
- **THEN** `partial_refactor` is called with `changed_cols` containing both affected columns
- **AND** `cache.metrics.multi_bit_rank1_hits` is incremented (assuming gate fires below `MAX_PATH_LENGTH_RATIO`)

---

### Requirement: Variable-step PI-controlled integrator
The DSED scheduler SHALL adapt the integration step size between
events using a PI controller on the local truncation error
estimate, with default tolerances `rtol=1e-6, atol=1e-9` and PI
gains `kP=0.7, kI=0.3` (per arXiv 2503.09898).

#### Scenario: smooth region step growth
- **GIVEN** the DSED scheduler is in a smooth steady-state region of a buck converter
- **WHEN** consecutive integrator steps report `err << tol`
- **THEN** the step size grows multiplicatively (capped at `rho_max=5`)
- **AND** the step never exceeds `dt_max` (user-provided ceiling, default = 1/10 of the smallest natural time constant)

#### Scenario: step rejection on excess error
- **GIVEN** an integrator step reports `err > tol`
- **WHEN** the scheduler evaluates the next step decision
- **THEN** the step is rejected, dt is halved, and retried
- **AND** after 5 consecutive rejections a `DSEDError` is raised

---

### Requirement: Result struct extended with event log
The simulation result SHALL include an `event_log` field
containing chronological event records of the form
`(time, event_type, old_mask, new_mask, predicate_name)`.

#### Scenario: event log on buck simulation
- **GIVEN** a buck CCM simulation in DSED mode with 10 cycles
- **WHEN** the simulation completes
- **THEN** `result.event_log` contains 20 entries (one per gate edge over 10 cycles)
- **AND** each entry's `time` field matches the gate edge time within machine epsilon
- **AND** each entry's `event_type` is `GateEdge`

#### Scenario: event log on DCM buck
- **GIVEN** a buck DCM simulation
- **WHEN** the simulation completes
- **THEN** `result.event_log` contains gate edges plus body-diode commutation events
- **AND** the `event_type` field distinguishes the two

---

### Requirement: No regression on `engine='pwl'`
The v1.4.0 fixed-step PWL cache execution path SHALL remain
unchanged when `engine='pwl'` is selected (the default). All
existing v1.4.0 unit, integration, and benchmark tests SHALL
continue to pass without modification.

#### Scenario: regression test sweep
- **WHEN** the v1.4.0 test suite is run against v2.0.0 with `engine='pwl'`
- **THEN** all 498+ existing C++ assertions pass
- **AND** all 6 existing Python end-to-end tests pass
- **AND** the captured `[rank1][microbench]` numbers match v1.4.0 within ±5 % (allowing for machine noise)

---

### Requirement: Output sampling modes
The DSED engine SHALL support two output sampling modes:
`output='native'` (irregular time grid from the actual scheduler
steps and events) and `output='fixed_dt'` (interpolated to a
uniform grid with user-specified `output_dt`).

#### Scenario: native output
- **GIVEN** `pp.simulate(..., engine='dsed', output='native')` is called
- **WHEN** the simulation completes
- **THEN** `result.times` is an irregularly-spaced array reflecting actual scheduler decisions
- **AND** `len(result.times) == len(result.states)`

#### Scenario: fixed-dt output for downstream tooling
- **GIVEN** `pp.simulate(..., engine='dsed', output='fixed_dt', output_dt=1e-7)` is called
- **WHEN** the simulation completes
- **THEN** `result.times = arange(0, t_end, 1e-7)`
- **AND** `result.states` is interpolated (Hermite cubic between native scheduler samples) onto the fixed grid
- **AND** the interpolation introduces no more than `rtol` additional error vs the native grid
