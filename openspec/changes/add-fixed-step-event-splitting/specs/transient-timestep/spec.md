## ADDED Requirements

### Requirement: Fixed-Step Event Location

The fixed-step engine SHALL locate, inside a step, the time of every
diode commutation, gate edge and source jump when
`enable_substep_state_correction` is set. A diode commutation SHALL be
located by linear interpolation of `SwitchedDiode::crossing_signal`
between the state at the start of the step and the first solve of the
event iteration, taken under the bits the step started with. A gate
edge SHALL be taken from the schedule's `next_edge_after` when it has
one; a `switch_fn` without it SHALL keep the restart rule's
start-of-step sampling. `SimulationResult::commutation_events` SHALL record
the located time.

#### Scenario: Diode turn-off in a DCM buck
- **GIVEN** a buck in discontinuous conduction at T/100 for 1000
  periods
- **WHEN** the result's `commutation_events` are read
- **THEN** every turn-off event's `t_estimated` is within 0.05 dt of
  the zero crossing of the inductor current (it is an integer grid
  index 0.32–0.33 dt early without this change)

#### Scenario: Edge from a schedule with an edge oracle
- **GIVEN** a `NativePwm2Switch` whose duty is not a whole number of
  steps
- **WHEN** it drives a fixed-step run
- **THEN** each edge is split at `next_edge_after`

### Requirement: Fixed-Step Event Splitting

The fixed-step engine SHALL take a step that contains a located event
as a trapezoidal sub-step up to the event under the pre-event mask,
followed by the discontinuity restart rule from the event: an
SDIRK(2,2) step to the grid point and SDIRK steps for the next two grid
steps. The stored samples SHALL stay on the uniform grid
`t_start + k·dt`. Events in one step SHALL be taken in time order; an
event closer than 1 % of dt to a boundary of its interval SHALL be
taken at that boundary. Every sub-step solve SHALL pass the stateful
devices the step size and the integration rule it uses. The engine
SHALL thereby be second-order accurate for gate edges that do not lie
on the grid. The number of split steps SHALL be reported in
`SimulationResult::n_split_steps`.

#### Scenario: Gate edge off the grid is second order
- **GIVEN** 10 V, a switch, 1 Ω and 1 mH, the switch closing at
  `(k + 0.3)·dt`
- **WHEN** it runs at dt = 8, 4, 2 and 1 µs
- **THEN** the inductor-current error 0.5 ms after the edge is at most
  1e-5 A at dt = 2 µs (2.43e-3 A on the engine before the restart
  rule; first order with the restart rule alone, which snaps the edge
  to a step boundary)
- **AND** it falls by a factor of at least 3.7 per halving of dt

#### Scenario: Duty that is not a multiple of dt
- **GIVEN** a 24 V buck (100 µH, 47 µF, 1 Ω, 100 kHz) with D = 0.4567
  against the exact piecewise-LTI solution
- **WHEN** it runs 40 periods at T/50 … T/800
- **THEN** the output-voltage error is at most 2e-5 V at T/50 and
  falls by a factor of at least 3.5 per halving of dt

#### Scenario: Result does not depend on where the grid falls
- **GIVEN** a buck in discontinuous conduction run to steady state at
  T/100, T/101 and T/400
- **WHEN** the output voltage at the end is compared
- **THEN** the three values agree to below 1e-4 relative (T/101 is
  0.47 % away from the other two on the default engine; the restart
  rule alone already brings it to 6.9e-6)

#### Scenario: Output grid is unchanged
- **GIVEN** any run with split steps
- **WHEN** `result.times` is read
- **THEN** `times[k] = t_start + k·dt·store_every` exactly and
  `n_split_steps` is greater than zero

#### Scenario: Stateful device in a split step
- **GIVEN** a circuit with a charge-based nonlinear capacitor and a
  gate edge inside a step
- **WHEN** the step is split
- **THEN** the device is stamped at `dt1` under the trapezoidal rule
  and at `γ·dt2` for the SDIRK stages, and its result converges at
  second order (Richardson ratio between 3.4 and 4.7)

### Requirement: Event Splitting Is Opt-In With The Restart Rule

`enable_substep_state_correction` SHALL require the discontinuity
restart rule and the engine SHALL raise an error naming both options
when it is set without it. With the option off the fixed-step engine
SHALL produce the results of the restart rule alone, bit for bit.

#### Scenario: Option set without the restart rule
- **GIVEN** `enable_substep_state_correction=True` and
  `discontinuity_restart=False`
- **WHEN** `simulate` is called
- **THEN** it raises `UnsupportedError` (a `ValueError`) that names
  both options

#### Scenario: Option off
- **GIVEN** a circuit with off-grid edges
- **WHEN** it runs with the option off
- **THEN** its states equal those of the restart rule alone, bit for
  bit, and `n_split_steps` is 0
