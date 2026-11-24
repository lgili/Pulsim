## ADDED Requirements

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
