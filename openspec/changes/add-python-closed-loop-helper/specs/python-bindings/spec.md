## ADDED Requirements

### Requirement: PI-PWM Closed-Loop Binding Helper

The `pulsim.control` module SHALL expose a
`bind_pi_to_switch(builder, *, pi, measured, setpoint, switch,
freq, t_start=0.0)` factory that packages a PI controller and a
PWM-driven switch into a `ClosedLoop` dataclass exposing the
`switch_fn`, `step_observer`, and per-tick history lists required
by `pulsim.simulate(...)`.

The factory SHALL:
- accept `switch` as either a device name (resolved through
  `CircuitBuilder.switch_index_of` when named lookups are
  available) or an integer index,
- throttle the controller's `update()` call to once per PWM
  period (`T_PWM = 1.0 / freq`),
- preserve the controller's existing semantics
  (`pi.update(setpoint=…, measured=…, dt=T_PWM)`),
- expose the resulting duty trace as a list of
  `(t_seconds, duty_unit)` tuples on
  `ClosedLoop.duty_history`,
- expose the per-tick controller error as
  `ClosedLoop.error_history`.

`ClosedLoop` SHALL be a frozen dataclass with the following
fields:
- `switch_fn: Callable[[float], SwitchStateMask]`
- `step_observer: Callable[[float, np.ndarray], None]`
- `duty_history: list[tuple[float, float]]`
- `error_history: list[tuple[float, float]]`

#### Scenario: Wrap a buck closed-loop in one call
- **GIVEN** a buck circuit built via `CircuitBuilder` with a
  MOSFET `"Q1"`, an output node `"vout"`, and a
  `PIController(Kp=0.08, Ki=40.0, output_min=0.05,
  output_max=0.95)`
- **WHEN** Python calls
  ```
  loop = pulsim.control.bind_pi_to_switch(
      builder,
      pi=pi,
      measured=lambda x: x[builder.node_id_of("vout")],
      setpoint=5.0,
      switch="Q1",
      freq=10e3,
  )
  res = pulsim.simulate(builder, t_end=20e-3, dt=2e-6,
                        switch_fn=loop.switch_fn,
                        step_observer=loop.step_observer)
  ```
- **THEN** the simulation completes successfully
- **AND** `np.mean(res.v("vout")[-1000:])` is within ±5 % of
  `5.0` V (V_in = 12 V → ideal duty ≈ 0.4167)
- **AND** `loop.duty_history` has at least one entry per PWM
  period in the simulated window
- **AND** the final entry of `loop.duty_history[-1][1]` is
  within ±10 % of the ideal duty `5.0 / 12.0`.

#### Scenario: Composing multiple closed loops
- **GIVEN** a dual-output buck with two independent PI loops
  bound to switches `"Q1"` and `"Q2"` and setpoints `5.0` V
  and `3.3` V respectively
- **WHEN** Python calls
  ```
  res = pulsim.simulate(
      builder, t_end=20e-3, dt=2e-6,
      closed_loops=[loop_a, loop_b],
  )
  ```
- **THEN** the simulation completes successfully
- **AND** both rails reach steady-state within their respective
  ±5 % tolerances over the last 2 ms of the run
- **AND** `loop_a.duty_history` and `loop_b.duty_history` evolve
  independently (each has its own entry per PWM period; no
  entries are missed because of composition).

#### Scenario: Rejecting conflicting callback wiring
- **GIVEN** a caller that passes both `closed_loops=[loop]` and an
  explicit `switch_fn=custom_fn` to `simulate(...)`
- **WHEN** the call is made
- **THEN** `simulate(...)` raises `ValueError`
- **AND** the message explicitly mentions both `closed_loops` and
  `switch_fn` (and `step_observer` if also set) as mutually
  exclusive.

#### Scenario: Histories track per-cycle updates
- **GIVEN** a closed-loop run over `t ∈ [0, 10 ms]` with
  `freq = 10 kHz` (i.e., 100 PWM cycles in the window)
- **WHEN** the run completes
- **THEN** `len(loop.duty_history)` is in `[99, 101]` (allowing
  one cycle of throttle slack at start/end)
- **AND** every entry's first element (timestamp) is
  monotonically non-decreasing
- **AND** the timestamps span at least `[0, 9.5 ms]`.

#### Scenario: Duty callable variant for half-bridge complementary pair
- **GIVEN** a half-bridge with switches `"Q_HIGH"` and `"Q_LOW"`
  (no dead-time for this test) driven by a single PI controller
- **WHEN** Python calls
  ```
  duty_get, observer, history = pulsim.control.bind_pi_to_duty_callable(
      builder, pi=pi, measured=..., setpoint=..., freq=10e3,
  )
  def switch_fn(t):
      mask = pulsim.SwitchStateMask(builder.num_switches)
      phase = (t % 100e-6) / 100e-6
      d = duty_get()
      mask.set(builder.switch_index_of("Q_HIGH"), phase < d)
      mask.set(builder.switch_index_of("Q_LOW"),  phase >= d)
      return mask
  res = pulsim.simulate(builder, t_end=10e-3, dt=1e-6,
                        switch_fn=switch_fn, step_observer=observer)
  ```
- **THEN** the simulation completes successfully
- **AND** `Q_HIGH` and `Q_LOW` are mutually exclusive bit-wise in
  every step (their masks never both equal `True` at the same
  `t`).
