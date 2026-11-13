## ADDED Requirements

### Requirement: Fixed-Step Discontinuity Restart

The fixed-step engine SHALL, when
`SimulationOptions::discontinuity_restart` is set, take every step that
follows a discontinuity as two backward-Euler half-steps of `dt/2`
(critical damping adjustment, Marti & Lin 1989) instead of one
trapezoidal step, and SHALL keep the trapezoidal rule on every other
step. A discontinuity is: a change of
the stamped switch mask (gate or diode) with respect to the previous
committed step; a diode flip found by the event iteration of the step
itself; a source jump at or inside the step; and the first step of a
run, except a run resumed from a `SolverSnapshot` and a
`start_from_dc_op` run whose first mask equals the mask of the DC
solution with no source jump at `t_start`.

The half-steps SHALL be solved with the matrix of the nominal step
(`2C/dt`, `2L/dt`): no `(mask, dt)` factor is created for them. The
fixed-step engine SHALL thereby be second-order accurate across mask
changes and source jumps that fall on the time grid, and from any
initial state. The number of restart steps SHALL be reported in
`SimulationResult::n_restart_steps`.

#### Scenario: Gate edge on the grid is second order
- **GIVEN** 10 V, a switch (`g_on` = 1e3 S), 1 Ω and 1 mH in series,
  the switch closing at a grid point `te`
- **WHEN** it runs on the fixed engine at dt = 4, 2 and 1 µs with the
  rule on
- **THEN** the inductor current 0.5 ms after `te` is within 1e-5 A of
  `V/(R + R_on)·(1 − exp(−t/τ))` at dt = 2 µs (it is 6.07e-3 A off
  without the rule)
- **AND** the error falls by a factor of at least 3.7 per halving of dt

#### Scenario: Buck in continuous conduction converges at second order
- **GIVEN** a 24 V buck (100 µH, 47 µF, 1 Ω, 100 kHz, D = 0.5, PWL
  switch and diode) with both gate edges on the grid, started from
  rest, against the exact piecewise-LTI solution
- **WHEN** it runs for 40 periods at T/50, T/100, T/200, T/400 and
  T/800 with the rule on
- **THEN** the inductor-current error at the end of period 40 is at
  most 5e-5 A at T/50 (1.18e-2 A without the rule)
- **AND** it falls by a factor of at least 3.8 per halving of dt

#### Scenario: The nominal factor is reused
- **GIVEN** a buck run of 1000 periods with the rule on
- **WHEN** the run completes
- **THEN** the cache holds one factor per visited mask at the nominal
  dt and its event-solver build and refactor counters are zero

#### Scenario: A consistent start does not restart
- **GIVEN** a circuit started with `start_from_dc_op=True` whose mask
  and sources do not change during the run
- **WHEN** it runs with the rule on and with the rule off
- **THEN** the two state trajectories are bit-identical and
  `n_restart_steps` is 0

#### Scenario: A resumed run stays exact
- **GIVEN** a run split at 0.2 ms and resumed from `final_snapshot`
- **WHEN** both halves use the rule
- **THEN** the resumed trajectory equals the uninterrupted one to
  1e-12 and the first resumed step is not a restart step

### Requirement: Fixed-Step First Step From An Inconsistent Start

With the restart rule on, the fixed-step engine SHALL be second-order
accurate from a start that carries no derivative history: a zero
state, builder initial conditions (`i0=`, `c0=`) and `initial_state=`.
The first step of such a run SHALL be a restart step.

#### Scenario: RL decay from an initial current
- **GIVEN** 1 Ω and 1 mH in parallel with `i0` = 5 A
- **WHEN** it runs to 5 τ at dt = 4, 2, 1 and 0.5 µs
- **THEN** the relative error against `5·exp(−5)` is below 1e-5 at
  dt = 4 µs (1.997e-3 without the rule) and its order is above 1.85
- **AND** the first step satisfies `i_1 = i_0 / (1 + dt·R/(2L))²`

#### Scenario: A start that was already consistent stays second order
- **GIVEN** an RL driven from rest by a sine source that is 0 V at
  t = 0
- **WHEN** it runs to 5 τ at dt = 4 µs
- **THEN** the relative error stays below 1e-6 and the order above 1.9

### Requirement: Fixed-Step Gate Sampling At The Step Start

With the restart rule on, the fixed-step engine SHALL stamp the step
`(t_{k−1}, t_k]` with the gate mask `switch_fn(t_{k−1})`, so that a
right-continuous schedule whose edge lies on a grid point takes effect
exactly at that point. `SimulationResult::switch_state_bits` SHALL keep
recording, for every stored sample, the mask that sample was solved
under. With the rule off the engine SHALL keep stamping the step with
`switch_fn(t_k)`.

#### Scenario: Right-continuous PWM on the grid
- **GIVEN** the RL circuit above and a schedule that is closed for
  `t ≥ te`, `te` on the grid
- **WHEN** it runs with the rule on
- **THEN** the first sample recorded with the closed mask is the one at
  `te + dt` and the equivalent edge-time error is below 0.01 dt
  (+0.500 dt without the rule)

#### Scenario: Recorded mask is the stamped mask
- **GIVEN** any fixed-step run with a `switch_fn`
- **WHEN** the result is read
- **THEN** `switch_state_bits[k]` equals the gate mask at the start of
  the step that produced sample `k`, combined with the diode bits the
  sample was solved under

### Requirement: Restart Step Event Iteration

Each backward-Euler half-step of a restart step SHALL run its own
diode event iteration from the state at the start of that half-step.
After the last mask change the engine SHALL take two half-steps under
the settled mask before the trapezoidal rule resumes: when a diode
flips during the second half-step, the following step SHALL be a
restart step too.

#### Scenario: Freewheel diode picks up the inductor current
- **GIVEN** the buck above at T/100
- **WHEN** the switch opens on a grid point
- **THEN** the freewheel diode is ON in the first half-step after the
  edge and the output settles at the reference value (12.229 V at the
  end of period 40), not at the 0.29 V a decision taken only after the
  second half-step gives

#### Scenario: No alternation in the idle interval of a DCM buck
- **GIVEN** the same buck with 50 Ω and D = 0.3 (discontinuous
  conduction)
- **WHEN** it runs at T/100 and at T/400 with the rule on
- **THEN** `v(sw)` stays within 1 % of `v_out` over the idle interval
  (it alternates 0.50 V / 26.15 V around 13.31 V without the rule)

#### Scenario: Hard-switched capacitor
- **GIVEN** 100 V through 10 Ω into a node with 1 nF and a 1 mΩ switch
  to ground, switched at 100 kHz
- **WHEN** it runs at dt = 1e-8 with the rule on
- **THEN** the on-state switch current stays within 10.00 ± 0.01 A
  (−9.8 / +29.8 A without the rule)
- **AND** at dt = 1e-9 it stays within 10.0 ± 0.5 A (−79 / +99 A
  without the rule)

### Requirement: Backward-Euler Stage For Stateful Devices

`pwl::TrBdf2Stage` SHALL provide a `BackwardEuler` stage,
`dX/dt = (X − X_n)/h_e` with `h_e` the interval the solve advances,
and the nonlinear capacitor, the saturable inductor (including gapped
and Jiles–Atherton cores), the Lauritzen diode, the IGBT tail and the
MNA-native PMSM SHALL each stamp and commit under it. The Jacobian of
the `BackwardEuler` stage at `h_e` SHALL equal the Jacobian of the
`Trapezoidal` stage at `2·h_e`; the saturable inductor's row SHALL stay
in conductance form. Every solve SHALL set the device step and the
device stage at the single point all solves pass through.

#### Scenario: Charge is conserved from the first step
- **GIVEN** a charge-based nonlinear capacitor charged from rest by a
  constant 6 A at dt = 2e-11
- **WHEN** it runs with the rule on
- **THEN** `Q(v(t))` equals `I·t` to 1e-6 relative at every sample
  from the first (it equals `I·(t − dt/2)` without the rule)

#### Scenario: Saturable inductor across an edge
- **GIVEN** a saturable inductor driven into saturation behind a
  switch that closes on a grid point onto 10 V and 1 Ω, against a
  reference integrated to 1e-12
- **WHEN** it runs to 0.5 ms at dt = 4, 2, 1 and 0.5 µs with the rule
  on and off
- **THEN** with the rule on the current error falls by a factor of at
  least 3.7 per halving of dt (it halves without the rule)
- **AND** at dt = 2 µs it is at least 20 times smaller than without
  the rule

#### Scenario: Same matrix as the trapezoidal stage
- **GIVEN** each of the five device kinds at an arbitrary operating
  point
- **WHEN** it is stamped under `BackwardEuler` at `h_e` and under
  `Trapezoidal` at `2·h_e`
- **THEN** the two Jacobian contributions are equal to 1e-14 relative

### Requirement: Fixed-Step Source Jumps

With the restart rule on, the fixed-step engine SHALL know the jump
times of pulse sources with zero rise or fall time and of PWM voltage
sources. A step that ends on a jump SHALL evaluate that source at its
left limit, and the step that starts on the jump SHALL be a restart
step. A jump strictly inside a step SHALL make that step a restart
step.

#### Scenario: Source step on the grid into an RC
- **GIVEN** a 0 → 10 V pulse source whose edge lies on a grid point,
  1 kΩ and 1 nF (τ = 1 µs)
- **WHEN** it runs at dt = τ/10, τ/20, τ/40
- **THEN** the largest error against `10·(1 − exp(−(t − te)/τ))` is at
  most 0.03 V at dt = τ/10 (0.475 V without the rule)
- **AND** it falls by a factor of at least 3.5 per halving of dt
