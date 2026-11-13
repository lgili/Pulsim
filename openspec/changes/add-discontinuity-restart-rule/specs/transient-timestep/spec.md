## ADDED Requirements

### Requirement: Fixed-Step Discontinuity Restart

The fixed-step engine SHALL, when
`SimulationOptions::discontinuity_restart` is set, take the step that
contains a discontinuity and the next two steps as Alexander SDIRK(2,2)
steps (`γ = 1 − 1/√2`, L-stable, second order) instead of trapezoidal
steps, and SHALL keep the trapezoidal rule on every other step. Each
SDIRK step from `t0` over `h` SHALL be two backward-Euler-like stages of
`γh`: stage 1 ends at `t0 + γh`; stage 2 ends at `t0 + h` and starts
from the pseudo previous state `x0* = x0 + ((1 − γ)/γ)(x1 − x0)`; the
commit SHALL be taken once per step from stage 2.

A discontinuity is:
- a change of the stamped switch mask (gate or diode) with respect to
  the previous committed step;
- a diode flip found by the event iteration of the step itself, including
  one inside a restart step, which SHALL re-arm the window for the next
  two steps;
- a source jump at or inside the step;
- a jump declared by a `b_extra_fn` through its optional
  `next_jump_after(t)` attribute;
- the first step of a run, except a run resumed from a `SolverSnapshot`
  and a `start_from_dc_op` run.

The number of steps still owed SHALL be carried by
`SolverSnapshot::restart_steps_left` and its pickle. Both stages SHALL be
solved on one factor per mask at the fixed stage step `2γ·dt`, built
once. The number of restart steps SHALL be reported in
`SimulationResult::n_restart_steps` and the factorisations made for
them in `SimulationResult::n_restart_factorizations`. Undeclared jumps
of a `b_extra_fn` SHALL NOT restart. Enabling the rule together with
`enable_substep_state_correction` SHALL be refused, naming both options.

#### Scenario: Gate edge on the grid is second order
- **GIVEN** 10 V, a switch (`g_on` = 1e3 S), 1 Ω and 1 mH in series,
  the switch closing at a grid point `te`
- **WHEN** it runs on the fixed engine at dt = 4, 2 and 1 µs with the
  rule on
- **THEN** the inductor current 0.5 ms after `te` is within 2e-6 A of
  the exact value at dt = 2 µs (it is 6.07e-3 A off without the rule)
- **AND** the error falls by a factor of at least 3.7 per halving of dt

#### Scenario: Buck in continuous conduction converges at second order
- **GIVEN** a 24 V buck (100 µH, 47 µF, 1 Ω, 100 kHz, D = 0.5, PWL
  switch and diode) with both gate edges on the grid, started from
  rest, against the exact piecewise-LTI solution
- **WHEN** it runs for 40 periods at T/50, T/100 and T/200 with the rule
  on
- **THEN** the inductor-current error at the end of period 40 is at
  most 2.5e-7 A at T/50 (1.18e-2 A without the rule)
- **AND** it falls by a factor of at least 3.8 per halving of dt

#### Scenario: One factor per mask
- **GIVEN** a buck run of 1000 periods with the rule on
- **WHEN** the run completes
- **THEN** the event-solver build and refactor counters are zero and no
  stage solve refactors

#### Scenario: A consistent start does not restart
- **GIVEN** a circuit started with `start_from_dc_op=True` whose mask
  and sources do not change during the run
- **WHEN** it runs with the rule on and with the rule off
- **THEN** the two state trajectories are bit-identical and
  `n_restart_steps` is 0

#### Scenario: A run resumed inside a restart window stays exact
- **GIVEN** a buck run stopped one step after a gate edge, its
  `final_snapshot` pickled and resumed
- **WHEN** both parts use the rule
- **THEN** the snapshot owes two restart steps and the resumed
  trajectory equals the uninterrupted one to 1e-12

#### Scenario: Resonant modes survive the restarts
- **GIVEN** a Q = 50 series tank driven at resonance by a ±50 V PWM
  source (two restart windows per period)
- **WHEN** it runs 400 periods at T/50 and T/100 with the rule on
- **THEN** the steady-state peak current error is within 1.5 times the
  error without the rule
- **AND** an ideal LC beside an electrically isolated PWM source keeps
  its energy after 100 periods within 0.5 % of the rule-off value

#### Scenario: A declared b_extra_fn jump restarts
- **GIVEN** an RC driven by a 10 V step injected by a `b_extra_fn` that
  declares its jump through `next_jump_after`
- **WHEN** it runs at dt = τ/10 with the rule on
- **THEN** the step restarts and the largest error is below 5e-3 V
- **AND** the callback is called once per stage time, never twice for
  the same time

### Requirement: Fixed-Step First Step From An Inconsistent Start

With the restart rule on, the fixed-step engine SHALL be second-order
accurate from a start that carries no derivative history: a zero
state, builder initial conditions (`i0=`, `c0=`) and `initial_state=`.
The first three steps of such a run SHALL be restart steps.

#### Scenario: RL decay from an initial current
- **GIVEN** 1 Ω and 1 mH in parallel with `i0` = 5 A
- **WHEN** it runs to 5 τ at dt = 4, 2, 1 and 0.5 µs
- **THEN** the relative error against `5·exp(−5)` is below 1e-5 at
  dt = 4 µs (1.997e-3 without the rule) and its order is above 1.85
- **AND** each of the first three steps multiplies the current by the
  SDIRK(2,2) amplification factor of `di/dt = −(R/L) i`, and the fourth
  is trapezoidal

#### Scenario: A start that was already consistent stays second order
- **GIVEN** an RL driven from rest by a sine source that is 0 V at
  t = 0
- **WHEN** it runs to 5 τ at dt = 4 µs
- **THEN** the relative error stays below 1e-6 and the order above 1.9

### Requirement: Fixed-Step Gate Sampling At The Step Start

With the restart rule on, the fixed-step engine SHALL stamp the step
`(t_{k−1}, t_k]` with the gate mask `switch_fn(t_{k−1})`, so that a
right-continuous schedule whose edge lies on a grid point takes effect
exactly at that point, on the dynamic path and on the static path (no
inductor or capacitor) alike. `SimulationResult::switch_state_bits`
SHALL keep recording, for every stored sample, the mask that sample was
solved under. With the rule off the engine SHALL keep stamping the step
with `switch_fn(t_k)`.

#### Scenario: Right-continuous PWM on the grid
- **GIVEN** the RL circuit above and a schedule that is closed for
  `t ≥ te`, `te` on the grid
- **WHEN** it runs with the rule on
- **THEN** the first sample recorded with the closed mask is the one at
  `te + dt` and the equivalent edge-time error is below 0.01 dt
  (+0.500 dt without the rule)

#### Scenario: The static path follows the same rule
- **GIVEN** a switch and resistors only, with a PWM source, run through
  `run_transient` on a cache built at dt = 0
- **WHEN** it runs with the rule on
- **THEN** the first closed sample is one step later than with the rule
  off and the sample on a PWM jump reads the left limit

#### Scenario: Recorded mask is the stamped mask
- **GIVEN** any fixed-step run with a `switch_fn`
- **WHEN** the result is read
- **THEN** `switch_state_bits[k]` equals the gate mask at the start of
  the step that produced sample `k`, combined with the diode bits the
  sample was solved under

### Requirement: Restart Step Event Iteration

Each stage of a restart step SHALL run its own diode event iteration
from the state at the start of that stage. A diode flip in either stage
SHALL extend the restart window to cover the following two steps.

#### Scenario: Freewheel diode picks up the inductor current
- **GIVEN** the buck above at T/100
- **WHEN** the switch opens on a grid point
- **THEN** the freewheel diode is ON in the first stage after the edge
  and the output settles at the reference value (12.229 V at the end of
  period 40)

#### Scenario: A late diode flip re-arms the window
- **GIVEN** a series L-C-diode whose half-sine lasts 1.5 dt after a gate
  edge
- **WHEN** it runs with the rule on
- **THEN** the diode turns off in the second restart step and the
  window is extended: 7 restart steps in all (6 without the re-arm)

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
- **AND** at dt = 1e-9 it stays within 10.00 ± 0.02 A (−79 / +99 A
  without the rule)

### Requirement: Backward-Euler Stage For Stateful Devices

`pwl::TrBdf2Stage` SHALL provide a `BackwardEuler` stage,
`dX/dt = (X − X_n)/h_e` with `h_e` the interval the solve advances,
and the nonlinear capacitor, the saturable inductor (including gapped
and Jiles–Atherton cores), the Lauritzen diode, the IGBT tail and the
MNA-native PMSM SHALL each stamp and commit under it, and SHALL each
extrapolate their stored charge or flux to the stage-2 pseudo previous
state. The Jacobian of the `BackwardEuler` stage at `h_e` SHALL equal
the Jacobian of the `Trapezoidal` stage at `2·h_e`; the saturable
inductor's row SHALL stay in conductance form. Every solve SHALL set the
device step and the device stage at the single point all solves pass
through.

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

#### Scenario: Stateful devices and coupled windings through a restart
- **GIVEN** a Lauritzen diode reversed by a source jump, an IGBT with a
  tail turned off by a gate jump, a two-winding transformer behind a
  switch, and a salient MNA PMSM stepped by a source jump
- **WHEN** each runs at four step sizes halving from the coarsest with
  the rule on
- **THEN** the step-halving differences of the observed current shrink
  by a factor between 3.5 and 4.6 (second order); without the rule the
  first three shrink by a factor of about 2

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
  most 5e-3 V at dt = τ/10 (0.475 V without the rule)
- **AND** it falls by a factor of at least 3.5 per halving of dt
