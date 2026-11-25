## ADDED Requirements

### Requirement: DSED Auto Integrator Prefers the Exact Stepper

The DSED engine with `integrator='auto'` SHALL advance a mode with the
closed-form (exact) propagator whenever that mode has one, regardless of
the mode's stiffness. It SHALL fall back to the stiffness detector's
choice between the explicit and the BDF2 integrator only for a mode
without an exact stepper. The result SHALL report the number of steps
taken by each of the three paths.

#### Scenario: Fast RC mode on the default integrator
- **GIVEN** 100 V through 10 Ω onto a node with 1 nF to ground and a
  1 mΩ switch to ground, gated at 100 kHz, 50 % duty, 4 periods
- **WHEN** `simulate(b, 4e-5, engine='dsed', switch_fn=fn)` runs with
  the default integrator
- **THEN** in each interval with the switch open the node voltage
  follows `100 − 99.99·exp(−(t − t_off)/10 ns)` within 1e-5 V, it stays
  within 0 V and 100.001 V over the whole run, and
  `n_exact_steps > 0` with `n_bdf2_steps == 0` (it ranged from −99.98 V
  to 196.08 V before)

#### Scenario: A mode without an exact stepper still runs
- **WHEN** a circuit whose current mode has no exact stepper is run
  with `integrator='auto'`
- **THEN** the run completes on the numeric integrators and agrees with
  the fixed-step engine at a converged step within 0.1 % of the
  observable's span

### Requirement: DSED BDF2 Bootstrap Is L-Stable

The first step of every DSED BDF2 segment SHALL be a backward-Euler
step, `(I − h·A)·y₁ = y₀ + h·b(t + h)`. It SHALL NOT be a
Crank–Nicolson step. The integrator SHALL remain second order on
non-stiff modes.

#### Scenario: Stiff start does not overshoot the source
- **GIVEN** `y' = (100 − y)/τ` from `y₀ = 0`
- **WHEN** one bootstrap step is taken with `h/τ = 100`
- **THEN** `y₁ = 99.0099` within 1e-9 relative (it was 196.0784), and
  the following BDF2 step gives 100.4731

#### Scenario: Explicit BDF2 on the fast RC mode
- **WHEN** the circuit of the previous requirement runs with
  `integrator='bdf2'`
- **THEN** the first sample after each turn-off is
  `100·(h/τ)/(1 + h/τ)` within 0.01 V, `h` being the step the run took
  (90.909 V at 0.1 µs), and the node voltage stays between 0 V and
  102.8 V (it reached 166.67 V and −99.98 V before)

#### Scenario: Second order is kept
- **WHEN** a mode with `h/τ = 0.01` is integrated at `h` and at `h/2`
- **THEN** the global error falls by a factor of 4 within 10 %
