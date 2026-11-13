## ADDED Requirements

### Requirement: Transient Newton Roundoff Acceptance

A Newton solve taken by a transient engine SHALL return its last
iterate instead of failing, when its iteration budget is exhausted
without the absolute convergence test passing, if the step `dx` did
not contract over its last two iterations, `‖dx‖∞` is at most 1e-6 of
`max(1, ‖x‖∞)`, and the row-wise backward error
`max_i |f_i| / (|J|·|x| + |b|)_i` is at most eight times the machine
epsilon. Every such acceptance SHALL be recorded in the result (count,
first time, largest `‖dx‖`, the row it came from) and reported to the
user once per run. The absolute tolerances `tol_newton_dx` and
`tol_newton_res` SHALL keep their meaning and their defaults, and a
solve that passes the absolute test SHALL return exactly what it
returned before.

#### Scenario: Series inductors below the roundoff floor
- **GIVEN** 10 V, 10 Ω, two series 100 µH inductors carrying 0.9 A and
  a Shockley diode
- **WHEN** it runs on the fixed engine at dt = 1e-11, 1e-12 and 1e-13 s
- **THEN** each run completes (each aborts with `‖dx‖` = 2^-29, 2^-26,
  2^-23 without this change) and reports its roundoff acceptances
- **AND** the inductor current agrees with the dt = 1e-8 s run to 1e-6
  relative

#### Scenario: A converging run is unchanged
- **GIVEN** a circuit whose every Newton solve passed the absolute
  test before
- **WHEN** it runs with the acceptance enabled
- **THEN** its states are bit-identical and no roundoff acceptance is
  recorded

#### Scenario: A bouncing iterate is still refused
- **GIVEN** the phase-shifted full bridge with PWL rectifier diodes and
  a saturable primary at dt = 10 ns, whose Newton iterate bounces by
  0.76 V at a residual of 5.4e-10
- **WHEN** it runs
- **THEN** it still aborts with the diode-mask chatter diagnostic

### Requirement: DC Cascade Keeps The Absolute Step Test

The roundoff acceptance SHALL NOT be available to the DC operating
point solves (direct, gmin stepping, source stepping, continuation,
pseudo-transient). Those solves SHALL keep the absolute `dx` and
residual tests as their only convergence criterion, so that a direct
solve that cannot reach the absolute tolerance still falls through to
the next stage of the cascade.

#### Scenario: Stiff diode chain falls through to gmin stepping
- **GIVEN** the diode chain of
  `test_gmin_stepping_solves_what_the_direct_solve_cannot`
- **WHEN** the DC operating point is computed with `auto`
- **THEN** the direct solve fails, gmin stepping supplies the answer,
  and the operating point equals the previous one bit for bit (it does
  not stop at the root 10.67 V away)

### Requirement: Step Floor At The Newton Roundoff Bound

The transient engines SHALL NOT commit a step of a Newton circuit
shorter than `2·ε·max(L·|i_L|) / (1e-6·max(1, ‖x‖∞))`, the step below
which the last bit of an inductor row exceeds 1e-6 of the solution,
evaluated over the inductors of the committed state. The variable-step
engine SHALL take a step the controller wants below that bound at the
bound and SHALL count it; the fixed-step engine's step-reduction
ladder SHALL stop above it and a user step below it SHALL be refused
with a message that names the inductor and the bound. Probe solves
that commit nothing are exempt.

#### Scenario: The two bounds of a 10 mH pair
- **GIVEN** two series 10 mH inductors carrying 0.9 A in a 10 V
  circuit
- **WHEN** the bounds are evaluated
- **THEN** the absolute step test is unreachable below 4.0e-9 s (the
  fixed engine runs at 1e-8 s and aborts at 1e-9 s without this
  change) and the step floor is 4.0e-13 s

#### Scenario: Variable-step run stays above the floor
- **GIVEN** the same circuit on the variable-step engine
- **WHEN** it runs
- **THEN** no committed step is shorter than 4e-13 s and the run
  completes

#### Scenario: Step below the floor is refused
- **GIVEN** the same circuit on the fixed engine at dt = 1e-13 s
- **WHEN** `simulate` is called
- **THEN** it raises an error naming the inductor and the 4.0e-13 s
  bound (the run completes with 6e-5 V of jitter on the divider node
  without this change)

#### Scenario: Linear circuits are not floored
- **GIVEN** a circuit with no Newton device
- **WHEN** it runs on the variable-step engine
- **THEN** its step floor is the engine's `h_min` as before
