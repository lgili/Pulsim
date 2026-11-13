## ADDED Requirements

### Requirement: Transient Newton Roundoff Acceptance

A Newton solve taken by a transient engine SHALL return its last
iterate instead of failing, when its iteration budget is exhausted
without the absolute convergence test passing, if the step `dx` did
not contract over its last two iterations, `‖dx‖∞` is at most 1e-6 of
`max(1, ‖x‖∞)`, and the row-wise backward error
`max_i |f_i| / (|J|·|x| + |b|)_i` is at most eight times the machine
epsilon, with `|J|·|x| + |b|` the sum of the magnitudes of the terms
the row is made of. Every such acceptance SHALL be recorded in the
result (count, the largest backward error, the row it came from) and
reported to the user once per run, and an abort whose acceptance
check declined SHALL say which condition failed. The absolute tolerances `tol_newton_dx` and
`tol_newton_res` SHALL keep their meaning and their defaults, and a
solve that passes the absolute test SHALL return exactly what it
returned before.

#### Scenario: Series inductors below the roundoff floor
- **GIVEN** 10 V, 10 Ω, two series 100 µH inductors carrying 0.9 A and
  a Shockley diode
- **WHEN** it runs on the fixed engine at dt = 1e-11, 1e-12 and 1e-13 s
- **THEN** each run completes (each aborts with `‖dx‖` = 2^-29, 2^-26,
  2^-23 without this change) and reports its roundoff acceptances
- **AND** the inductor current's change over the run, divided by
  `dt/L`, equals that of the dt = 1e-8 s, 10 mH run (no acceptance) to
  1e-3 relative

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
