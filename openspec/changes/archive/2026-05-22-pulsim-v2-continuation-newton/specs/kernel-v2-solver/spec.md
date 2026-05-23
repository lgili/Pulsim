## ADDED Requirements

### Requirement: continuation_solve primitive

`continuation_solve` SHALL run `solve_with_newton_b_extra`
once for each `NonlinearRefreshFn` in the provided sequence,
warm-starting each subsequent solve from the previous one's
converged `x`. The function MUST return the final solution
after all refreshes have been processed.

If ANY step in the sequence throws (Newton non-convergence),
`continuation_solve` SHALL propagate the exception with a
message identifying which step in the sequence failed.

#### Scenario: Single-refresh sequence matches direct Newton

- **GIVEN** a circuit and a single `NonlinearRefreshFn`
- **WHEN** the user calls `continuation_solve` with
  `refresh_sequence = {refresh}` and direct `solve_with_newton_b_extra`
  with the same refresh
- **THEN** the two results SHALL be bit-identical (same warm
  start, same single solve).

#### Scenario: Multi-step continuation converges via warm-starting

- **GIVEN** a kappa-override refresh sequence `{2, 5, 10, 20}`
  for a smooth-blend IdealDiode
- **AND** a stiff DC diode load-line that fails with κ=20
  direct Newton
- **WHEN** the user calls `continuation_solve` with the
  sequence and x_init = 0
- **THEN** the function SHALL converge to a sensible operating
  point (no throw)
- **AND** the source constraint SHALL be satisfied.

### Requirement: make_kappa_override_refresh helper

`make_kappa_override_refresh(Real kappa_override)` SHALL
return a `NonlinearRefreshFn` that stamps the smooth-blend
`IdealDiode` using `kappa_override` for ALL diodes (instead
of each diode's pool-stored kappa). Other parameters
(V_F0, R_d, G_off) MUST come from the pool unchanged.

#### Scenario: Override refresh uses override kappa

- **GIVEN** a `DevicePool` with one smooth-blend IdealDiode at
  pool-stored κ = 20
- **AND** an override refresh built with κ_override = 2
- **WHEN** the user computes the residual at a known x
  using BOTH refreshes
- **THEN** the residual at the override refresh SHALL differ
  from the pool-default refresh (different sigmoid steepness
  produces different current).
