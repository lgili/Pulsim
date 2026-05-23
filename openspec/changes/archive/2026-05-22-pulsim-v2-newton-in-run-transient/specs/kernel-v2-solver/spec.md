## ADDED Requirements

### Requirement: SimulationOptions — Newton tolerances

`SimulationOptions` SHALL expose three Newton-related fields
(defaults: `max_newton_iterations = 50`, `tol_newton_dx = 1e-9`,
`tol_newton_res = 1e-9`). These MUST be passed through to
`solve_with_newton` when `run_transient` uses Newton in the
inner solve.

The fields SHALL NOT affect `valid()` — any non-negative /
finite values are acceptable.

#### Scenario: Default Newton tolerances

- **GIVEN** a default-constructed `SimulationOptions`
- **WHEN** the user reads the Newton fields
- **THEN** `max_newton_iterations` SHALL equal `50`
- **AND** `tol_newton_dx` SHALL equal `1e-9`
- **AND** `tol_newton_res` SHALL equal `1e-9`.

### Requirement: solve_with_newton — b_extra overload

`solve_with_newton` SHALL provide an overload that accepts a
`Vector b_extra` parameter and uses `seg.b_constant + b_extra`
in the residual computation. The existing no-b_extra signature
SHALL delegate to the new overload with a zero vector,
preserving Layer 4 V3 behaviour bit-identically.

#### Scenario: Zero b_extra matches Layer 4 V3 result

- **GIVEN** a linear circuit with no Nonlinear branches
- **WHEN** the user calls the b_extra overload with a zero
  vector
- **THEN** the result SHALL be bit-identical to the original
  no-b_extra call (and to `cache.solve`).

### Requirement: run_transient — Newton inner solve

`run_transient` SHALL accept an optional final parameter
`const NonlinearRefreshFn& nl_refresh = {}`. When non-empty,
the function MUST use `solve_with_newton` instead of
`cache.solve` for the inner step solve, passing:
- `seg = cache.lookup(combined_mask)`
- `refresh = nl_refresh`
- `x_init = current x` (warm start)
- `b_extra = b_extra_history + b_extra_fn(t)`
- `max_iters = opts.max_newton_iterations`
- `tol_dx  = opts.tol_newton_dx`
- `tol_res = opts.tol_newton_res`

When `nl_refresh` is empty, the function MUST preserve
Layer 5 V3 behaviour bit-identically (cache.solve).

#### Scenario: Layer 5 V3 regression

- **GIVEN** an existing run_transient call without `nl_refresh`
- **WHEN** the test runs
- **THEN** the result SHALL be bit-identical to Layer 5 V3.

#### Scenario: Smooth-blend half-wave rectifier converges

- **GIVEN** the half-wave rectifier topology with a smooth-blend
  IdealDiode (V_F0 = 0.7) and a `refresh_smooth_diodes` callback
- **WHEN** the user runs a 2-cycle transient at 60 Hz
- **THEN** positive-half samples SHALL track V_sine − V_F0
  within 1 V
- **AND** negative-half samples SHALL be within 0.5 V of zero
- **AND** the mean output power SHALL be within 10 % of
  `(V_amp − V_F0)² / (4·R)`.
