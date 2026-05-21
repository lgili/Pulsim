## ADDED Requirements

### Requirement: make_diode_aware_initial_guess helper

`make_diode_aware_initial_guess(graph, pool, b_extra)` SHALL
return a `Vector` of size `pool.state_size(graph)` that places
Newton inside the correct basin of attraction for typical
source → diode → load circuits.

The helper MUST walk all branches in the graph. For each
branch stored as `DevicePool::StoredKind::VoltageSource`, it
SHALL:
- Read the source's pool-stored voltage `V_pool` and the
  source's branch-current variable index `src_var`.
- Compute the effective voltage `V_eff = V_pool −
  b_extra[src_var]` (the standard MNA modulation convention).
- Write `V_eff` to the returned vector at index `branch.from`,
  provided `branch.from` is not the ground node.

The helper MUST NOT modify the `pool` or `graph`. It returns
a fresh vector built from scratch each call.

#### Scenario: Source value is written onto the from-node

- **GIVEN** a circuit with a single voltage source (V=3.5)
  from `n0` to ground
- **AND** `b_extra = 0`
- **WHEN** the user calls `make_diode_aware_initial_guess`
- **THEN** the returned vector SHALL have `guess[n0] = 3.5`
  within numerical precision
- **AND** all other entries SHALL be zero.

#### Scenario: b_extra modulation folds into the source value

- **GIVEN** a voltage source registered with `pool.V = 0`
- **AND** `b_extra[source_branch_var] = −7.5` (the canonical
  pulsim pattern for overlaying a sinusoidal source value)
- **WHEN** the user calls `make_diode_aware_initial_guess`
- **THEN** `guess[source_from_node]` SHALL equal +7.5
  within numerical precision (the effective source voltage
  at the time the b_extra was sampled).

#### Scenario: κ=20 stiff rectifier solves from the auto warm-start

- **GIVEN** the κ=20 sinusoidal rectifier (V_amp=10 V,
  60 Hz, R_load=10 Ω) deferred from V4 → V9
- **WHEN** the user calls plain
  `solve_with_newton_b_extra` at every time step with
  `x_init = make_diode_aware_initial_guess(...)` (no manual
  load-line) and `enable_line_search = true`
- **THEN** Newton SHALL converge at every step
- **AND** > 95 % of positive-half samples SHALL track
  `max(V_sine − V_F0, 0)` within 1 V
- **AND** > 95 % of negative-half samples SHALL be within
  0.5 V of zero
- **AND** the mean output power SHALL be within 15 % of
  the analytical half-wave value.
