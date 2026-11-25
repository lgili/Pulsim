## ADDED Requirements

### Requirement: TR-BDF2 Step Size On A Geometric Grid

The TR-BDF2 engine SHALL round every step size chosen by its controller
down to the grid `{c_j · 2^e}`, where `c_j = 2^(j/m)` for
`j = 0 … m−1`, `e` is an integer and `m` is `step_grid_per_octave`
(default 4). The rounded step SHALL never exceed the controller's
proposal and SHALL be larger than the proposal divided by `2^(1/m)`.
Steps shortened to land on a gate edge, a controller tick, a diode
crossing or `t_end`, a step equal to `h_max`, and a step at the lower
bound SHALL be exempt. The rounding SHALL use exact operations only, so
that the step sequence is the same on every supported platform. The
acceptance test of a step SHALL be unchanged. With
`step_grid_per_octave = 0`, reachable from `simulate()`, the engine
SHALL produce the step sequence and states it produced without the
grid.

#### Scenario: Controller steps lie on the grid
- **GIVEN** the reference buck on `trbdf2` with the default grid
- **WHEN** the run completes
- **THEN** every accepted step that is not a landing step, not `h_max`
  and not the lower bound has a size of the form `c_j · 2^e`
- **AND** `stats["n_offgrid_steps"]` equals the number of exempt steps

#### Scenario: Never larger than proposed
- **WHEN** `quantise_down(h, 4)` is evaluated for any finite `h > 0`
- **THEN** the result is `≤ h`, `> h / 2^(1/4)`, and
  `quantise_down` of the result is the result

#### Scenario: Grid disabled restores the previous sequence
- **WHEN** the `trbdf2` golden circuits are run with
  `step_grid_per_octave = 0`
- **THEN** sample times and states are `np.array_equal` to the goldens

#### Scenario: Accuracy is kept
- **WHEN** the analytic reference set is run with the default grid at
  the default tolerances
- **THEN** each reference error is within the engine's stated tolerance
  and at most 1.25 times its error with the grid disabled
- **AND** `n_accept` is at most 1.20 times its value with the grid
  disabled

### Requirement: TR-BDF2 Factor Reuse

The TR-BDF2 engine SHALL mark each solve as on-grid or off-grid when it
requests a factor, so that a `(mask, step size)` pair that recurs is
served from a cached factor. The TR-BDF2 statistics SHALL report
`n_offgrid_steps`.

#### Scenario: A circuit with no events reuses its factors
- **GIVEN** an RLC ladder with a sine source and no switch on `trbdf2`
- **WHEN** the run completes with the default grid
- **THEN** `event_refactors` is at most the number of distinct step
  sizes used plus `event_slot_evictions`

#### Scenario: Refactorisations no longer track steps
- **GIVEN** the reference buck, 10 ms, on `trbdf2` with the default grid
- **WHEN** the run completes
- **THEN** `event_refactors` is at most 0.30 times `n_accept`
