## ADDED Requirements

### Requirement: SimulationOptions — line-search flag

`SimulationOptions` SHALL include a `bool enable_newton_line_search`
field (default `false`). When `true`, the Newton inner solve in
`run_transient` MUST enable backtracking line search to globalize
Newton iterations across stiff transitions.

#### Scenario: Default line-search flag is false

- **GIVEN** a default-constructed `SimulationOptions`
- **WHEN** the user reads `opts.enable_newton_line_search`
- **THEN** the value SHALL be `false` (preserves V4 behaviour).

### Requirement: solve_with_newton_b_extra — line-search overload

`solve_with_newton_b_extra` SHALL accept an optional
`bool enable_line_search` parameter (default `false`). When
`true`, each Newton iteration MUST perform Armijo-style
backtracking:

1. Compute the full Newton step `dx`.
2. Evaluate `||f(x + α · dx)||_∞` starting with `α = 1`.
3. If the residual at `α = 1` is NOT smaller than the
   baseline, halve `α` (up to 8 backtracks).
4. Accept the trial that first reduced the residual.
5. If no trial reduced the residual after 8 halvings, accept
   `α = 1` anyway (the "no progress" fallback — equivalent to
   plain Newton at that iteration).

When `enable_line_search = false`, the function MUST produce
bit-identical results to V3 (plain Newton, `α = 1` always).

#### Scenario: Disabled line search matches V3 result

- **GIVEN** a circuit + refresh callback that converges under
  plain Newton
- **WHEN** the user calls the function with
  `enable_line_search = false`
- **THEN** the result SHALL be bit-identical to the V3
  no-line-search behaviour.

#### Scenario: Line search on well-behaved DC problem converges

- **GIVEN** a DC diode load-line circuit (V_dc=2V → smooth
  diode κ=20 → R=1kΩ) that converges under plain Newton
- **AND** `enable_line_search = true`
- **WHEN** the user calls `solve_with_newton_b_extra`
- **THEN** the result SHALL converge to the same operating
  point as plain Newton within 1 part in 1000.

Note: The sinusoidal smooth-blend rectifier scenario originally
targeted by this OpenSpec remains DEFERRED — even with V0 line
search, the κ=20 sigmoid is too stiff at zero-crossings for
plain backtracking. Trust-region or continuation methods are
the natural next step (future research OpenSpec).
