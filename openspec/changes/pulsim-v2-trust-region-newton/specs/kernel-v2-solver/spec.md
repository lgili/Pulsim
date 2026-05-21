## ADDED Requirements

### Requirement: SimulationOptions — Levenberg-Marquardt flag

`SimulationOptions` SHALL include a `bool enable_newton_lm`
field (default `false`). When `true`, the Newton inner solve
in `run_transient` MUST use Levenberg-Marquardt damping
instead of plain Newton.

When BOTH `enable_newton_lm` and `enable_newton_line_search`
are set, LM SHALL take precedence (LM is strictly more
general).

#### Scenario: Default LM flag is false

- **GIVEN** a default-constructed `SimulationOptions`
- **WHEN** the user reads `opts.enable_newton_lm`
- **THEN** the value SHALL be `false`.

### Requirement: solve_with_newton_b_extra — LM overload

`solve_with_newton_b_extra` SHALL accept a final optional
parameter `bool enable_lm` (default `false`). When `true`, each
Newton iteration MUST:

1. Solve `(J + λ·I) · dx = -f` for the current `λ` (starting at
   `λ_init = 1e-6`).
2. Compute the residual at `x_trial = x + dx`.
3. If the residual decreased: accept the step, `λ *= 0.5`.
4. Else: reject, `λ *= 10`, re-solve with the new `λ`.
5. If `λ > 1e8` without acceptance, throw `std::runtime_error`.

When `enable_lm = false`, the function MUST produce
bit-identical results to V4 (plain Newton with optional line
search).

#### Scenario: Disabled LM matches V4 result

- **GIVEN** a circuit + refresh callback that converges under
  plain Newton
- **WHEN** the user calls the function with
  `enable_lm = false`
- **THEN** the result SHALL be bit-identical to the V4
  no-LM behaviour.

#### Scenario: LM on soft-sigmoid diode reaches sensible answer

- **GIVEN** a DC diode load-line: V_dc=2V → smooth-blend
  IdealDiode (κ=5) → R=1kΩ → GND
- **AND** `enable_lm = true` with relaxed tolerances
  (`tol_dx=1e-4`, `tol_res=1e-2`)
- **WHEN** the user calls `solve_with_newton_b_extra`
- **THEN** the call SHALL NOT throw
- **AND** the source constraint v_n0 = V_dc SHALL hold within
  1e-4
- **AND** v_n1 SHALL be in a sensible operating-point range
  (LM may converge to a local minimum of ||f||₂² rather than
  the analytical answer for non-convex landscapes — this is a
  known LM property).

Note: The sinusoidal κ=20 rectifier scenario originally
targeted by this OpenSpec remains DEFERRED. LM is more robust
than line search but does not solve every non-convex problem
— for truly stiff cases (steep sigmoids + sinusoidal sources
near zero-crossings), LM either converges to a local minimum
or fails. Continuation / homotopy methods (gradually increase
κ from 2 to 20) are the natural next step (future research
OpenSpec).
