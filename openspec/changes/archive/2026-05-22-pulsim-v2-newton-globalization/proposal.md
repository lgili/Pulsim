## Why

Layer 5 V4 wired Newton into `run_transient`, but plain Newton
is fragile on stiff nonlinearities: when the warm-start `x` is
in the "wrong region" of a sharp transition (e.g., the smooth-
blend diode at v_F0 during sinusoidal zero-crossings), Newton's
full step can land in a worse spot than where it started.

The standard fix is **globalization**: a damped Newton step
that backtracks when the full step would make the residual
worse. Algorithm:

```
1. Compute Newton direction:  J · dx = -f
2. Try alpha = 1 first.
3. If ||f(x + alpha·dx)|| > ||f(x)||:
     alpha *= 0.5
     repeat (up to max_backtrack)
4. Update: x := x + alpha · dx
```

This is "Armijo-lite" backtracking — simple, robust, no need to
tune extra parameters beyond a backtrack cap.

## What Changes

**Scope decision — Layer 4 V4** (the globalization
capability):

- Extend `solve_with_newton_b_extra` with an optional
  `bool enable_line_search` parameter (default `false` to
  preserve V3 behaviour).
- When enabled, each Newton iteration:
  1. Computes the full step `dx`.
  2. Evaluates `||f(x + alpha · dx)||` at `alpha = 1`.
  3. If the residual increased, halve `alpha` up to a cap
     (e.g., 8 backtracks).
  4. Accepts the step that reduced the residual (or accepts
     `alpha = 1` if none did — falls back to plain Newton at
     that iteration).

- Extend `SimulationOptions` with:
  - `bool enable_newton_line_search = false`
  - (Internally `max_backtrack = 8` is fine; not exposed.)

- `run_transient` plumbs the flag through to
  `solve_with_newton_b_extra`.

- **Test**: re-enable the sinusoidal smooth-blend half-wave
  rectifier (the test that failed in V4 with plain Newton).
  With line search enabled, Newton converges every step and
  the output power matches the analytical formula within 10 %.

## Impact

- **Affected specs**: ADDED requirement on `kernel-v2-solver`
  (line-search flag + behaviour).
- **Affected code**:
  - MODIFIED `pwl/nonlinear_solve.hpp` (~30 LOC for the
    line-search loop)
  - MODIFIED `solver/options.hpp` (+1 field)
  - MODIFIED `solver/run_transient.hpp` (~5 LOC plumb-through)
  - NEW `tests/v2/layer5_v4/test_integration_sin_rectifier_globalized.cpp`
- **Migration**: zero. Default `enable_newton_line_search = false`
  preserves V4 behaviour bit-identically.
- **Risk**: low. Line search ONLY accepts steps that reduce the
  residual; worst case is `alpha = 1` which IS plain Newton.
