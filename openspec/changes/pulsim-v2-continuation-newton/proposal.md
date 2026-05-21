## Why

Layer 4 V4 (line search) and V5 (LM damping) made Newton more
robust but BOTH fail on the κ=20 sinusoidal rectifier — the
sigmoid is too steep at zero-crossings for any single-shot
Newton variant to converge from a warm-start that's "in the
wrong basin".

The classical fix is **continuation (homotopy)**: solve a
SEQUENCE of progressively harder problems, warm-starting each
from the previous. Start with κ_easy (smooth, easy) → Newton
converges. Use that solution as warm-start for κ_intermediate.
Eventually reach κ_target.

This OpenSpec ships **a generic continuation_solve utility** +
**a kappa-override refresh helper** for the smooth-blend
IdealDiode. Together they enable the user to solve the κ=20
problem that was deferred from earlier OpenSpecs.

## What Changes

**Scope decision — Layer 4 V8** (continuation Newton):

- New header `pwl/continuation.hpp`:
  ```cpp
  Vector continuation_solve(
      const PwlSegment& seg,
      const std::vector<NonlinearRefreshFn>& refresh_sequence,
      const Graph& graph,
      const DevicePool& pool,
      const Vector& x_init,
      const Vector& b_extra,
      Size max_iters_per_step = 100,
      Real tol_dx  = 1e-7,
      Real tol_res = 1e-5,
      bool enable_line_search = false,
      bool enable_lm = false);
  ```
  Runs `solve_with_newton_b_extra` for each refresh in the
  sequence, warm-starting from the previous step's `x`. Returns
  the final converged solution.

- Helper `make_kappa_override_refresh(kappa)` in
  `pwl/nonlinear_refresh_diode.hpp`. Returns a `NonlinearRefreshFn`
  that stamps the smooth-blend IdealDiode with the GIVEN kappa,
  overriding the pool's stored value. Used by continuation
  loops.

- **Test**: build the sinusoidal rectifier at κ=20. Use
  continuation with `{2, 5, 10, 20}` to solve. Verify the
  resulting V_out tracks the expected half-wave shape (within
  the same tolerances as the original Layer 5 V4 deferred
  test).

## Impact

- **Affected specs**: ADDED requirement on `kernel-v2-solver`
  for continuation_solve.
- **Affected code** (~150 LOC):
  - NEW `pwl/continuation.hpp`
  - MODIFIED `pwl/nonlinear_refresh_diode.hpp` (+
    `make_kappa_override_refresh`)
  - NEW `tests/v2/layer5_v4/test_continuation_rectifier.cpp`
- **Migration**: zero. All additive.
- **Risk**: medium. Continuation is a well-known technique;
  the V0 sequence picks 4 fixed kappa values which may not be
  optimal for every problem. Future tuning can adapt the
  sequence based on convergence quality.
