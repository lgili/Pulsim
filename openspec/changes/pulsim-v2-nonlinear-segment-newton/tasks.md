## Phase 1 — solve_with_newton primitive (~0.5 days)

- [ ] 1.1 `pwl/nonlinear_refresh.hpp`: typedef
      `NonlinearRefreshFn`.
- [ ] 1.2 `pwl/nonlinear_solve.hpp`: `solve_with_newton(...)`
      function — Newton loop with explicit refactor per iter.
- [ ] 1.3 Convergence: ||dx||∞ < tol AND ||residual||∞ < tol.
- [ ] 1.4 Throws on non-convergence (with diagnostic info in
      the message).

## Phase 2 — Built-in smooth-blend IdealDiode refresh (~0.3 days)

- [ ] 2.1 `pwl/nonlinear_refresh_diode.hpp`: refresh function
      that walks the graph, finds Nonlinear branches that are
      registered as `IdealDiode` (the existing sigmoid-blend
      model), calls Layer 2's evaluate_current_and_jacobian, and
      stamps the standard 2-terminal pattern.
- [ ] 2.2 The Layer 4 cache needs to know about Nonlinear
      branches — extend DevicePool to optionally hold smooth-
      blend IdealDiode params.

## Phase 3 — Tests (~0.4 days)

- [ ] 3.1 DC diode load-line: V_dc=2V, 1kΩ, Shockley model.
      Verify Newton converges, V_diode + I·R = V_dc.
- [ ] 3.2 No-nonlinear regression: a linear circuit with
      `solve_with_newton` should converge in 1 iteration with
      the same answer as `cache.solve`.
- [ ] 3.3 Non-convergence test: degenerate circuit, expect
      throw.

## Phase 4 — CMake + regression + docs (~0.2 days)

- [ ] 4.1 New target `pulsim_v2_layer4_v3_tests`.
- [ ] 4.2 All previous tests stay green (this is purely
      additive).
- [ ] 4.3 `openspec validate
      pulsim-v2-nonlinear-segment-newton --strict` passes.
- [ ] 4.4 `docs/pulsim-v2/layer4-v3-nonlinear-newton.md`.
