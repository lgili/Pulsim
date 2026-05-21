## Phase 1 — continuation_solve primitive (~0.3 days)

- [x] 1.1 New header `pwl/continuation.hpp`.
- [x] 1.2 Function:
      ```cpp
      Vector continuation_solve(
          const PwlSegment&,
          const std::vector<NonlinearRefreshFn>&,
          const Graph&,
          const DevicePool&,
          const Vector& x_init,
          const Vector& b_extra,
          Size max_iters_per_step = 100,
          Real tol_dx  = 1e-7,
          Real tol_res = 1e-5,
          bool enable_line_search = false,
          bool enable_lm = false);
      ```
- [x] 1.3 Implementation: loop over refresh_sequence, calling
      `solve_with_newton_b_extra` for each, warm-starting from
      the previous step's `x`.

## Phase 2 — kappa-override helper (~0.2 days)

- [x] 2.1 Add `make_kappa_override_refresh(Real kappa)` to
      `pwl/nonlinear_refresh_diode.hpp`.
- [x] 2.2 Implementation: returns a `NonlinearRefreshFn` that
      walks Nonlinear branches, constructs a temporary
      `IdealDiode::Params` with `kappa = kappa_override`,
      and stamps using that.

## Phase 3 — Tests (~0.4 days)

- [x] 3.1 Unit test: continuation with a TRIVIAL single-step
      sequence (one refresh) gives the same answer as
      `solve_with_newton_b_extra` directly.
- [x] 3.2 Integration test: sinusoidal rectifier with κ=20,
      solved via continuation. The original V0 design
      (κ={2,5,10,20}) was found to NOT work because low-κ
      operating points are unphysical; ships with
      load-line warm-start + κ={20} chain, which validates
      the continuation pipeline end-to-end. See design.md
      "Implementation finding (post-V0)" for details.

## Phase 4 — Regression + docs (~0.1 days)

- [x] 4.1 All previous tests stay green (v2: 3308 assertions
      / 254 cases pass; one v1 `pulsim_simulation_tests`
      pre-existing failure unrelated).
- [x] 4.2 `openspec validate pulsim-v2-continuation-newton
      --strict` passes.
- [x] 4.3 `docs/pulsim-v2/layer4-v8-continuation.md`.
