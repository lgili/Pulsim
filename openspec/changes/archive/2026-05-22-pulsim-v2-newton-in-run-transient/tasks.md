## Phase 1 — Extend solve_with_newton with b_extra (~0.25 days)

- [ ] 1.1 Add an overload `solve_with_newton(seg, refresh, ...,
      b_extra, ...)` that uses `seg.b_constant + b_extra` in
      the residual.
- [ ] 1.2 The existing no-b_extra signature delegates to the
      new one with a zero vector.

## Phase 2 — SimulationOptions Newton tolerances (~0.1 days)

- [ ] 2.1 Add `max_newton_iterations`, `tol_newton_dx`,
      `tol_newton_res` fields with sensible defaults.

## Phase 3 — run_transient Newton branch (~0.4 days)

- [ ] 3.1 Add `const NonlinearRefreshFn& nl_refresh = {}`
      parameter.
- [ ] 3.2 Inside the event-iteration loop, if `nl_refresh` is
      non-empty, replace `cache.solve` with
      `solve_with_newton`. Otherwise keep `cache.solve`.
- [ ] 3.3 Warm-start: pass the current `x` as `x_init` to
      `solve_with_newton`.
- [ ] 3.4 Pass `opts.max_newton_iterations`, `opts.tol_newton_dx`,
      `opts.tol_newton_res`.

## Phase 4 — Smooth-blend half-wave rectifier test (~0.4 days)

- [ ] 4.1 New test file
      `tests/v2/layer5_v4/test_integration_smooth_rectifier.cpp`.
- [ ] 4.2 Same V_sine topology as Layer 5 V2's binary-diode
      test, but `add_nonlinear_diode` instead of `add_diode`.
- [ ] 4.3 Use `refresh_smooth_diodes` as the `nl_refresh`.
- [ ] 4.4 Validate positive-half tracking, negative-half ≈ 0,
      mean power matches `(V_amp − V_F0)² / (4·R)`.

## Phase 5 — CMake + regression + docs (~0.15 days)

- [ ] 5.1 New target `pulsim_v2_layer5_v4_tests`.
- [ ] 5.2 All previous tests stay green.
- [ ] 5.3 `openspec validate pulsim-v2-newton-in-run-transient
      --strict` passes.
- [ ] 5.4 `docs/pulsim-v2/layer5-v4-newton-in-run-transient.md`.
