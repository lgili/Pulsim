## Phase 1 — Line-search loop in solve_with_newton_b_extra (~0.4 days)

- [ ] 1.1 Add `bool enable_line_search` parameter.
- [ ] 1.2 After computing `dx`, evaluate `||f(x + α · dx)||`
      at α = 1.
- [ ] 1.3 If residual increased, halve α (up to 8 backtracks).
- [ ] 1.4 If no α reduces residual, accept α = 1 anyway
      (matches plain Newton's pathological behaviour).
- [ ] 1.5 Track `total_backtracks` for diagnostics.

## Phase 2 — SimulationOptions + run_transient plumb (~0.1 days)

- [ ] 2.1 Add `bool enable_newton_line_search = false` to
      `SimulationOptions`.
- [ ] 2.2 Pass through to `solve_with_newton_b_extra` from
      `run_transient`.

## Phase 3 — Test: sinusoidal half-wave rectifier (~0.4 days)

- [ ] 3.1 Re-add the test from the V4 OpenSpec (deferred at
      that time due to plain-Newton fragility).
- [ ] 3.2 With `enable_newton_line_search = true`, verify:
      - > 95 % of positive-half samples track V_sine - V_F0.
      - > 95 % of negative-half samples within 0.5 V of zero.
      - Mean power within 10 % of analytical.
- [ ] 3.3 Sanity: without line search, the same test fails
      (regression confirming the bug is real and the fix is
      needed).

## Phase 4 — Regression + docs (~0.1 days)

- [ ] 4.1 All previous tests stay green.
- [ ] 4.2 Default `enable_newton_line_search = false` keeps
      bit-identical behaviour.
- [ ] 4.3 `openspec validate pulsim-v2-newton-globalization
      --strict` passes.
- [ ] 4.4 `docs/pulsim-v2/layer4-v4-newton-globalization.md`.
