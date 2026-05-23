## Phase 1 — LM loop in solve_with_newton_b_extra (~0.4 days)

- [ ] 1.1 Add `bool enable_lm` parameter (default `false`).
- [ ] 1.2 Internal constants: `lm_init = 1e-6`,
      `lm_shrink = 0.5`, `lm_grow = 10`, `lm_max = 1e8`.
- [ ] 1.3 Each iter: solve `(J + λ·I) · dx = -f`. Accept if
      residual drops, shrink λ. Else grow λ and retry (up to
      ~30 retries before throwing).
- [ ] 1.4 Adding λ·I: iterate `J.coeffRef(i, i) += λ` for the
      diagonal. After accept, the `-λ` undo is implicit (we
      rebuild J from seg.J fresh next iter).

## Phase 2 — SimulationOptions + run_transient plumb (~0.1 days)

- [ ] 2.1 Add `bool enable_newton_lm = false` to
      `SimulationOptions`.
- [ ] 2.2 Pass through to `solve_with_newton_b_extra` from
      `run_transient`.

## Phase 3 — Test: stiff sinusoidal rectifier (~0.4 days)

- [ ] 3.1 New file
      `tests/v2/layer5_v4/test_lm_rectifier.cpp`.
- [ ] 3.2 Same V_sine topology as V4 with κ=20 sigmoid.
- [ ] 3.3 With `enable_newton_lm = true`, verify:
      - > 95 % positive-half match
      - > 95 % negative-half match
      - mean power within 10 % of analytical

## Phase 4 — Regression + docs (~0.1 days)

- [ ] 4.1 Default `enable_newton_lm = false` preserves V4.
- [ ] 4.2 All previous tests stay green.
- [ ] 4.3 `openspec validate pulsim-v2-trust-region-newton
      --strict` passes.
- [ ] 4.4 `docs/pulsim-v2/layer4-v5-lm-newton.md`.
