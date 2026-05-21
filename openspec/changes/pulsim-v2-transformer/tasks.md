## Phase 1 — TwoWindingTransformer model (~0.2 days)

- [ ] 1.1 New `pulsim/v2/models/transformer.hpp` with
      Params struct + `mutual_inductance` and `cross_dt`
      helpers.

## Phase 2 — DevicePool coupling registry (~0.2 days)

- [ ] 2.1 Add `TransformerCoupling` struct +
      `add_transformer_coupling` + `transformer_couplings`
      accessor.

## Phase 3 — Cross-term stamping (~0.3 days)

- [ ] 3.1 Extend `assemble.hpp` with a post-loop pass
      that adds `-(2M/dt)` to J at cross-row/col pairs.
- [ ] 3.2 Extend `history_state.hpp`'s `compute_b_extra`
      with the cross-current history contribution.

## Phase 4 — Builder + Python (~0.2 days)

- [ ] 4.1 `CircuitBuilder::add_transformer` creates 2
      inductor branches + 1 coupling.
- [ ] 4.2 Python binding `add_transformer`.

## Phase 5 — Tests (~0.4 days)

- [ ] 5.1 `add_transformer` smoke: 2 branches + 1
      coupling.
- [ ] 5.2 k=1 1:1 transformer voltage transmission.
- [ ] 5.3 k=1 2:1 transformer turns ratio.
- [ ] 5.4 k=0 isolation test.
- [ ] 5.5 Flyback topology builds + factors.
- [ ] 5.6 Python smoke.

## Phase 6 — Regression + docs (~0.1 days)

- [ ] 6.1 All previous v2 tests stay green.
- [ ] 6.2 `openspec validate pulsim-v2-transformer
      --strict` passes.
- [ ] 6.3 `docs/pulsim-v2/layer2-v2-transformer.md`.
