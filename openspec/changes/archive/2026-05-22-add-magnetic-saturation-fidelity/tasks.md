## 1. Saturable inductor with B-H curve
- [ ] 1.1 Audit `saturable_inductor` (parser + runtime) and document current behavior.
- [ ] 1.2 Extend YAML to accept a `bh_curve: [{B: <T>, H: <A/m>}, ...]` (piecewise-linear).
- [ ] 1.3 At each Jacobian assembly, derive μ_eff(I) from the current operating point on the curve.
- [ ] 1.4 Unit test: a saturable inductor + ideal voltage source produces the documented I_L "knee" at the saturation current.

## 2. Multi-winding transformer
- [ ] 2.1 Decide on the model: either extend `coupled_inductor` to N windings, or introduce a new `multi_winding_transformer` virtual component.
- [ ] 2.2 Implement the N-winding mutual stamp (one mutual coefficient per pair, sharing a single magnetizing inductance).
- [ ] 2.3 Unit test: center-tapped transformer used in a full-wave rectifier, verify primary current = sum of secondary currents.

## 3. Core loss KPI
- [ ] 3.1 Add `compute_core_loss(B_samples, k, alpha, beta)` to `benchmarks/kpi/`.
- [ ] 3.2 In benchmarks that supply Steinmetz coefficients in YAML, the KPI layer emits `kpi__core_loss_w_per_kg` per inductor.

## 4. Benchmarks
- [ ] 4.1 `saturating_inductor_step.yaml` — voltage source step on a saturable inductor; observe i_L knee.
- [ ] 4.2 `transformer_saturation_dc_offset.yaml` — DC-offset square wave drive; primary i_mag walks B-H asymmetrically.
- [ ] 4.3 `core_loss_steinmetz.yaml` — sine drive at a documented frequency / amplitude; KPI loss matches Steinmetz prediction within 10 %.
- [ ] 4.4 `center_tapped_full_wave_rectifier.yaml` — uses the new multi-winding transformer.

## 5. Wiring + smoke-run
- [ ] 5.1 Register all four benchmarks in `benchmarks.yaml`.
- [ ] 5.2 Generate baselines.
- [ ] 5.3 Confirm dashboard pass + no regressions.
- [ ] 5.4 Document the saturation and multi-winding modeling in `docs/MAGNETICS.md`.
