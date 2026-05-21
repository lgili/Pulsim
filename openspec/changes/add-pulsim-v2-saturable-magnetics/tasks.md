## 1. Model definitions

- [x] 1.1 Write `models::SaturableInductor::Params` (L_0, I_sat, L_residual; n_exp dropped in favour of fixed n=2 for AD compatibility)
- [x] 1.2 Implement `SaturableInductor::current<S>` template for AD + Real (V17 — uses `(i/I_sat)²` form since ADRealN lacks `pow`)
- [x] 1.3 Write `models::MultiWindingTransformer::Params` (vector of L_i, NxN coupling matrix k_ij)
- [x] 1.4 Implement transformer mutual-coupling stamping math (pair-wise decomposition via existing `TransformerCoupling`)
- [x] 1.5 Implement Steinmetz core-loss estimator (`pulsim/v2/analysis/core_loss.hpp`) — slightly broader than the original `CoreLossEstimator::Params` scope: a post-process function `estimate_steinmetz(result, state_idx, params)` that does B_peak extraction + frequency detection + P_v computation in one call

## 2. Pool / Assemble integration

- [x] 2.1 Add `StoredKind::SaturableInductor = 14` (MultiWindingTransformer is built from pair-wise primitives — no dedicated StoredKind needed)
- [x] 2.2 Extend `Entry` variant with `SaturableInductor::Params`
- [x] 2.3 Add `add_saturable_inductor` to `DevicePool` (and `saturable_inductor_branches()` for refresh enumeration)
- [x] 2.4 Add `saturable_inductor_params(branch_id)` getter
- [x] 2.5 Dispatch in `assemble.hpp` (saturable as Nonlinear; adds a 1e-12 G_min on the branch row so KLU can factor the linear cache)

## 3. Newton refresh

- [x] 3.1 Create `nonlinear_refresh_saturable_inductor.hpp` with standalone `refresh_saturable_inductors` + composable `make_combined_nonlinear_refresh`
- [x] 3.2 `run_transient` auto-wraps the user's `nl_refresh` additively to add the saturable contribution (uses `SaturableInductorHistory` + `Real refresh_dt`)
- [ ] 3.3 Python binding flag updates — **deferred to proposal #3.4** which extended `enable_nonlinear_refresh` auto-detection to include saturable inductors

## 4. Builder + YAML + Python

- [x] 4.1 `CircuitBuilder::add_saturable_inductor(name, from, to, L_0, I_sat, L_residual)`
- [x] 4.2 `CircuitBuilder::add_multi_winding_transformer(name, windings, k_matrix)`
- [ ] 4.3 YAML `type: saturable_inductor` + `type: multi_winding_transformer` — **deferred** (Python + C++ surfaces cover the showcase use case)
- [x] 4.4 Python bindings: `b.add_saturable_inductor`, `b.add_multi_winding_transformer`

## 5. Tests

- [x] 5.1 Unit: saturation curve shape (`tests/v2/layer2/test_saturable_inductor.cpp`)
- [x] 5.2 Unit: multi-winding coupling matrix (`test_multi_winding_transformer.cpp`)
- [x] 5.3 Unit: Newton convergence on a saturating RL circuit (`test_saturable_inductor.cpp` integration)
- [x] 5.4 Showcase: boost converter with saturating primary inductor (`test_boost_saturable_inductor.cpp`, `examples/v2/boost_saturable_inductor.yaml`)
- [ ] 5.5 Showcase: 3-winding flyback w/ aux bias — **deferred** (multi-winding integration is verified via the unit test; full showcase requires more work)

## 6. Documentation

- [ ] 6.1 Update YAML schema docs — **deferred** (no YAML support added)
- [x] 6.2 Add comments in `saturable_inductor.hpp` explaining smoothing for Newton stability
- [x] 6.3 Add inline example in CircuitBuilder doc-comments

## 7. Regression + commit

- [x] 7.1 Build all targets
- [x] 7.2 Run full v2 regression (commits `547bd4d`, `1414192`, `118eb66`)
- [x] 7.3 Run Python tests
- [x] 7.4 `openspec validate add-pulsim-v2-saturable-magnetics --strict`
- [x] 7.5 Commit and push (three commits — V16 model, V17 transient, V18 showcase + Steinmetz)
