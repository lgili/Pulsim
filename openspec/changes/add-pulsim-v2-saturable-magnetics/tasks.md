## 1. Model definitions

- [ ] 1.1 Write `models::SaturableInductor::Params` (L_0, I_sat, smoothing factor, optional hysteresis stub)
- [ ] 1.2 Implement `SaturableInductor::current<S>` template for AD + Real
- [ ] 1.3 Write `models::MultiWindingTransformer::Params` (vector of L_i, coupling matrix k_ij, optional `core_id` for shared saturation)
- [ ] 1.4 Implement transformer mutual-coupling stamping math
- [ ] 1.5 Write `models::CoreLossEstimator::Params` (Steinmetz K, α, β, plus B-field history)

## 2. Pool / Assemble integration

- [ ] 2.1 Add `StoredKind::SaturableInductor` and `StoredKind::MultiWindingTransformer`
- [ ] 2.2 Extend `Entry` variant with new params types
- [ ] 2.3 Add `add_saturable_inductor` / `add_multi_winding_transformer` to `DevicePool`
- [ ] 2.4 Add getters + node-references storage (analogous to MOSFET gate_node lookup)
- [ ] 2.5 Dispatch in `assemble.hpp` (saturable as Nonlinear, multi-winding as PassiveLinear w/ cross-coupling)

## 3. Newton refresh

- [ ] 3.1 Create `nonlinear_refresh_saturable_inductor.hpp` with standalone + composable variants
- [ ] 3.2 Extend `make_combined_diode_mosfet_refresh` → rename to `make_combined_nonlinear_refresh` and include saturable magnetics
- [ ] 3.3 Update `run_transient` Python binding flag if needed

## 4. Builder + YAML + Python

- [ ] 4.1 `CircuitBuilder::add_saturable_inductor(name, from, to, L_0, I_sat, ...)`
- [ ] 4.2 `CircuitBuilder::add_multi_winding_transformer(...)` (variadic or builder-style)
- [ ] 4.3 YAML `type: saturable_inductor` + `type: multi_winding_transformer`
- [ ] 4.4 Python bindings: `b.add_saturable_inductor`, `b.add_multi_winding_transformer`

## 5. Tests

- [ ] 5.1 Unit: saturation curve shape (L decreases monotonically with |i| above I_sat)
- [ ] 5.2 Unit: multi-winding coupling matrix produces correct mutual inductance pairs
- [ ] 5.3 Unit: Newton convergence on a saturating RL circuit
- [ ] 5.4 Showcase: flyback w/ saturating primary — verify peak current is clamped
- [ ] 5.5 Showcase: 3-winding flyback w/ aux bias — verify all three outputs settle

## 6. Documentation

- [ ] 6.1 Update YAML schema docs (loader.hpp header comment)
- [ ] 6.2 Add comments in saturable_inductor.hpp explaining smoothing for Newton stability
- [ ] 6.3 Add inline example in CircuitBuilder method doc-comments

## 7. Regression + commit

- [ ] 7.1 Build all targets
- [ ] 7.2 Run full v2 regression
- [ ] 7.3 Run Python tests
- [ ] 7.4 `openspec validate add-pulsim-v2-saturable-magnetics --strict`
- [ ] 7.5 Commit and push
