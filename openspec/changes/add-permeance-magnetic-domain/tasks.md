## 1. Linear magnetic networks (after `add-controlled-source-set`)
- [ ] 1.1 `add_winding` (gyrator `g = 1/N`, `turns > 0` refused otherwise by name), `add_permeance`, `add_air_gap` (`P = μ₀·A/l`), `add_magnetic_resistance`; builder registry of magnetic nodes
- [ ] 1.2 Preflight: a magnetic island without a reference node is refused naming one of its nodes; no tie resistor is inserted on a magnetic node
- [ ] 1.3 Test: N = 25, A = 76 mm², l = 72 mm, gap 0.5 mm, μ_r0 = 2000 (linear) → terminal inductance 111.36 µH within 1e-9 (from the current slope under a voltage step)
- [ ] 1.4 Test: two-winding transformer in permeance form against the T-model with linear magnetising inductance (`L_m = N_p²·P_c`, `L_lp = N_p²·P_σp`, `L_ls = N_s²·P_σs`), same grid → primary and secondary currents within 1e-6
- [ ] 1.5 Test: three-limb core, N = 100, `P_L` = 5 µH, `P_air` = 50 nH → `L_1 = 50 mH`, `L_0 = 166.1 µH` within 1e-6
- [ ] 1.6 Test: EE coupled inductor, N = 10, outer legs 2 µH, centre 0.5 µH → `L = 111.1 µH`, `k = 0.8` within 1e-6
- [ ] 1.7 Test: eddy resistance `R_m = 0.01 S`, N = 10, 100 V peak sine → 0.500 W within 1e-4; equals a 10 kΩ resistor across the winding
- [ ] 1.8 Flux conservation: at an internal node `|Σ Φ| ≤ 1e-12·max|Φ|` at every step

## 2. DC operating point
- [ ] 2.1 `pwl/dc_assemble.hpp`: detect nodes whose incident branches are all charge-type; replace their row by `Σ Φ_k(F) = Φ₀`
- [ ] 2.2 Test: gapped core with a winding carrying 2 A DC → `F` across the core and the gap divide as the reluctances (1e-9); the following transient starts with no flux jump
- [ ] 2.3 Test: an ordinary circuit with two series capacitors and no magnetic element is unchanged unless it was singular before

## 3. Saturable permeance
- [ ] 3.1 Odd closed-form material law and `FluxTable` law as charge laws of `NonlinearCapacitor` (shared with `add-power-mosfet-datasheet-device` step 1); `P(F)` equals the finite difference of `Φ(F)` within 1e-7; `Φ(F)` strictly increasing at 1 pA-turn spacing around 0, the knee and 10× the knee
- [ ] 3.2 `add_saturable_permeance`; validation by name (`area`, `length` > 0, `mu_r > 1`, `B_sat > 0`)
- [ ] 3.3 Test: the ETD29-class core of `gapped_core.hpp` in permeance form (saturable core permeance in series with the gap permeance) against `add_saturable_transformer`, driven to 3× the knee current → currents within 1e-3; saturated inductance 0.823 µH within 1 %
- [ ] 3.4 Engine agreement `pwl` / `trbdf2` within 1e-4 on 3.3
- [ ] 3.5 Three-limb transformer with saturable limbs under unbalanced load (tutorial case; flux in each limb recorded)

## 4. Hysteresis and excess loss (after `add-unified-device-interface`)
- [ ] 4.1 `add_hysteretic_permeance` on the existing JA law; test against `add_hysteretic_core_inductor` for the same core within 1e-6; loop area equals `∮F·dΦ` within 1e-6
- [ ] 4.2 `models/magnetic_loss.hpp`: excess-loss element with the flux rate as its unknown; `dF/du` against finite differences (1e-7); monotone at 1e-12 spacing around `u = 0`
- [ ] 4.3 Test: `c = 0.01`, `ε = 0.01 V`, N = 10, 100 V peak sine → 0.1759 W within 0.5 %
- [ ] 4.4 `fit_core_loss_elements(f, B, P_v, *, geometry)`; round-trip on synthetic loss data (1e-6); the fitted points are excluded from any reported agreement
- [ ] 4.5 Consistency report (not a validation): JA + eddy + excess fitted on sine data, compared with iGSE on a triangular flux; the difference is reported, not gated

## 5. Outputs and surface
- [ ] 5.1 `res.flux(name)`, `res.b_field(name)`, `res.h_field(name)`; loss elements in `device_loss_summary` under `core_loss`
- [ ] 5.2 Bindings, stubs, `components()` entries
- [ ] 5.3 YAML types `winding`, `permeance`, `air_gap`, `saturable_permeance`, `hysteretic_permeance`, `magnetic_resistance`, `excess_loss`; strict keys; loader tests
- [ ] 5.4 Conformance rows (`add-engine-conformance-matrix`): each element × {pwl, trbdf2, dsed, DC, AC}

## 6. Documentation
- [ ] 6.1 `AI_GUIDE.md` (when to use the T-model and when the network; the reference-node rule), `docs/losses-and-thermal.md` (in-loop core loss versus Steinmetz / iGSE), `docs/known-limitations.md` and `KNOWN_LIMITATIONS.md` (no fringing, no core-shape library, loss model not validated against measurement), `CHANGELOG.md`
- [ ] 6.2 Tutorials: three-limb transformer; integrated coupled inductor

## 7. Verification
- [ ] 7.1 `ruff check`, `pytest`, `ctest`, `openspec validate --all --strict`, stub check
