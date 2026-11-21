Status (2026-11-19, feat/core-p5-line-and-excess): no open boxes — 4.2, 4.3, 4.5 done (DECISIONS 13). (Closeout 2026-11-19 had them open: DECISIONS 12.)

## 1. Linear magnetic networks (after `add-controlled-source-set`)
- [x] 1.1 `add_winding` (gyrator `g = 1/N`, `turns > 0` refused otherwise by name), `add_permeance`, `add_air_gap` (`P = μ₀·A/l`), `add_magnetic_resistance`; builder registry of magnetic nodes
- [x] 1.2 Preflight: a magnetic island without a reference node is refused naming one of its nodes; no tie resistor is inserted on a magnetic node
- [x] 1.3 Test: N = 25, A = 76 mm², l = 72 mm, gap 0.5 mm, μ_r0 = 2000 (linear) → terminal inductance 111.36 µH within 1e-9 (from the current slope under a voltage step)
- [x] 1.4 Test: two-winding transformer in permeance form against the T-model with linear magnetising inductance (`L_m = N_p²·P_c`, `L_lp = N_p²·P_σp`, `L_ls = N_s²·P_σs`), same grid → primary and secondary currents within 1e-6
- [x] 1.5 Test: three-limb core, N = 100, `P_L` = 5 µH, `P_air` = 50 nH → `L_1 = 50 mH`, `L_0 = 166.1 µH` within 1e-6
- [x] 1.6 Test: EE coupled inductor, N = 10, outer legs 2 µH, centre 0.5 µH → `L = 111.1 µH`, `k = 0.8` within 1e-6 — `k` is −0.8 (same-facing ports on parallel legs; DECISIONS 9)
- [x] 1.7 Test: eddy resistance `R_m = 0.01 S`, N = 10, 100 V peak sine → 0.500 W within 1e-4; equals a 10 kΩ resistor across the winding
- [x] 1.8 Flux conservation: at an internal node `|Σ Φ| ≤ 1e-12·max|Φ|` at every step — bound 1e-11: 1e-12 is the rounding floor (1.1e-12 measured after 200 steps)

## 2. DC operating point
- [x] 2.1 `pwl/dc_assemble.hpp`: detect nodes whose incident branches are all charge-type; replace their row by `Σ Φ_k(F) = Φ₀` — linear permeances only, Φ₀ = 0; one row per node reached only through permeances, or per group joined only by magnetic resistances and winding ports; a saturable permeance there is refused by name (DECISIONS 5)
- [x] 2.2 Test: gapped core with a winding carrying 2 A DC → `F` across the core and the gap divide as the reluctances (1e-9); the following transient starts with no flux jump
- [x] 2.3 Test: an ordinary circuit with two series capacitors and no magnetic element is unchanged unless it was singular before

## 3. Saturable permeance
- [x] 3.1 Odd closed-form material law and `FluxTable` law as charge laws of `NonlinearCapacitor` (shared with `add-power-mosfet-datasheet-device` step 1); `P(F)` equals the finite difference of `Φ(F)` within 1e-7; `Φ(F)` strictly increasing at 1 pA-turn spacing around 0, the knee and 10× the knee — tanh law + the existing `ChargeTable`; at 10× the knee strict at 1 nA-turn, non-decreasing at 1 pA-turn (DECISIONS 4)
- [x] 3.2 `add_saturable_permeance`; validation by name (`area`, `length` > 0, `mu_r > 1`, `B_sat > 0`)
- [x] 3.3 Test: the ETD29-class core of `gapped_core.hpp` in permeance form (saturable core permeance in series with the gap permeance) against `add_saturable_transformer`, driven to 3× the knee current → currents within 1e-3; saturated inductance 0.823 µH within 1 %
- [x] 3.4 Engine agreement `pwl` / `trbdf2` within 1e-4 on 3.3
- [x] 3.5 Three-limb transformer with saturable limbs under unbalanced load (tutorial case; flux in each limb recorded) — tutorial 09

## 4. Hysteresis and excess loss (after `add-unified-device-interface`)
add-unified-device-interface has landed (DECISIONS 7 no longer applies); closeout 2026-11-19: DECISIONS 10–12.
- [x] 4.1 `add_hysteretic_permeance` on the existing JA law; test against `add_hysteretic_core_inductor` for the same core within 1e-6; loop area equals `∮F·dΦ` within 1e-6 — unit winding + one-turn JA inductor (DECISIONS 10); measured 1.0e-9 (pwl) / 1.3e-9 (trbdf2) of the peak current, ∮F·dΦ = absorbed electrical energy to 1.1e-9 (a power identity through the lossless unit gyrator, not an independent truth; the independent check of the loop area is the `compute_bh_loop` replay of the same JA law, within 5 %) (`test_hysteretic_permeance_*`); YAML `hysteretic_permeance`; conformance cell `magnetic[hysteretic_permeance]`
- [x] 4.2 `models/magnetic_loss.hpp`: excess-loss element with the flux rate as its unknown; `dF/du` against finite differences (1e-7); monotone at 1e-12 spacing around `u = 0` — new kind `StoredKind::ExcessLoss` on every surface (DECISIONS 13); `test_device_laws.cpp` "[excess]": AD slope vs closed form (16 ulp) and central difference (1e-7), strictly increasing at 1e-12 spacing around 0, ±ε, ±10 V, odd bit-exactly, hand stamp = interface law (1e-12); `add_excess_loss`
- [x] 4.3 Test: `c = 0.01`, `ε = 0.01 V`, N = 10, 100 V peak sine → 0.1759 W within 0.5 % — measured 0.1759547 W: −8.8e-9 (pwl) / −4.0e-9 (trbdf2) vs the regularised law (mpmath), −7.4e-7 vs the ideal c·U^1.5·0.5564179 (`test_excess_loss.py`); f^1.5 slope 1.5000000013 over two decades
- [x] 4.4 `fit_core_loss_elements(f, B, P_v, *, geometry)`; round-trip on synthetic loss data (1e-6); the fitted points are excluded from any reported agreement — `pulsim.fit_core_loss_elements` (DECISIONS 11); round trip of β, k_h, R_m, c to 1e-6; the fitted R_m dissipates its eddy share k_e·f²·B²·volume in a run within 1e-4 (its `device_loss_summary` core-loss row, from the solved MMF); `fitted_on` lists every point
- [x] 4.5 Consistency report (not a validation): JA + eddy + excess fitted on sine data, compared with iGSE on a triangular flux; the difference is reported, not gated — `probes/core_loss_consistency.py` (hysteresis share from the fit, eddy + excess simulated): −0.1 … −0.4 % vs iGSE on synthetic data (DECISIONS 13)

## 5. Outputs and surface
- [x] 5.1 `res.flux(name)`, `res.b_field(name)`, `res.h_field(name)`; loss elements in `device_loss_summary` under `core_loss`
- [x] 5.2 Bindings, stubs, `components()` entries — `components()` unchanged: the elements are listed under their existing kinds
- [x] 5.3 YAML types `winding`, `permeance`, `air_gap`, `saturable_permeance`, `hysteretic_permeance`, `magnetic_resistance`, `excess_loss`; strict keys; loader tests — all seven read (`excess_loss` since DECISIONS 13: keys `from`, `to`, `c`, `eps`)
- [x] 5.4 Conformance rows (`add-engine-conformance-matrix`): each element × {pwl, trbdf2, dsed, DC, AC} — cells on pwl / trbdf2 / dsed / bdf1 (saturable: refused by name off the Newton engines); AC: linear cell against its electrical equivalent, saturable refused by name; DC: the C++ flux-row tests (`test_permeance.cpp`)

## 6. Documentation
- [x] 6.1 `AI_GUIDE.md` (when to use the T-model and when the network; the reference-node rule), `docs/losses-and-thermal.md` (in-loop core loss versus Steinmetz / iGSE), `docs/known-limitations.md` and `KNOWN_LIMITATIONS.md` (no fringing, no core-shape library, loss model not validated against measurement), `CHANGELOG.md`
- [x] 6.2 Tutorials: three-limb transformer; integrated coupled inductor

## 7. Verification
- [x] 7.1 `ruff check`, `pytest`, `ctest`, `openspec validate --all --strict`, stub check — plus `pytest -m reference`, golden compare (ALL IDENTICAL), `reference_status` (exit 0), [permeance] on an Eigen 3.4.0 build
