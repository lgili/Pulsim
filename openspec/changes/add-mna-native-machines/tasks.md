## 1. DC machine composite (today's structure)
- [x] 1.1 `CircuitBuilder::add_dc_machine` (sub-elements `{name}.ra`, `{name}.la`, `{name}.em`, `{name}.j`, `{name}.b`, `{name}.tl`); validation by name (`R_a ≥ 0`, `L_a > 0`, `K > 0`, `J > 0`, `B ≥ 0`) — as a Python method installed on `CircuitBuilder` (`python/pulsim/machines.py`), DECISIONS D1; `{name}.ra`/`.b`/`.tl` only when non-zero (D6)
- [ ] 1.2 Binding, stubs, `components()` entry `dc_machine`; YAML type `dc_machine` — PARTIAL: Python method + stubs, YAML type `dc_machine` (loader expansion pinned equal to the Python one); NO `components()` composite entry (DECISIONS D2)
- [x] 1.3 Test (accuracy): the `p5b_dcmotor.py` case at dt = 12.5 µs, speed error against the exact solution ≤ 1e-4 (measured 7.45e-5) — measured 7.45e-5
- [x] 1.4 Test (order): errors at 100, 50, 25, 12.5 µs fall by ≥ 3.5 per halving (measured 4.76e-3, 1.19e-3, 2.98e-4, 7.45e-5) — measured 4.76e-3, 1.19e-3, 2.98e-4, 7.45e-5 (ratios 3.99, 4.00, 4.00)
- [x] 1.5 Test (DC operating point): 10 V, `K = 0.1`, `R_a = 0.5 Ω`, `T_load = 0.2 N·m` → `i = 2 A`, `ω = 90 rad/s` within 1e-9 — preflight false positive found and fixed (DECISIONS D9)
- [x] 1.6 Test (`trbdf2`): same case, error ≤ 2e-4 (measured 9.2e-5); energy check `∫v·i dt = ∫R i² dt + ΔE_L + ΔE_J` within 1e-6 — measured 9.19e-5; energy 4.2e-9 on `pwl` (bound 1e-8), 5.5e-6 on `trbdf2` (bound 2e-5, not 1e-6: DECISIONS D8)
- [x] 1.7 Test: a legacy call with `Kt ≠ Ke` routed to the native path is refused with the SI explanation
- [ ] 1.8 Optional `theta_node` through the VCCS of `add-controlled-source-set`: `θ(t) = ∫ω dt` within 1e-6 — NOT DONE: refused by name until the VCCS exists (DECISIONS D5)
- [x] 1.9 AC check: armature admittance of the composite against the analytic `1/(R_a + sL_a + K²/(sJ + B))` at 10 frequencies within 1e-6 (after `fix-mna-sweep-correctness` if the sweep needs it) — kernel complex MNA sweep, measured 1.9e-12
- [ ] 1.10 Conformance rows (`add-engine-conformance-matrix`): `dc_machine` × {pwl ok, trbdf2 ok, dsed refused, DC ok, AC ok} — NOT DONE: the conformance matrix does not exist yet

## 2. Induction machine (after `add-unified-device-interface`)
- [ ] 2.1 `models/induction_machine_mna.hpp`: parameters, validation, `eval` with f and q for 3 stator rows, 2 rotor-flux rows and the speed-node torque; closed-form Jacobian
- [ ] 2.2 C++ test: Jacobian against central differences at 100 random states (1e-7); power balance `Σ v_k i_k = R_s|i|² + R_r|i_r|² + dW/dt + T_em·ω` within 1e-9
- [ ] 2.3 `add_induction_machine`, binding, stubs, YAML type `induction_machine`; `induction_motor` gains `model: mna | observer`
- [ ] 2.4 Tests with the declared test machine (0.5 Ω, 2 mH, 0.4 Ω, 2 mH, 60 mH, pp = 2; 100 V rms phase, 60 Hz), speed node held by a source: locked rotor 57.88 A rms; synchronous speed 4.277 A rms and |mean torque| ≤ 0.01 N·m; slip 0.05 → 16.12 N·m and 12.29 A rms; each within 0.5 %
- [ ] 2.5 Free acceleration against an independent SciPy integration (rtol 1e-10): 1e-4 on `trbdf2`; order ≥ 3.5 per halving on `pwl`
- [ ] 2.6 Engine agreement `pwl` / `trbdf2` within 1e-3; inverter-fed case (PWL bridge + machine)
- [ ] 2.7 Conformance rows: `induction_machine` × {pwl, trbdf2, dsed refused, DC held, AC}

## 3. PM-flux harmonics and BLDC
- [ ] 3.1 `PmsmMna::Params` harmonic list `(h, ψ_h, φ_h)`; `λ_pm`, `dλ_pm/dθ`, `d²λ_pm/dθ²` from the series; `ψ_1` only is bit-identical to the current device
- [ ] 3.2 `add_bldc_mna(..., k_e, flat_top_deg=120, n_harmonics=25)` and `pm_harmonics_from_waveform(theta, emf, omega)`
- [ ] 3.3 Test: open-circuit back-EMF at constant speed against the ideal trapezoid, RMS deviation ≤ 0.5 % of the flat top at 25 harmonics (computed 0.40 %)
- [ ] 3.4 Test: torque from `∫ Σ e_k i_k dt / (ω·t)` equals the device's `T_em` mean within 1e-6 (co-energy consistency)
- [ ] 3.5 DC operating point with held mechanics for `PmsmMna` (today: fails, no DC stamp): at `θ0`, `ω0 = 0`, `i = v/R_s` within 1e-9
- [ ] 3.6 YAML types `pmsm`, `bldc`
- [ ] 3.7 Conformance rows: `pmsm_mna[harmonics]`, `pmsm_mna[flux_map]` × {pwl, trbdf2, dsed refused, DC held, AC}

## 4. Flux maps
- [ ] 4.1 `FluxMapDQ.from_tables(i_d, i_q, lambda_d, lambda_q, *, tol=0.02, smooth=0.0, source=...)`: co-energy bicubic fit, residual, positive-definite check, refusals by name
- [ ] 4.2 Kernel evaluation of `∇W'` and the Hessian; picoampere-spacing monotonicity test along both axes
- [ ] 4.3 Test: a map generated from `L_d = 1 mH`, `L_q = 3 mH`, `ψ_pm = 0.05 Wb` reproduces the linear `PmsmMna` run within 1e-9
- [ ] 4.4 Test: reciprocity at 100 random points within 1e-9; a non-reciprocal input (one table scaled by 1.1) is refused naming the residual
- [ ] 4.5 Test: a saturating map (analytic `λ_d = ψ_pm + L_d0·I_s·tanh(i_d/I_s)`) — locked-rotor current step against an independent SciPy integration, 1e-4

## 5. Deprecation
- [ ] 5.1 `DeprecationWarning` in `add_dc_motor`, `add_pmsm`, `add_bldc`, `add_induction_motor` and the `make_*_observer` functions, naming the native replacement and the measured first-order error — PARTIAL: `add_dc_motor` and `make_dc_motor_observer` only (DECISIONS D4)
- [ ] 5.2 `im_parameters_from_nameplate` result labelled `estimated`
- [ ] 5.3 Next release: YAML `induction_motor` default `model: mna` (BREAKING, CHANGELOG)

## 6. Documentation
- [ ] 6.1 `AI_GUIDE.md` (machine table: native versus observer, which engine), `docs/motors-induction.md`, `docs/known-limitations.md` and `KNOWN_LIMITATIONS.md` (non-goals; no machine is validated against measurement), `CHANGELOG.md` — PARTIAL (DC machine): AI_GUIDE row, motors-induction see-also, both limitation pages, CHANGELOG
- [x] 6.2 Tutorial: DC machine as a circuit (the speed node, loads as elements) — `docs/tutorials/07-dc-machine.md` (executed by test_doc_snippets)

## 7. Verification
- [ ] 7.1 `ruff check`, `pytest`, `ctest`, `openspec validate --all --strict`, stub check
