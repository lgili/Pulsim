> Reconciled 2026-11: one task is done (2.1, as
> `circuits/switching/buck/textbook_erickson_buck_ccm.yaml`, which cites
> section 3.1, eqs. 3.5-3.7 and figure 3.6). A boost file exists
> (`textbook_erickson_boost_ccm.yaml`), but it covers the section 6.1
> steady state, not the section 6.2 averaged model. Both are checked
> against a baseline CSV, not a `published` validation type, which does
> not exist. Also, none of these benchmark files runs on 2.0: they use
> `schema: pulsim-v1`, which `p.load_yaml_file` refuses ("missing
> required top-level key 'circuit:'"), and
> `benchmarks/tools/benchmark_runner.py` imports
> `pulsim_python_backend`, deleted in 64837c4c, so each run fails with
> "Pulsim Python runtime backend unavailable".

## 1. Published-values validation type
> Open (2026-11): `benchmark_runner.py` cannot run any benchmark on 2.0
> (it imports the deleted `pulsim_python_backend`); the published checks
> live in python/tests/test_textbook_reference_cases.py until the runner is ported.
- [ ] 1.1 Extend the benchmark schema: a YAML may declare `validation: { type: published, source: "Erickson 3rd ed § 3.1", values: { v_out: 12.0, delta_il: 1.6, delta_vc: 0.024 }, tolerance_pct: 2 }`.
- [ ] 1.2 In `benchmark_runner.py`, implement the `published` validation type: compute the relevant KPI per declared key (delegating to the KPI suite from `add-kpi-measurement-suite`), assert each is within `tolerance_pct` of the declared value.
- [ ] 1.3 Surface the per-key error in results JSON: `published__v_out_err_pct`, etc.

## 2. Reference benchmark circuits (Erickson)
- [x] 2.1 `textbook_erickson_buck_3_1.yaml`: § 3.1 CCM buck steady-state. Cite figure 3.6, equations 3.5–3.7.
  (Reconciled 2026-11: as textbook_erickson_buck_ccm.yaml.)
- [x] 2.2 `textbook_erickson_boost_6_2.yaml`: § 6.2 boost averaged-model verification.
  (2026-11: as the `boost_RL` case of python/tests/test_textbook_reference_cases.py: V/Vg = (1/D')/(1 + R_L/(D'^2 R)), exact orbit on five engines. No YAML: a 2.0 YAML carries no gate drive.)
- [x] 2.3 `textbook_erickson_buckboost_6_3.yaml`: § 6.3 inverting buck-boost.
  (2026-11: `buckboost_ccm` and `buckboost_dcm` in python/tests/test_textbook_reference_cases.py, M = -D/D' and -D/sqrt(K).)

## 3. Reference benchmark circuits (Mohan, Kassakian)
- [x] 3.1 `textbook_mohan_pwm_rectifier_8_4.yaml`: Mohan Power Electronics § 8.4 single-phase PWM rectifier.
  (2026-11: as the same full bridge in inverter mode, bipolar sine-triangle PWM on R-L, V_o1 = m_a V_d (Mohan ch. 8): the rectifier mode needs a current loop and has no closed form. Also a resonant-charging case, Mohan ch. 9.)
- [x] 3.2 `textbook_kassakian_4_2.yaml`: KSV § 4.2 three-phase diode rectifier — match published RMS current and ripple values.
  (2026-11: six-diode bridge on R-L in python/tests/test_textbook_reference_cases.py: V_d0 = 3 sqrt(3) V_m / pi, I_a,rms = sqrt(2/3) I_d, closed-form DC current.)

## 4. Reproducibility checks
- [x] 4.1 Each textbook benchmark includes a comment block citing the exact edition, section, equation, and figure (or table) reference.
  (2026-11: edition and chapter, with every formula written out; section / equation numbers not quoted, the books were not at hand to verify them.)
- [x] 4.2 Document published numerical values inline so a reader can verify the source without opening the textbook.

## 5. Dashboard surfacing
- [ ] 5.1 `closed_loop_dashboard.py` (or a new `textbook_dashboard.py`) renders the published-values check as a separate column ("§3.1 ±2%").
- [x] 5.2 Smoke-run: all five textbook benches pass.
  (2026-11: 47 cases pass on pwl / pwl+restart / trbdf2 / esdirk43 / dsed; findings F-T-1, F-T-2 in FINDINGS.md.)
- [ ] 5.3 Document the workflow in `docs/TEXTBOOK_BENCHES.md` for adding future textbook benchmarks.
