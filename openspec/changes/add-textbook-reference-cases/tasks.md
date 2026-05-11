## 1. Published-values validation type
- [ ] 1.1 Extend the benchmark schema: a YAML may declare `validation: { type: published, source: "Erickson 3rd ed § 3.1", values: { v_out: 12.0, delta_il: 1.6, delta_vc: 0.024 }, tolerance_pct: 2 }`.
- [ ] 1.2 In `benchmark_runner.py`, implement the `published` validation type: compute the relevant KPI per declared key (delegating to the KPI suite from `add-kpi-measurement-suite`), assert each is within `tolerance_pct` of the declared value.
- [ ] 1.3 Surface the per-key error in results JSON: `published__v_out_err_pct`, etc.

## 2. Reference benchmark circuits (Erickson)
- [ ] 2.1 `textbook_erickson_buck_3_1.yaml`: § 3.1 CCM buck steady-state. Cite figure 3.6, equations 3.5–3.7.
- [ ] 2.2 `textbook_erickson_boost_6_2.yaml`: § 6.2 boost averaged-model verification.
- [ ] 2.3 `textbook_erickson_buckboost_6_3.yaml`: § 6.3 inverting buck-boost.

## 3. Reference benchmark circuits (Mohan, Kassakian)
- [ ] 3.1 `textbook_mohan_pwm_rectifier_8_4.yaml`: Mohan Power Electronics § 8.4 single-phase PWM rectifier.
- [ ] 3.2 `textbook_kassakian_4_2.yaml`: KSV § 4.2 three-phase diode rectifier — match published RMS current and ripple values.

## 4. Reproducibility checks
- [ ] 4.1 Each textbook benchmark includes a comment block citing the exact edition, section, equation, and figure (or table) reference.
- [ ] 4.2 Document published numerical values inline so a reader can verify the source without opening the textbook.

## 5. Dashboard surfacing
- [ ] 5.1 `closed_loop_dashboard.py` (or a new `textbook_dashboard.py`) renders the published-values check as a separate column ("§3.1 ±2%").
- [ ] 5.2 Smoke-run: all five textbook benches pass.
- [ ] 5.3 Document the workflow in `docs/TEXTBOOK_BENCHES.md` for adding future textbook benchmarks.
