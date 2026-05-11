## 1. DC brush motor (simplest, sanity baseline)
- [ ] 1.1 Create `motor_dc_brush_step_load.yaml`: voltage source → R_a → L_a → back_EMF (= K·ω modeled as DC source) → return. Add a switchable load resistor that toggles at t = 5 ms.
- [ ] 1.2 Capture baseline of i_armature(t).
- [ ] 1.3 Analytical check: steady-state current = (V − V_BE)/R_a within 2 %.

## 2. PMSM in dq frame
- [ ] 2.1 Create `motor_pmsm_dq_open_loop.yaml`: V_d and V_q as DC voltage sources, L_d and L_q as inductors with coupling λ_pm·ω modeled as an extra voltage source on the q-axis.
- [ ] 2.2 Apply constant V_d = 0, V_q = nominal; capture i_d, i_q reaching steady state.
- [ ] 2.3 Analytical check: steady-state i_q = (V_q − ω·λ_pm) / R_s within 2 %.
- [ ] 2.4 Add `kpi: [torque]` computing T_e = (3/2)·p·λ_pm·i_q from the measured i_q.

## 3. BLDC six-step
- [ ] 3.1 Create `motor_bldc_six_step.yaml`: three phases with trapezoidal back-EMF (each phase a `pulse` source phase-shifted 120°), driven by six vcswitches in a 3-phase inverter.
- [ ] 3.2 Sequence the gates manually (6 pwm voltage sources) to produce the six-step commutation pattern.
- [ ] 3.3 Capture baselines of i_a, i_b, i_c.
- [ ] 3.4 KPI: current THD per phase, torque ripple % over one electrical period.

## 4. Three-phase induction (locked rotor)
- [ ] 4.1 Create `motor_induction_locked_rotor.yaml`: three sine sources at 60 Hz, 120° apart, driving a Y-connected stator (3 R_s + L_ls) with magnetizing branch L_m and locked-rotor secondary (R_r/s with s = 1, so R_r).
- [ ] 4.2 Capture baseline of i_stator phase A.
- [ ] 4.3 Analytical check: inrush current matches per-phase impedance √(R² + ω²L²) within 3 %.

## 5. Wiring + dashboard
- [ ] 5.1 Register all four motor benchmarks in `benchmarks.yaml` under `scenarios: [default]` with `validation: type: reference`.
- [ ] 5.2 Generate baselines via `--generate-baselines`.
- [ ] 5.3 Confirm `closed_loop_dashboard.py` auto-discovers and runs them.
- [ ] 5.4 Document the dq-frame modeling convention used (in `docs/` or `MOTOR_MODELS.md`) so other developers can extend with new motor types.
