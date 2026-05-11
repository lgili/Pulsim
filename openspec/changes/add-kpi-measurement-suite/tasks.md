## 1. KPI measurement library
- [ ] 1.1 Create `benchmarks/kpi/__init__.py` exposing `compute_thd`, `compute_power_factor`, `compute_efficiency`, `compute_loss_breakdown`, `compute_transient_response`, `compute_ripple_pkpk`.
- [ ] 1.2 Implement Goertzel-based DFT for `compute_thd` (resolve fundamental + first 20 harmonics, return %THD).
- [ ] 1.3 Implement `compute_power_factor` from time-domain V·I samples (real / apparent over an integer number of fundamental periods).
- [ ] 1.4 Implement `compute_efficiency` from per-step P_in and P_out series (steady-state average over the last 20 % of samples).
- [ ] 1.5 Implement `compute_loss_breakdown(switch_states, branch_currents, branch_voltages, R_on, t_sw)` returning conduction-loss (∫i²R) and switching-loss (½·V·I·f_sw per event) per named device.
- [ ] 1.6 Implement `compute_transient_response(times, samples, target, tolerance)` → `{rise_time, settling_time, overshoot_pct, undershoot_pct}`.
- [ ] 1.7 Implement `compute_ripple_pkpk(samples)` over the steady-state window (last 20 % by default).
- [ ] 1.8 Unit tests in `benchmarks/kpi/test_kpi.py` against synthetic signals (sine, RC step) with hand-computed answers.

## 2. Runner / YAML integration
- [ ] 2.1 Extend the benchmark manifest schema: per-benchmark `kpi:` block listing metrics, observables, and optional fundamental frequency / target / sample-period.
- [ ] 2.2 In `benchmarks/benchmark_runner.py`, after the validation step, dispatch each requested KPI through the package; write results into the per-scenario JSON.
- [ ] 2.3 Surface KPIs in `results.csv` as new columns (`kpi__thd_v_in`, `kpi__pf_in`, `kpi__efficiency`, etc.).

## 3. Dashboard surfacing
- [ ] 3.1 Update `scripts/closed_loop_dashboard.py` to detect KPI columns in `results.json` and render an extra row per benchmark (rich mode + plain mode).
- [ ] 3.2 Add a `--kpi-only` flag that suppresses the max_err column and prints only the KPI summary.

## 4. Wire KPIs into existing benches
- [ ] 4.1 `boost_pfc_open_loop.yaml`: declare `thd` on input current and `power_factor` between `Vac` and `I(Vac)` — record steady-state values.
- [ ] 4.2 `cl_buck_pi.yaml`: declare `transient_response` for the first 500 µs against V_ref = 5 V.
- [ ] 4.3 `flyback_converter.yaml`: declare `efficiency` between `I(Vin)·V(vin)` and `V(out)²/Rload`.
- [ ] 4.4 `cascaded_buck_buck.yaml`: declare `efficiency` end-to-end + ripple_pkpk on V(bus) and V(out).
- [ ] 4.5 `lcc_resonant_inverter.yaml`: declare `thd` on V(out) (sine target).
- [ ] 4.6 `synchronous_boost.yaml`: declare `efficiency` + `ripple_pkpk` on V(out).

## 5. Documentation & smoke-run
- [ ] 5.1 Add a short section to `docs/` (or a top-level `KPI.md`) describing the new `kpi:` block and example output.
- [ ] 5.2 Run the full regression dashboard end-to-end; confirm all benches still pass and that the wired-up KPI values land in sensible ranges (efficiencies between 0 and 100 %, etc.).
- [ ] 5.3 Update closed-loop dashboard rich table to show KPI columns and verify visual readability.
