## 1. Frequency-domain measurement primitives
- [ ] 1.1 Create `benchmarks/frequency/__init__.py` exposing `ac_sweep`, `extract_bode_point`, `compare_to_analytical`.
- [ ] 1.2 Implement `ac_sweep(yaml, source, observable, freq_range)`: for each f_k, run a short transient with `set_ac_perturbation(source, ε, f_k, 0)`, settle, then extract magnitude/phase of the observable at f_k via Goertzel.
- [ ] 1.3 Implement `extract_bode_point` (single-frequency helper) — used internally by `ac_sweep` and also exposed for manual probing.
- [ ] 1.4 Implement `compare_to_analytical(measured_points, model_fn)` returning per-point `dB_error` and `phase_error_deg`.
- [ ] 1.5 Unit tests against a 2nd-order RC + RLC low-pass with analytical formulas.

## 2. New benchmark validation type
- [ ] 2.1 Extend manifest schema: a benchmark can declare `validation: { type: bode, source: <name>, observable: V(out), freq_range: {start, stop, points}, analytical: <python_module:fn> }`.
- [ ] 2.2 In `benchmark_runner.py`, handle the new `bode` validation type — dispatch to the frequency package, compare against the analytical callable, emit `bode__db_max_err` and `bode__phase_max_err_deg` in results.
- [ ] 2.3 Auto-discovery in `closed_loop_dashboard.py` (or new `frequency_dashboard.py`) for `validation: type: bode` entries.

## 3. Bode benchmark library
- [ ] 3.1 Create `benchmarks/frequency/models/erickson_buck.py` (analytical small-signal model from Erickson § 7.2).
- [ ] 3.2 Create `benchmarks/frequency/models/erickson_boost.py` (analytical model with RHP zero).
- [ ] 3.3 Create `benchmarks/frequency/models/rlc_low_pass.py` (simple 2nd-order Bode formula).
- [ ] 3.4 Create `benchmarks/circuits/bode_buck_plant.yaml` — open-loop buck with AC perturbation hook on duty, observable on V(out).
- [ ] 3.5 Create `benchmarks/circuits/bode_boost_plant.yaml` — same shape for boost.
- [ ] 3.6 Create `benchmarks/circuits/bode_buck_pi_loop_gain.yaml` — PI-compensated buck, recover loop gain T(s).
- [ ] 3.7 Create `benchmarks/circuits/bode_rlc_low_pass.yaml` — purely passive sanity baseline.

## 4. Margin extraction
- [ ] 4.1 Helper `extract_margins(measured_bode)` → `{gain_margin_db, phase_margin_deg, crossover_freq_hz}`.
- [ ] 4.2 Emit margins in results JSON for the loop-gain benchmark.
- [ ] 4.3 Test threshold: bode_buck_pi_loop_gain phase margin within ±5° of hand-computed value.

## 5. Wiring + smoke-run
- [ ] 5.1 Register the four new benchmarks in `benchmarks.yaml` with `scenarios: [default]`.
- [ ] 5.2 Run the full bench dashboard; confirm Bode tests appear and pass.
- [ ] 5.3 Document the workflow in `docs/` (or `BODE.md`) — how to add a new Bode benchmark with a custom analytical model.
