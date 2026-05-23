## Why
Pulsim's regression matrix (79 tests, Phases 17–22) validates V(out) waveform shape against captured baselines, but commercial simulators like PSIM and PLECS surface the metrics that power-electronics engineers actually compare in datasheets — Total Harmonic Distortion, power factor, efficiency, conduction/switching loss split, output ripple, and transient response figures.

Today every benchmark passes or fails on a single observable column. To claim measurement parity with the commercial tools we need a post-processor layer that derives quantitative KPIs from any existing simulation trace and surfaces them in the dashboard — no new circuits, just better measurements on the ones we already have.

## What Changes
- Add a `benchmarks/kpi/` Python package with reusable measurement primitives:
  - `compute_thd(samples, fundamental_hz)` — discrete Fourier + Σ harmonic energy / fundamental
  - `compute_power_factor(v_samples, i_samples)` — real power ÷ apparent power
  - `compute_efficiency(p_in_samples, p_out_samples)` — running η with steady-state average
  - `compute_loss_breakdown(switch_states, currents, voltages, params)` — conduction vs switching loss per device
  - `compute_transient_response(samples, time, target)` — settling time, overshoot, rise time, undershoot
  - `compute_ripple_pkpk(samples)` — output ripple peak-to-peak in the steady-state window
- Extend `benchmark_runner.py` so a benchmark's YAML may declare a `kpi:` block listing which metrics to compute over which observables. The metrics are written into `results.json` alongside the existing `max_error` / `rms_error`.
- Extend `scripts/closed_loop_dashboard.py` to render the KPI columns when present (rich + plain modes).
- Wire KPIs into existing benches that have natural targets:
  - `boost_pfc_open_loop` → input THD, PF, output ripple
  - `cl_buck_pi` → settling time + overshoot vs step
  - `flyback_converter` → efficiency = P_out / P_in
  - `cascaded_buck_buck` → η per stage
  - `lcc_resonant_inverter` → output THD
  - `synchronous_boost` → η + output ripple

## Impact
- Affected specs: `benchmark-suite`
- Affected code: new `benchmarks/kpi/` package, `benchmarks/benchmark_runner.py`, `scripts/closed_loop_dashboard.py`, and 5–10 existing circuit YAMLs (add `kpi:` block).
- Backward compatibility: existing benches without `kpi:` block keep their current pass/fail behavior unchanged.
