# Benchmarks and Parity

Use these workflows to guard runtime, numerical stability, and external parity.

## 1) Standard Benchmark Suite

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_runner.py \
  --output-dir benchmarks/out
```

For switching-heavy converter cases:

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_runner.py \
  --only buck_switching boost_switching_complex interleaved_buck_3ph buck_mosfet_nonlinear \
  --output-dir benchmarks/out_converters
```

## 2) Solver/Integrator Validation Matrix

```bash
PYTHONPATH=build/python python3 benchmarks/validation_matrix.py \
  --output-dir benchmarks/matrix
```

This matrix is useful to detect solver regressions across fixed/variable timestep modes.

## 3) External Parity

### ngspice

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_ngspice.py \
  --backend ngspice \
  --output-dir benchmarks/ngspice_out
```

### LTspice

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_ngspice.py \
  --backend ltspice \
  --ltspice-exe "/Applications/LTspice.app/Contents/MacOS/LTspice" \
  --output-dir benchmarks/ltspice_out
```

## 4) Tiered Stress Suite

```bash
PYTHONPATH=build/python python3 benchmarks/stress_suite.py \
  --output-dir benchmarks/stress_out
```

Electro-thermal stress variant:

```bash
PYTHONPATH=build/python python3 benchmarks/stress_suite.py \
  --benchmarks benchmarks/electrothermal_benchmarks.yaml \
  --catalog benchmarks/electrothermal_stress_catalog.yaml \
  --output-dir benchmarks/stress_out_electrothermal
```

## 5) GUI-to-Backend Parity Gate

```bash
PYTHONPATH=build/python pytest -q python/tests/test_gui_component_parity.py
PYTHONPATH=build/python pytest -q python/tests/test_runtime_bindings.py
./build-test/core/pulsim_simulation_tests "[v1][yaml][gui-parity]"
```

## 6) KPI Regression Gate

```bash
python3 benchmarks/kpi_gate.py \
  --bench-results benchmarks/out/results.json \
  --stress-summary benchmarks/stress_out/stress_summary.json \
  --report-out benchmarks/out/kpi_gate_report.json \
  --print-report
```

Baseline and threshold files:

- `benchmarks/kpi_baselines/phase0_2026-02-23/kpi_baseline.json`
- `benchmarks/kpi_thresholds.yaml`

## Artifact Contract

These files are the recommended CI contract:

- benchmark: `results.csv`, `results.json`, `summary.json`
- parity: `parity_results.csv`, `parity_results.json`, `parity_summary.json`
- stress: `stress_results.csv`, `stress_results.json`, `stress_summary.json`

Primary hybrid/electro-thermal KPI keys emitted in benchmark outputs:

- `state_space_primary_ratio`
- `dae_fallback_ratio`
- `loss_energy_balance_error`
- `thermal_peak_temperature_delta`
- `component_coverage_rate`
- `component_coverage_gap`
- `component_loss_summary_consistency_error`
- `component_thermal_summary_consistency_error`

## Soft-switching benchmarks (Phase 24)

Phase 24 added ZVS/ZCS detection and the dedicated soft-switching
benchmarks `lcc_resonant_inverter`, `active_clamp_forward`,
`half_bridge_inverter_lc`, and others under the `closed_loop` category.
Each declares a `kpi:` block of `zvs_fraction` / `zcs_fraction` /
`switching_loss` metrics — the runner walks the captured switch state +
V_DS / I_D traces and reports the fraction of soft commutations.

```yaml
benchmark:
  id: lcc_resonant_inverter
  kpi:
    - metric: zvs_fraction
      switch_observable: SH.state
      voltage_observable: V(sh)
      label: SH
      threshold_v: 1.0
    - metric: switching_loss
      switch_observable: SH.state
      voltage_observable: V(sh)
      current_observable: I(L_res)
      label: SH
```

KPI gates in `benchmarks/kpi_thresholds.yaml` typically set
`kpi__zvs_fraction__SH: { min: 0.95 }` so soft-switching regressions
trip CI. See [KPI Reference](kpi-reference.md#7-zvs-fraction-zvs_fraction).

## Long-duration / numerical-stress benchmarks (Phase 25)

Phase 25 added a small suite of benches designed to surface
long-horizon numerical drift, high-frequency loss accumulation, and
multi-cell modular topologies that would otherwise hide in a "happy
path" regression. They live in the standard `closed_loop` dashboard
and run alongside everything else:

| Benchmark | Why it exists |
|---|---|
| `long_run_drift_buck` | 1+ s buck simulation to catch conservation drift. |
| `high_freq_gan_buck` | 1 MHz GaN switching — surfaces high-frequency loss bookkeeping bugs. |
| `mmc_4cell_chain` | 4-cell modular multilevel converter — stresses event scheduling. |
| `stiff_rc_high_freq_switching` | very stiff RC + 1 MHz switching combo for the linear-solver cache. |
| `cascaded_h_bridge_5level` | 5-level cascaded H-bridge — multi-source orchestration. |
| `t_type_3level_halfbridge`, `npc_three_level_halfbridge`, `flying_capacitor_3level` | Multi-level inverter variants. |

These benches all gate on the standard `max_error` /
`steady_state_max_error` tolerances plus the conservation-error
columns in `results.csv`. They've been the regression net that caught
several silent drift bugs in TR-BDF2 corner cases.
