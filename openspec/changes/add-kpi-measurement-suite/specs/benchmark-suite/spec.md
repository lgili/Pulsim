## ADDED Requirements

### Requirement: Quantitative KPI Extraction
The benchmark suite SHALL compute industry-standard KPIs (THD, power factor, efficiency, loss breakdown, transient response, ripple) from any captured simulation trace and surface them alongside the existing pass/fail observable comparison.

#### Scenario: PFC boost reports input THD and power factor
- **WHEN** the `boost_pfc_open_loop` benchmark runs and declares `kpi: [thd, power_factor]` on the input current waveform
- **THEN** the runner emits `kpi__thd_in_pct` and `kpi__pf_in` columns in `results.csv`
- **AND** both values are computed over an integer number of 60 Hz periods within the steady-state window (last 20 % of samples)

#### Scenario: Closed-loop buck reports transient response
- **WHEN** the `cl_buck_pi` benchmark declares `kpi: [transient_response]` against a 5 V step reference
- **THEN** the runner emits `kpi__rise_time_us`, `kpi__settling_time_us`, `kpi__overshoot_pct`, `kpi__undershoot_pct` in the per-scenario JSON
- **AND** the metrics are derived by locating the 10 %–90 % rise window and the first time the signal stays within tolerance of the target

#### Scenario: Cascaded converter reports per-stage efficiency
- **WHEN** the `cascaded_buck_buck` benchmark declares an `efficiency` KPI between V_in·I_in and V_out·I_out
- **THEN** the runner emits `kpi__efficiency_pct` as the steady-state average of P_out / P_in
- **AND** the value is clamped to [0, 100] %; values outside that range cause the benchmark to be marked `failed` with reason `kpi_implausible`

### Requirement: KPI Dashboard Surfacing
The dashboards SHALL display computed KPI values alongside pass/fail status when a benchmark declares any KPI metric.

#### Scenario: Rich dashboard renders KPI columns
- **WHEN** running `scripts/closed_loop_dashboard.py` on a manifest containing benches with `kpi:` blocks
- **THEN** the rendered table shows one extra column per declared KPI type (e.g. `THD %`, `PF`, `η %`, `settling µs`)
- **AND** benches without any `kpi:` block render `—` in those columns
- **AND** the `--kpi-only` flag suppresses the `max_err` / `thr_max` columns and prints only the KPI summary
