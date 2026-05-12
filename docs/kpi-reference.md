# KPI Reference

> **Status:** authoritative — every metric `benchmark_runner.py` can
> compute against a captured trace. Source: `benchmarks/kpi/__init__.py`
> (computation) and `benchmarks/benchmark_runner.py` (YAML wiring).
> Updated through Phase 27.

Pulsim ships a trace-scoring layer that runs *after* the simulation and
turns a captured CSV into engineering KPIs — THD, ripple, efficiency,
ZVS/ZCS fractions, core loss, junction temperature, etc. The functions
are pure Python with no Pulsim runtime dependency, so the same module
can score Pulsim traces, PSIM / PLECS exports, or an oscilloscope CSV.

## How KPIs land in benchmark results

Each `circuits/<bench>.yaml` may declare a `benchmark.kpi:` list. The
runner evaluates every entry against the trace and writes them as
`kpi__<metric>__<label>` columns in `benchmarks/.../results.csv`. KPI
thresholds (gate values used by `kpi_gate.py`) live in
`benchmarks/kpi_thresholds.yaml`.

```yaml
benchmark:
  id: my_buck
  category: closed_loop
  kpi:
    - metric: ripple_pkpk
      observable: V(out)
      label: vout
      steady_state_fraction: 0.2
    - metric: transient_response
      observable: V(out)
      label: vout
      target: 12.0
      tolerance_pct: 2.0
    - metric: thd
      observable: I(L1)
      label: iL
      fundamental_hz: 60.0
      num_harmonics: 20
```

The `label` becomes the suffix on the CSV column — `kpi__ripple_pkpk__vout`,
`kpi__rise_time__vout`, etc.

## Quick index

| Metric | YAML keyword | Returns | Use case |
|---|---|---|---|
| Total harmonic distortion | `thd` | `thd_pct` | Sine quality, EMI compliance |
| Power factor | `power_factor` | `pf` | AC source loading |
| Efficiency | `efficiency` | `efficiency_pct` | Converter η |
| Transient response | `transient_response` | `rise_time`, `settling_time`, `overshoot_pct`, `undershoot_pct` | Step response analysis |
| Peak-to-peak ripple | `ripple_pkpk` | `ripple_pkpk` | Output filter sizing |
| Conduction + switching loss | `loss_breakdown` | `conduction_w_avg`, `switching_w_avg` | Loss budgeting |
| ZVS fraction | `zvs_fraction` | `zvs_fraction`, `total_turn_on_events` | Soft-switching validation |
| ZCS fraction | `zcs_fraction` | `zcs_fraction`, `total_turn_off_events` | Soft-switching validation |
| Per-event switching loss | `switching_loss` | `switching_loss_w_avg`, `switching_event_count` | High-frequency loss |
| Steinmetz core loss | `core_loss_steinmetz` | `core_loss_w_per_kg`, `b_peak_tesla` | Magnetic sizing |
| Junction temperature | `junction_temperature` | `t_j_max_c`, `t_j_final_c`, `delta_t_j_c` | Thermal margin |

---

## 1. Total Harmonic Distortion — `thd`

**Function:** `compute_thd(samples, sample_rate_hz, fundamental_hz, num_harmonics=20)`
→ percentage.

`THD% = 100 · √(Σ_{k=2..N} A_k²) / A_1` where `A_k` is the magnitude of
the k-th harmonic recovered via Goertzel. The window should cover an
integer number of fundamental periods for a clean reading.

```yaml
- metric: thd
  observable: V(out)
  label: vout
  fundamental_hz: 60.0
  num_harmonics: 20
```

| YAML key | Default | Meaning |
|---|---|---|
| `observable` | — | CSV column to FFT. |
| `fundamental_hz` | `60.0` | Hz of the fundamental. |
| `num_harmonics` | `20` | Harmonics to sum (k = 2..N). |
| `label` | `metric` | Suffix in the result column. |

Outputs: `kpi__thd_pct__<label>`.

---

## 2. Power factor — `power_factor`

**Function:** `compute_power_factor(v_samples, i_samples)` → `[-1, 1]`.

`PF = ⟨v·i⟩ / (RMS(v)·RMS(i))`. Sign tracks real-power direction.

```yaml
- metric: power_factor
  observable: V(grid)
  current_observable: I(grid)
  label: grid
```

Outputs: `kpi__pf__<label>`.

---

## 3. Efficiency — `efficiency`

**Function:** `compute_efficiency(p_in, p_out, steady_state_fraction=0.2)`
→ percent.

`η = 100 · ⟨P_out⟩ / ⟨P_in⟩` averaged over the **last fraction** of the
trace, so transient warm-up doesn't pull the number down.

```yaml
- metric: efficiency
  observable: P_in
  p_in_observable: P_in
  p_out_observable: P_out
  label: converter
  steady_state_fraction: 0.2
```

| YAML key | Default | Meaning |
|---|---|---|
| `p_in_observable` | — | Column for input power (W). |
| `p_out_observable` | — | Column for output power (W). |
| `steady_state_fraction` | `0.2` | Window for the average (last fraction). |

Outputs: `kpi__efficiency_pct__<label>`.

---

## 4. Transient response — `transient_response`

**Function:** `compute_transient_response(times, samples, target, tolerance_pct=2, rise_low_pct=10, rise_high_pct=90)`
→ dict.

```yaml
- metric: transient_response
  observable: V(out)
  label: vout
  target: 12.0
  tolerance_pct: 2.0
```

| YAML key | Default | Meaning |
|---|---|---|
| `target` | `0.0` | Steady-state target value. |
| `tolerance_pct` | `2.0` | Settling band (±%). |

Outputs (each populated when measurable):
- `kpi__rise_time__<label>` — `rise_low_pct%` → `rise_high_pct%` (default 10–90%).
- `kpi__settling_time__<label>` — time after which the trace stays inside the band.
- `kpi__overshoot_pct__<label>` — `100 · (peak − target) / |target|`.
- `kpi__undershoot_pct__<label>` — `100 · (target − trough) / |target|`.

---

## 5. Peak-to-peak ripple — `ripple_pkpk`

**Function:** `compute_ripple_pkpk(samples, steady_state_fraction=0.2)`.

```yaml
- metric: ripple_pkpk
  observable: V(out)
  label: vout
  steady_state_fraction: 0.2
```

Outputs: `kpi__ripple_pkpk__<label>` in the units of the observable.

---

## 6. Conduction + switching loss — `loss_breakdown`

**Function:** `compute_loss_breakdown(switch_states, branch_currents, branch_voltages, r_on, times)`
→ `{conduction_w_avg, switching_w_avg}`.

Conduction = `R_on · I²` integrated while ON. Switching = `½ · |V·I|`
at every state transition. Both divided by total simulated time.

```yaml
- metric: loss_breakdown
  observable: SW1.state            # ignored; uses switch_observable
  switch_observable: SW1.state
  current_observable: I(SW1)
  voltage_observable: V(sw)
  r_on: 5e-3
  label: SW1
```

Outputs: `kpi__conduction_w_avg__<label>`, `kpi__switching_w_avg__<label>`.

---

## 7. ZVS fraction — `zvs_fraction`

**Function:** `compute_zvs_fraction(switch_states, v_ds_samples, threshold_v=1.0, lookback_samples=1)`.

For every `False → True` (turn-ON) transition, looks `lookback_samples`
back at V_DS. If `|V_DS| < threshold_v`, the event counts as ZVS.

```yaml
- metric: zvs_fraction
  switch_observable: SH.state
  voltage_observable: V(sh)
  threshold_v: 1.0
  lookback_samples: 1
  label: SH
```

Outputs: `kpi__zvs_fraction__<label>` (0..1), `kpi__total_turn_on_events__<label>`.

---

## 8. ZCS fraction — `zcs_fraction`

**Function:** `compute_zcs_fraction(switch_states, i_d_samples, threshold_a=0.1, lookback_samples=1)`.

Mirror of ZVS but at turn-OFF and on drain current.

```yaml
- metric: zcs_fraction
  switch_observable: SH.state
  current_observable: I(L1)
  threshold_a: 0.1
  lookback_samples: 1
  label: SH
```

Outputs: `kpi__zcs_fraction__<label>`, `kpi__total_turn_off_events__<label>`.

---

## 9. Per-event switching loss — `switching_loss`

**Function:** `compute_switching_loss_per_event(switch_states, v_ds, i_d, times)`.

Sums `½·|V_DS·I_D|` at every state transition and divides by total time.

```yaml
- metric: switching_loss
  switch_observable: SH.state
  voltage_observable: V(sh)
  current_observable: I(L1)
  label: SH
```

Outputs: `kpi__switching_loss_w_avg__<label>`, `kpi__switching_event_count__<label>`.

---

## 10. Steinmetz core loss — `core_loss_steinmetz`

**Function:** `compute_core_loss_steinmetz(B_samples, sample_rate_hz, fundamental_hz, k, alpha, beta)`.

Recovers B_peak via Goertzel, then `P = k · f^α · B^β`. Pulsim ships a
companion helper `compute_inductor_flux_density(I_samples, L, turns, area_m²)`
to convert an inductor current trace into B(t).

```yaml
- metric: core_loss_steinmetz
  observable: I(L1)               # inductor current → B via L·I / (N·A)
  inductance_h: 500e-6
  turns: 60
  area_m2: 1.5e-4
  fundamental_hz: 60.0
  k: 16.0
  alpha: 1.45
  beta: 2.7
  label: ferrite
```

| YAML key | Default | Meaning |
|---|---|---|
| `inductance_h` | `1e-3` | Inductance for the L·I conversion (H). |
| `turns` | `50` | Number of turns. |
| `area_m2` | `1e-4` | Effective core cross-section (m²). |
| `fundamental_hz` | `60.0` | Operating frequency (Hz). |
| `k`, `alpha`, `beta` | `16, 1.45, 2.7` | Steinmetz coefficients for the material. |

Outputs: `kpi__core_loss_w_per_kg__<label>`, `kpi__b_peak_tesla__<label>`.

---

## 11. Junction temperature — `junction_temperature`

**Function:** `compute_junction_temperature(P_samples, times, r_th_jc, c_th_jc, t_ambient_c=25, r_th_ca=0)`.

Integrates `C_th · dT_j/dt + (T_j − T_amb)/R_th = P(t)` forward in time
(Foster network, one or many stages chained).

The YAML wiring derives P(t) from an observed voltage trace through a
resistance — useful for self-heating studies:

```yaml
- metric: junction_temperature
  observable: V(r_diss)            # voltage across the dissipating element
  r_resistor: 1.0                  # to derive P(t) = V²/R
  r_th_jc: 5.0                     # K/W
  c_th_jc: 0.1                     # J/K  (use 0 for steady-state-only)
  t_ambient_c: 25.0
  r_th_ca: 0.0                     # heatsink-to-ambient (K/W), optional
  label: r_diss
```

| YAML key | Default | Meaning |
|---|---|---|
| `r_resistor` | `1.0` | Resistance used to convert V² → P (Ω). |
| `r_th_jc` | `5.0` | Junction-to-case thermal resistance (K/W). |
| `c_th_jc` | `0.1` | Junction-to-case thermal capacitance (J/K). |
| `t_ambient_c` | `25.0` | Ambient temperature (°C). |
| `r_th_ca` | `0.0` | Case-to-ambient thermal resistance (K/W). |

Outputs: `kpi__t_j_max_c__<label>`, `kpi__t_j_final_c__<label>`,
`kpi__delta_t_j_c__<label>`.

---

## Calling the helpers directly (Python)

The same functions are importable from Python without going through the
runner — useful for one-off scoring of an oscilloscope CSV or a PSIM
export:

```python
from benchmarks.kpi import (
    compute_thd,
    compute_ripple_pkpk,
    compute_transient_response,
    compute_zvs_fraction,
    compute_junction_temperature,
    compute_inductor_flux_density,
    compute_core_loss_steinmetz,
)

# THD on a captured current trace
thd_pct = compute_thd(i_load, sample_rate_hz=1e5, fundamental_hz=60.0)

# Junction temperature from a resistor's V trace
import benchmarks.kpi as kpi
P = kpi.compute_power_dissipation_resistor(v_a, v_b, resistance=1.0)
result = compute_junction_temperature(
    P, times, r_th_jc=5.0, c_th_jc=0.1, t_ambient_c=25.0,
)
print(result["t_j_max_c"], "°C peak")
```

Every helper is pure Python (no Pulsim runtime) — they can be vendored
into a separate tool.

---

## KPI gating

The `benchmarks/kpi_gate.py` script reads `benchmarks/kpi_thresholds.yaml`
and fails a CI run if any KPI exceeds (or falls below) its threshold. The
threshold file follows the same `kpi__<metric>__<label>` naming as the
results CSV:

```yaml
# kpi_thresholds.yaml
gates:
  kpi__ripple_pkpk__vout:
    max: 0.5
  kpi__rise_time__vout:
    max: 5e-3
  kpi__zvs_fraction__SH:
    min: 0.95
  kpi__t_j_max_c__r_diss:
    max: 150.0
```

Run it after `benchmark_runner.py`:

```bash
python3 benchmarks/benchmark_runner.py --benchmarks benchmarks/benchmarks.yaml \
                                       --output-dir benchmarks/out
python3 benchmarks/kpi_gate.py        --results benchmarks/out/results.csv \
                                       --thresholds benchmarks/kpi_thresholds.yaml
```

The companion file `benchmarks/kpi_thresholds_electrothermal.yaml`
holds the dedicated thermal-margin gates used by the Phase 27
electrothermal benches.

---

## See also

- [Components Reference](components-reference.md) — for the components
  that produce the signals your KPIs read.
- [Control Blocks Reference](control-blocks-reference.md) — for the
  channels (`chan:PLL.theta`, `chan:SVM.d_a`, etc.) you can score.
- [Magnetic Models](magnetic-models.md) — Steinmetz coefficients and
  flux-density definitions.
- [Electrothermal Workflow](electrothermal-workflow.md) — the full thermal
  port and how to wire its outputs into `junction_temperature`.
- [Benchmarks and Parity](benchmarks-and-parity.md) — runner CLI flags,
  baseline regeneration, dashboard outputs.
