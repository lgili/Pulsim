# Waveform Post-Processing

Pulsim provides a backend-owned waveform post-processing pipeline as a standard workflow layer for power-electronics simulations. This page documents the configuration contract, metric definitions (with formulas), frontend consumption rules, and migration guidance.

---

## Overview

Post-processing jobs are declared inside the `simulation.post_processing` block of a YAML netlist, or constructed programmatically via the Python API. Jobs execute after the transient simulation completes and consume the `virtual_channels` surface of `SimulationResult`.

Three job families are supported:

| Kind | YAML value | Purpose |
|---|---|---|
| **Time-domain** | `time_domain` | RMS, mean, min, max, p2p, crest, ripple, std |
| **Spectral** | `spectral` | FFT bins, harmonic table, THD |
| **Power/efficiency** | `power_efficiency` | Average power, efficiency, power factor |

---

## YAML Configuration Contract

```yaml
simulation:
  tstart: 0.0
  tstop: 5e-3
  dt: 1e-7
  post_processing:
    jobs:
      - id: output_voltage_metrics
        kind: time_domain
        signals: [V(out)]
        window:
          mode: time
          t_start: 3e-3
          t_end: 5e-3
        metrics: [rms, mean, min, max, p2p, ripple]

      - id: output_spectrum
        kind: spectral
        signals: [V(out)]
        window:
          mode: time
          t_start: 3e-3
          t_end: 5e-3
        n_harmonics: 7
        window_function: hann

      - id: converter_efficiency
        kind: power_efficiency
        input_voltage: V(Vin)
        input_current: I(Vin)
        output_voltage: V(out)
        output_current: I(Rload)
        window:
          mode: time
          t_start: 3e-3
          t_end: 5e-3
```

### Job Fields

#### Common to all jobs

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | No | Stable job identifier (auto-generated as `job_N` if omitted) |
| `kind` | string | **Yes** | `time_domain`, `spectral`, or `power_efficiency` (aliases: `power`, `efficiency`) |
| `window` | map | No | Window spec (default: full simulation time range) |

#### `time_domain` specific

| Field | Type | Default | Description |
|---|---|---|---|
| `signals` | list\[string\] or string | — | Signal names from `virtual_channels` (comma-separated string also accepted) |
| `metrics` | list\[string\] or string | all | Metrics to compute (see [Time-Domain Metrics](#time-domain-metrics)) |

#### `spectral` specific

| Field | Type | Default | Description |
|---|---|---|---|
| `signals` | list\[string\] | — | Signal names |
| `fundamental_hz` | float | auto-detect | Explicit fundamental frequency; auto-detected from spectrum peak if omitted |
| `n_harmonics` | int | 5 | Number of harmonics for THD and harmonic table |
| `window_function` | string | `rectangular` | Window function: `rectangular`, `hann`, `hamming`, `blackman`, `flat_top` |

#### `power_efficiency` specific

| Field | Type | Description |
|---|---|---|
| `input_voltage` | string | Signal name for input voltage (e.g. `V(Vin)`) |
| `input_current` | string | Signal name for input current (e.g. `I(Vin)`) |
| `output_voltage` | string | Signal name for output voltage |
| `output_current` | string | Signal name for output current |

### Window Specification

| Mode | Fields | Description |
|---|---|---|
| `time` | `t_start`, `t_end` | Physical time bounds (seconds) |
| `index` | `i_start`, `i_end`, optional `min_samples` | Sample index bounds (half-open: `[i_start, i_end)`) |
| `cycle` | `cycle_start`, `cycle_end`, `period` | Cycle-number bounds with known switching period |

---

## Time-Domain Metrics

All time-domain metrics operate on a windowed sample array $x[0], x[1], \ldots, x[N-1]$.

### RMS

$$\text{RMS} = \sqrt{\frac{1}{N} \sum_{k=0}^{N-1} x[k]^2}$$

*Power-engineering convention: includes DC component.*

### Mean

$$\bar{x} = \frac{1}{N} \sum_{k=0}^{N-1} x[k]$$

### Min / Max

$$x_{\min} = \min_k x[k], \quad x_{\max} = \max_k x[k]$$

### Peak-to-Peak (p2p)

$$\text{p2p} = x_{\max} - x_{\min}$$

### Crest Factor

$$\text{CF} = \frac{\max_k |x[k]|}{\text{RMS}}$$

*Undefined when $\text{RMS} = 0$ (or negligibly small relative to signal amplitude). Reported as `PostProcessingDiagnosticCode.UndefinedMetric`.*

### Ripple

$$\text{Ripple} = \frac{\text{p2p}}{|\bar{x}|}$$

*Power-electronics convention (dimensionless). Undefined when $|\bar{x}|$ is less than 0.1% of RMS — occurs for zero-offset AC signals.*

### Standard Deviation

$$\sigma = \sqrt{\frac{1}{N} \sum_{k=0}^{N-1} (x[k] - \bar{x})^2}$$

---

## Spectral Metrics

Spectral analysis uses numpy's one-sided real FFT (`numpy.fft.rfft`) with coherent-gain window normalization.

### Window Functions

| Name | YAML value | Formula |
|---|---|---|
| Rectangular | `rectangular` | $w[n] = 1$ |
| Hann | `hann` | $w[n] = 0.5\left(1 - \cos\!\left(\tfrac{2\pi n}{N-1}\right)\right)$ |
| Hamming | `hamming` | $w[n] = 0.54 - 0.46\cos\!\left(\tfrac{2\pi n}{N-1}\right)$ |
| Blackman | `blackman` | $w[n] = 0.42 - 0.5\cos\!\left(\tfrac{2\pi n}{N-1}\right) + 0.08\cos\!\left(\tfrac{4\pi n}{N-1}\right)$ |
| Flat-top | `flat_top` | (numpy `flattop` definition) |

### Amplitude Spectrum

After windowing and normalization, each one-sided bin $k$ has amplitude:

$$A[k] = \frac{2 |X[k]|}{N \cdot \text{CG}}$$

where $X = \text{rfft}(w \cdot x)$ and $\text{CG} = \frac{1}{N}\sum_{n=0}^{N-1} w[n]$ is the coherent gain of the window.

### Fundamental Detection

If `fundamental_hz` is not specified, the fundamental is taken as the frequency of the peak amplitude bin in the one-sided spectrum.

### Total Harmonic Distortion (THD)

$$\text{THD} = \frac{\sqrt{\sum_{h=2}^{H} A_h^2}}{A_1} \times 100\%$$

where $A_1$ is the fundamental amplitude and $A_h$ is the amplitude of the $h$-th harmonic.

*Undefined when $A_1 \approx 0$ (e.g. near-DC signal or fundamental not present in window). Reported as `PostProcessingDiagnosticCode.UndefinedMetric`.*

---

## Power and Efficiency Metrics

### Average Power

$$P_{\text{avg}} = \frac{1}{N} \sum_{k=0}^{N-1} v[k] \cdot i[k]$$

For input and output:

$$P_{\text{in}} = \overline{v_{\text{in}} \cdot i_{\text{in}}}, \quad P_{\text{out}} = \overline{v_{\text{out}} \cdot i_{\text{out}}}$$

### Efficiency

$$\eta = \frac{P_{\text{out}}}{P_{\text{in}}} \times 100\%$$

*When both $P_{\text{in}} = 0$ and $P_{\text{out}} = 0$, efficiency is reported as 100% (no-load convention). When $P_{\text{in}} = 0$ and $P_{\text{out}} \neq 0$, efficiency is undefined.*

### Power Factor

$$\text{PF} = \frac{P_{\text{avg}}}{V_{\text{rms}} \cdot I_{\text{rms}}}$$

*Computed from input signals only. Ranges in $[-1, 1]$; positive for lagging loads. Undefined when $V_{\text{rms}} \cdot I_{\text{rms}} = 0$.*

---

## Diagnostic Codes

Every job result carries a `diagnostic` field with a machine-stable code:

| Code | Meaning |
|---|---|
| `ok` | Job succeeded with no issues |
| `invalid_configuration` | Malformed job, missing required field, or unknown metric name |
| `signal_not_found` | Named signal not present in `virtual_channels` |
| `invalid_window` | Window bounds empty, reversed, or beyond simulation time |
| `insufficient_samples` | Window contains fewer samples than `min_samples` (default: 4) |
| `sampling_mismatch` | Signal sampling incompatible with spectral requirements |
| `undefined_metric` | Metric denominator is zero (crest: RMS=0; ripple: mean≈0; THD: fundamental=0) |
| `numerical_failure` | Unexpected numerical error during computation |

---

## Python API

### Configuration

```python
import pulsim as ps

opts = ps.PostProcessingOptions(jobs=[
    ps.PostProcessingJob(
        job_id="voltage_ripple",
        kind=ps.PostProcessingJobKind.TimeDomain,
        signals=["V(out)"],
        metrics=["rms", "mean", "ripple"],
        window=ps.PostProcessingWindowSpec(
            mode=ps.PostProcessingWindowMode.Time,
            t_start=3e-3,
            t_end=5e-3,
        ),
    ),
    ps.PostProcessingJob(
        job_id="spectrum",
        kind=ps.PostProcessingJobKind.Spectral,
        signals=["V(out)"],
        n_harmonics=7,
        window_function=ps.WindowFunction.Hann,
    ),
    ps.PostProcessingJob(
        job_id="efficiency",
        kind=ps.PostProcessingJobKind.PowerEfficiency,
        input_voltage_signal="V(Vin)",
        input_current_signal="I(Vin)",
        output_voltage_signal="V(out)",
        output_current_signal="I(Rload)",
    ),
])
```

### Execution

```python
result = sim.run_transient(circuit.initial_state())
pp = ps.run_post_processing(result, opts)

if not pp.success:
    for jr in pp.jobs:
        if not jr.success:
            print(f"[{jr.job_id}] FAILED: {jr.diagnostic} — {jr.diagnostic_message}")
```

### Consuming Time-Domain Results

```python
jr = pp.jobs[0]
if jr.success:
    print("RMS:   ", jr.scalar_metrics["rms"].value, jr.scalar_metrics["rms"].unit)
    print("Mean:  ", jr.scalar_metrics["mean"].value)
    print("Ripple:", jr.scalar_metrics["ripple"].value)

for undef in jr.undefined_metrics:
    print(f"Undefined: {undef.name} — {undef.reason}: {undef.message}")
```

### Consuming Spectral Results

```python
jr = pp.jobs[1]
if jr.success:
    print("Fundamental:", jr.fundamental_hz, "Hz")
    print("THD:        ", jr.thd_pct, "%")
    for h in jr.harmonics:       # HarmonicEntry(harmonic_number, frequency_hz, magnitude, ...)
        print(f"  H{h.harmonic_number}: {h.frequency_hz:.1f} Hz  mag={h.magnitude:.4f}"
              f"  ({h.magnitude_pct_fundamental:.2f}% of fundamental)")
```

### Consuming Power/Efficiency Results

```python
jr = pp.jobs[2]
if jr.success:
    print("P_in:       ", jr.average_input_power, "W")
    print("P_out:      ", jr.average_output_power, "W")
    print("Efficiency: ", jr.efficiency, "%")
    print("Power factor:", jr.power_factor)
```

### Parsing YAML from Python

```python
yaml_node = {
    "jobs": [
        {"id": "vout", "kind": "time_domain", "signals": ["V(out)"],
         "metrics": ["rms", "ripple"]}
    ]
}
errors = []
opts = ps.parse_post_processing_yaml(yaml_node, errors)
if errors:
    print("Parse errors:", errors)
```

---

## Frontend Consumption Rules

The backend post-processing pipeline is the **single source of truth** for waveform metrics. Frontend code must **not**:

- Recompute RMS, THD, efficiency, or other metrics from raw `virtual_channels` data.
- Assume a specific unit or domain for a metric — always read `ScalarMetric.unit` and `ScalarMetric.domain`.
- Silently ignore `undefined_metrics` or `diagnostic` fields; surface them to the user.
- Depend on free-form log output to extract metric values; always consume `scalar_metrics`, `harmonics`, `spectrum_bins`, `efficiency`, and related structured fields.

The backend guarantees:

- Stable, ordered job results (same order as input `PostProcessingOptions.jobs`).
- Machine-stable diagnostic codes for programmatic branching.
- Deterministic outputs across identical repeated runs.

---

## Migration from Script-Based Post-Processing

If you previously performed waveform analysis via external notebooks or scripts:

### Before (ad-hoc)

```python
# External notebook / script — ad-hoc approach
import numpy as np

result = sim.run_transient(circuit.initial_state())
v = np.array(result.virtual_channels["V(out)"])
t = np.array(result.time)

# Window selection was manual and error-prone
mask = (t >= 3e-3) & (t <= 5e-3)
rms = np.sqrt(np.mean(v[mask] ** 2))
thd = compute_thd_script(v[mask], t[mask], f0=20000)  # custom, unvalidated
```

### After (canonical backend)

```python
import pulsim as ps

# Declare once, reuse across all runs
pp_opts = ps.PostProcessingOptions(jobs=[
    ps.PostProcessingJob(
        job_id="vout",
        kind=ps.PostProcessingJobKind.TimeDomain,
        signals=["V(out)"],
        metrics=["rms"],
        window=ps.PostProcessingWindowSpec(
            mode=ps.PostProcessingWindowMode.Time,
            t_start=3e-3, t_end=5e-3,
        ),
    ),
    ps.PostProcessingJob(
        job_id="spectrum",
        kind=ps.PostProcessingJobKind.Spectral,
        signals=["V(out)"],
        fundamental_hz=20000.0,
        n_harmonics=7,
        window_function=ps.WindowFunction.Hann,
        window=ps.PostProcessingWindowSpec(
            mode=ps.PostProcessingWindowMode.Time,
            t_start=3e-3, t_end=5e-3,
        ),
    ),
])

result = sim.run_transient(circuit.initial_state())
pp = ps.run_post_processing(result, pp_opts)

rms = pp.jobs[0].scalar_metrics["rms"].value
thd = pp.jobs[1].thd_pct
```

### Migration Checklist

- [ ] Replace all `np.sqrt(np.mean(v**2))` patterns → `PostProcessingJobKind.TimeDomain` with `metrics=["rms"]`.
- [ ] Replace all manual FFT + harmonic scripts → `PostProcessingJobKind.Spectral`.
- [ ] Replace all `P_out / P_in * 100` efficiency scripts → `PostProcessingJobKind.PowerEfficiency`.
- [ ] Replace manual window slicing with `PostProcessingWindowSpec` (Time/Index/Cycle).
- [ ] Update frontend code to consume `scalar_metrics`, `thd_pct`, `efficiency` directly from `PostProcessingJobResult`.
- [ ] Remove custom metric helper functions that duplicate backend computations.

---

## Known Limitations and Deferred Items

- **Settling-time and overshoot metrics**: Not supported in this release. Planned for a future post-processing phase.
- **Loop/stability metrics** (gain margin, phase crossover): Only supported via `run_harmonic_balance` / `FrequencyAnalysisResult`. Time-domain-derived loop metric estimators are deferred.
- **Cycle-window for variable-frequency switching**: The `cycle` window mode assumes a fixed `period`. Variable-frequency (e.g. spread-spectrum) is not supported.
- **Streaming / incremental post-processing**: Jobs execute on the complete `SimulationResult`. Streaming/online metric computation is deferred.
