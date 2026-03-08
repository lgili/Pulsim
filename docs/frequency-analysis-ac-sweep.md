# Frequency Analysis (AC Sweep)

This guide defines the backend contract for frequency-domain simulation in PulsimCore.

## Scope

The backend exposes first-class AC sweep execution with deterministic outputs for:

- `open_loop_transfer`
- `closed_loop_transfer`
- `input_impedance`
- `output_impedance`

Current solver support is intentionally strict and deterministic.

## YAML Contract

Use `simulation.frequency_analysis`:

```yaml
simulation:
  frequency_analysis:
    enabled: true
    mode: open_loop_transfer     # open_loop_transfer | closed_loop_transfer | input_impedance | output_impedance
    anchor: auto                 # dc | periodic | averaged | auto
    sweep:
      scale: log                 # log | linear
      f_start_hz: 10.0
      f_stop_hz: 100000.0
      points: 80
    injection_current_amplitude: 1.0
    perturbation:
      positive: in
      negative: 0
    output:
      positive: out
      negative: 0
```

Validation is strict for:

- invalid sweep ranges
- invalid point count
- missing perturbation/output ports for selected mode
- malformed node bindings

## Python APIs

Class API:

```python
import pulsim as ps

sim = ps.Simulator(circuit, options)
ac = sim.run_frequency_analysis(options.frequency_analysis)
```

Procedural API:

```python
ac = ps.run_frequency_analysis(circuit, options.frequency_analysis)
```

Structured exception surface (optional, recommended in automation):

```python
try:
    ac = ps.run_frequency_analysis(circuit, options.frequency_analysis, raise_on_failure=True)
except ps.FrequencyAnalysisError as err:
    print(err.reason_code)         # ex.: frequency_unsupported_configuration
    print(err.diagnostic.name)     # enum name
    print(err.failed_point_index)
    print(err.failed_frequency_hz)
```

## Result Contract

`FrequencyAnalysisResult` includes:

- sweep axis: `frequency_hz`
- complex response: `response_real`, `response_imag`
- derived response: `magnitude`, `magnitude_db`, `phase_deg`
- deterministic diagnostics: `success`, `diagnostic`, `message`, `failed_point_index`, `failed_frequency_hz`
- deterministic anchor record: `anchor_mode_selected`
- stability metrics: crossover and margins plus explicit undefined reason enums
- metadata routing: `channel_metadata` for each response channel

Undefined metrics are never silently synthesized. The backend returns explicit reason tags:

- `NotTransferMode`
- `NoGainCrossover`
- `NoPhaseCrossover`

## Current Backend Limits

Current AC execution supports:

- passive linear devices (`R`, `L`, `C`)
- independent sources (`voltage_source`, `current_source`, sine/pulse/pwm source families)
- passive virtual probes/scopes

Control/logic virtual components (for example PI/PID/PWM blocks) are currently rejected in AC mode with typed diagnostics. This is deliberate to keep deterministic correctness while full small-signal mixed-domain linearization is not yet enabled.

## Frontend Responsibilities

Frontend must:

- configure AC setup UX (ports, sweep grid, mode, anchor)
- call backend AC API and consume typed arrays directly
- plot `magnitude_db` and `phase_deg` over `frequency_hz`
- display diagnostics directly from backend (`diagnostic`, `message`, failure context)

Frontend must not:

- create synthetic AC curves
- infer failure cause from regex over logs
- fabricate crossover/margin values when backend marks them undefined

## Benchmark and KPI Coverage

AC benchmark coverage includes:

- analytical linear RC low-pass AC reference (`ac_analytical` validation)
- converter/control workflow expected-failure case with typed diagnostics

AC KPI keys emitted by the benchmark tooling:

- `ac_sweep_mag_error`
- `ac_sweep_phase_error`
- `ac_runtime_p95`

## Practical Example: RC Low-Pass

```yaml
schema: pulsim-v1
version: 1
simulation:
  tstart: 0.0
  tstop: 1e-3
  dt: 1e-6
  frequency_analysis:
    enabled: true
    mode: open_loop_transfer
    anchor: dc
    sweep:
      scale: log
      f_start_hz: 10.0
      f_stop_hz: 100000.0
      points: 80
    injection_current_amplitude: 1.0
    perturbation: {positive: in, negative: 0}
    output: {positive: out, negative: 0}
components:
  - type: resistor
    name: R1
    nodes: [in, out]
    value: 1k
  - type: capacitor
    name: C1
    nodes: [out, 0]
    value: 1u
```
