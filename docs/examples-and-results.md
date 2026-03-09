# Examples and Results

This page focuses on practical backend runs with expected outputs and validation criteria.

## Example 1: RC Step (sanity baseline)

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_runner.py \
  --only rc_step \
  --output-dir benchmarks/out_rc
```

Expected behavior:

- `V(out)` rises exponentially.
- very low analytical error for RC waveform.
- low rejection count and no unstable fallback loop.

## Example 2: Buck Converter (switching transient)

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_runner.py \
  --only buck_switching \
  --output-dir benchmarks/out_buck
```

Checkpoints:

- output settles around expected duty-dependent value.
- ripple remains physically plausible for chosen L/C and switching frequency.
- telemetry does not show runaway fallback escalation.

## Example 2a: Buck Averaged Pair (switching reference + averaged mode)

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_runner.py \
  --only buck_switching_paired buck_averaged_mvp buck_averaged_expected_failure \
  --output-dir benchmarks/out_buck_averaged
```

Checkpoints:

- both pair runs (`buck_switching_paired`, `buck_averaged_mvp`) are `passed`.
- `buck_averaged_mvp` emits `max_error` from `paired_reference` validation.
- expected-failure case matches typed diagnostic (`AveragedInvalidConfiguration`).

Runnable Python example:

```bash
PYTHONPATH=build/python python3 examples/run_buck_averaged_mvp.py
```

## Example 2b: AC Sweep RC Low-Pass (frequency-domain baseline)

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_runner.py \
  --only ac_rc_lowpass \
  --output-dir benchmarks/out_ac_rc
```

Checkpoints:

- output artifact includes `frequency_hz`, `magnitude_db`, and `phase_deg`.
- analytical validation (`ac_analytical`) remains within configured magnitude/phase thresholds.
- benchmark telemetry exposes `ac_sweep_mag_error` and `ac_sweep_phase_error`.

## Example 3: Closed-Loop Buck (controller + PWM path)

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_runner.py \
  --only buck_mosfet_nonlinear \
  --output-dir benchmarks/out_cl_buck
```

Checkpoints:

- duty command remains bounded in `[0, 1]`.
- output tracks reference without persistent divergence.
- control path remains deterministic between runs.

## Example 4: Electro-Thermal Scenario

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_runner.py \
  --benchmarks benchmarks/electrothermal_benchmarks.yaml \
  --output-dir benchmarks/out_electrothermal
```

Checkpoints:

- device temperatures increase consistently with losses.
- `thermal_summary.max_temperature` remains within design envelope.
- efficiency and loss totals are coherent with operating point.

## Example 5: Closed-Loop Buck + Thermal Port Validation

```bash
PYTHONPATH=build/python python3 - <<'PY'
import pulsim as ps

parser = ps.YamlParser(ps.YamlParserOptions())
circuit, options = parser.load("examples/09_buck_closed_loop_loss_thermal_validation_backend.yaml")
options.newton_options.num_nodes = int(circuit.num_nodes())
options.newton_options.num_branches = int(circuit.num_branches())

sim = ps.Simulator(circuit, options)
result = sim.run_transient(circuit.initial_state())

print("success:", result.success)
print("steps:", result.total_steps)
print("max_temp:", result.thermal_summary.max_temperature)
for item in result.component_electrothermal:
    if item.thermal_enabled:
        print(item.component_name, item.average_power, item.final_temperature)
PY
```

Checkpoints:

- closed-loop control remains bounded (`PI` output and PWM duty).
- thermal-enabled devices report non-empty telemetry in `component_electrothermal`.
- run remains stable for longer windows (10 ms, 20 ms, ...).

## Example 6: Magnetic Core MVP (saturation + frequency-sensitive loss)

```bash
PYTHONPATH=build/python python3 examples/run_magnetic_core_saturation_freq_loss.py
```

Checkpoints:

- `Lsat.core_loss` exists in `result.virtual_channels` and stays non-negative.
- metadata for `Lsat.core_loss` is coherent (`domain=loss`, `unit=W`, `source_component=Lsat`).
- when `loss_policy: loss_summary`, summary row `Lsat.core` is present in `loss_summary.device_losses`.
- average/peak trends are deterministic for repeated runs with the same setup.

Notebook walkthrough:

- `examples/notebooks/35_magnetic_core_mvp_tutorial.ipynb`

## Output Artifacts for Automation

Main files used in CI/regression tooling:

- `results.json`: benchmark metrics and telemetry summaries
- `results.csv`: tabular benchmark export
- `parity_results.json`: external simulator parity metrics
- `stress_results.json`: tiered stress evaluation

### Minimal `results.json` shape

```json
{
  "benchmark_id": "buck_converter",
  "status": "passed",
  "runtime_s": 0.42,
  "steps": 19876,
  "max_error": 0.0018,
  "telemetry": {
    "newton_iterations": 53211,
    "timestep_rejections": 21,
    "linear_fallbacks": 3
  }
}
```

## Notebook Coverage

Reference notebooks are under `examples/notebooks` and include:

- first-contact setup
- converter design scenarios
- thermal modeling
- benchmark and validation workflows

See [Notebooks](notebooks.md) for execution details.
