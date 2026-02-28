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
  --only buck_converter \
  --output-dir benchmarks/out_buck
```

Checkpoints:

- output settles around expected duty-dependent value.
- ripple remains physically plausible for chosen L/C and switching frequency.
- telemetry does not show runaway fallback escalation.

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
