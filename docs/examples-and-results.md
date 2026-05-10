# Examples and Results

This page focuses on practical backend runs with expected outputs and validation criteria.

## Runnable Python examples for new features

The May 9–10 roadmap (Fase 0–4) ships a set of runnable Python scripts
covering every new user-facing feature. They live under
[`examples/python/`](https://github.com/lgili/Pulsim/tree/main/examples/python)
and each one is a self-contained file with a TL;DR docstring at the
top, sensible defaults, and an optional matplotlib plot guarded by
`PULSIM_EXAMPLE_NOPLOT=1` so it runs cleanly in headless CI:

| Script | Demonstrates | Doc page |
|---|---|---|
| `01_ac_sweep_rc.py` | Analytical AC sweep + Bode (validates -3 dB / -45° corner). | [AC Analysis](ac-analysis.md) |
| `02_fra_vs_ac_sweep.py` | Empirical FRA overlaid on the analytical AC sweep. | [FRA](fra.md) |
| `03_buck_template.py` | `pulsim.templates.buck` auto-design + transient. | [Converter Templates](converter-templates.md) |
| `04_boost_buckboost_templates.py` | Boost + buck-boost templates side by side. | [Converter Templates](converter-templates.md) |
| `05_codegen_pil.py` | Codegen → C99 → gcc → PIL parity diff. | [Code Generation](code-generation.md) |
| `06_fmu_export.py` | FMI 2.0 CS `.fmu` export + 13/13 callback symbol check. | [FMI Export](fmi-export.md) |
| `07_parameter_sweep_lhs.py` | 64-point LHS over (L, C) with P5/P50/P95 metrics. | [Parameter Sweep](parameter-sweep.md) |
| `08_monte_carlo_yield.py` | 256 MC draws under component tolerance, yield histogram. | [Parameter Sweep](parameter-sweep.md) |
| `09_robustness_wrapper.py` | `pulsim.run_transient(robust=True)` retry loop. | [Robustness Policy](robustness-policy.md) |
| `10_linear_solver_cache.py` | Per-key cache hit rate via `BackendTelemetry`. | [Linear-Solver Cache](linear-solver-cache.md) |
| `11_yaml_ac_analysis.py` | Load circuit from YAML + dispatch AC sweep. | [AC Analysis](ac-analysis.md) |

Header-only features (magnetic models, motors, three-phase grid,
robustness profile) don't have Python bindings yet — their runnable
counterparts live in
[`examples/cpp/`](https://github.com/lgili/Pulsim/tree/main/examples/cpp).

## Legacy notebook examples

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
