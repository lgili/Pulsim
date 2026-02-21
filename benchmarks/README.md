# Benchmarks

This folder contains the YAML benchmark suite and validation runners.

## Structure

- `circuits/` — YAML netlists with embedded `benchmark` metadata.
- `benchmarks.yaml` — scenario matrix and benchmark list.
- `benchmark_runner.py` — executes benchmarks and produces results artifacts.
- `validation_matrix.py` — runs all solver/integrator combinations.
- `benchmark_ngspice.py` — optional Pulsim vs ngspice comparator (same circuit pair).

## Running

```bash
# Use local build bindings (repository workflow)
export PYTHONPATH=build/python

python3 benchmarks/benchmark_runner.py --output-dir benchmarks/out
python3 benchmarks/validation_matrix.py --output-dir benchmarks/matrix

# Compare Pulsim vs ngspice (manifest mode, uses ngspice_netlist mappings)
python3 benchmarks/benchmark_ngspice.py --output-dir benchmarks/ngspice_out

# Compare one YAML vs one .cir directly
python3 benchmarks/benchmark_ngspice.py \
  --pulsim-netlist benchmarks/circuits/rc_step.yaml \
  --spice-netlist benchmarks/ngspice/rc_step.cir \
  --output-dir benchmarks/ngspice_single
```

Benchmark runners are Python-first and execute through `pulsim` runtime bindings
(`YamlParser` + `Simulator`).
When a benchmark netlist omits `simulation.adaptive_timestep`, runners default to
fixed-step mode (`adaptive_timestep: false`) for deterministic comparisons.

Generate missing reference baselines:

```bash
python3 benchmarks/benchmark_runner.py --generate-baselines
```

## Output Artifacts

Each run produces:

- `results.csv` — per-scenario metrics.
- `results.json` — full structured results and metadata.
- `summary.json` — pass/fail summary.

Telemetry fields are sourced from structured simulation result objects and included in `results.json`.
Analytical `max_error` thresholds in `circuits/*.yaml` are calibrated for the current
Python-first runtime defaults (fixed-step unless explicitly overridden).

`benchmark_ngspice.py` also emits:

- `ngspice_results.csv` — per benchmark/scenario parity results.
- `ngspice_results.json` — per-observable metrics (`max_error`, `rms_error`, `samples`).
- `ngspice_summary.json` — pass/fail/skip totals.

## Adding Benchmarks

1. Create a YAML netlist in `circuits/` with a `benchmark` block.
2. Add it to `benchmarks.yaml` and assign scenarios.
3. If you want ngspice parity for this benchmark, add `ngspice_netlist: ngspice/<file>.cir` in `benchmarks.yaml`.
4. Optional: define `ngspice_observables` in `benchmarks.yaml` to map Pulsim CSV columns to ngspice vectors.
   - Example: `{ column: "V(out)", spice_vector: "v(out)" }`
5. If using `reference` validation, add a baseline CSV under `baselines/`.
6. Optional validation window controls:
   - `benchmark.validation.ignore_initial_samples`: ignore N leading samples.
   - `benchmark.validation.start_time`: compare only from a minimum time.
