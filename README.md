# PulsimCore

High-performance backend for power electronics simulation.

PulsimCore combines a C++ simulation kernel with a Python-first runtime so you can build, validate, and ship converter simulations with reproducible YAML netlists.

## Why PulsimCore

- Python-native workflow: `import pulsim`
- Versioned YAML netlist schema (`pulsim-v1`)
- Robust transient flow for switched converters (fallback-aware)
- Mixed-domain support (control, events, thermal coupling)
- Built-in benchmark, parity, and stress tooling for CI gates

## Quick Start

### Build local bindings

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPULSIM_BUILD_PYTHON=ON
cmake --build build -j
```

### Run a first simulation

```bash
PYTHONPATH=build/python python3 - <<'PY'
import pulsim as ps

parser = ps.YamlParser(ps.YamlParserOptions())
circuit, options = parser.load("benchmarks/circuits/rc_step.yaml")

options.newton_options.num_nodes = int(circuit.num_nodes())
options.newton_options.num_branches = int(circuit.num_branches())

sim = ps.Simulator(circuit, options)
result = sim.run_transient(circuit.initial_state())

print("success:", result.success, "steps:", result.total_steps)
PY
```

## Pulsim v2 — alpha (in active development)

The repo also ships a **next-generation kernel** (`pulsim.v2`) — a C++23
header-only simulator with a Python-first surface that builds circuits
fluently, runs transient + AC analysis, and includes built-in control
blocks (PI/PID/op-amps) for closed-loop SMPS workflows. v2 lives next to
v1 in the same tree; both work, choose with the import path.

### Quick install (dev mode)

```bash
git clone https://github.com/lgili/Pulsim.git
cd Pulsim

# Build kernel + Python extension
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPULSIM_BUILD_PYTHON=ON
cmake --build build -j

# Use v2 from the source tree (no `pip install` required)
export PYTHONPATH="$(pwd)/python:$PYTHONPATH"

# First run: open-loop buck
python3 examples/v2/scripts/run_buck.py

# Print available components + helpers
python3 -c "import pulsim.v2 as p; p.catalog()"
```

### First simulation — 8 lines

```python
import pulsim.v2 as p

b = p.CircuitBuilder()
b.add_voltage_source("Vin", "n0", "gnd", 5.0)
b.add_resistor       ("R1", "n0", "vc",   1000.0)
b.add_capacitor      ("C1", "vc", "gnd",  1e-6)

res = p.simulate(b, t_end=5e-3, dt=1e-5)
p.scope(b, res, signals=["vc"])           # one-liner plot
```

### What v2 ships

- **Builder**: `CircuitBuilder` with 20+ helpers covering passives,
  sources, MOSFETs (SH1), IGBTs (Level 1), saturable inductors,
  transformers, op-amps, …
- **YAML loader**: same surface, 13 ready-made example circuits in
  `examples/v2/`.
- **Solver**: PWL state-space cache + Newton refresh + event detection +
  optional state-aware `step_observer(t, x)` callback for closed loops.
- **Control library**: `PIController`, `PIDController`, `Comparator`,
  `RateLimiter`, `FirstOrderLowPass`, …
- **AC analysis**: swept-sine Bode + auto-tuning (`tune_pi_from_bode`)
  with phase-margin / gain-margin extraction.
- **Plot helpers**: `p.scope()`, `p.plot_bode()` — one-line waveform +
  Bode plots with sensible defaults.

### Where to learn more

- 6 narrative tutorials: [`docs/v2/tutorials/`](docs/v2/tutorials/)
- Mental model: [`docs/v2/mental-model.md`](docs/v2/mental-model.md)
- API reference: [`docs/v2/api-reference.md`](docs/v2/api-reference.md)
- Gotchas: [`docs/v2/gotchas.md`](docs/v2/gotchas.md)
- 20 runnable scripts: [`examples/v2/scripts/`](examples/v2/scripts/)

## Documentation

- Documentation site: [https://lgili.github.io/Pulsim/](https://lgili.github.io/Pulsim/)
- Getting started guide: [`docs/getting-started.md`](docs/getting-started.md)
- Electrothermal workflow: [`docs/electrothermal-workflow.md`](docs/electrothermal-workflow.md)
- API reference: [`docs/api-reference.md`](docs/api-reference.md)
- Benchmarks and parity: [`docs/benchmarks-and-parity.md`](docs/benchmarks-and-parity.md)

## Validation and Performance Workflows

```bash
# Python runtime tests
PYTHONPATH=build/python pytest python/tests -v --ignore=python/tests/validation

# C++ kernel tests
ctest --test-dir build --output-on-failure

# Benchmark suite
PYTHONPATH=build/python python3 benchmarks/benchmark_runner.py --output-dir benchmarks/out
```

## Product Surface

Supported user-facing surface:

- Python runtime (`import pulsim`)
- YAML netlists (`schema: pulsim-v1`)

Legacy CLI/gRPC/JSON-first paths are not the canonical integration target.

## Docs Deployment (GitHub Pages)

Docs are published by `.github/workflows/docs.yml` using MkDocs Material + mike:

- PR: strict docs build
- `main`: deploy `dev` docs channel
- `vX.Y.Z` tag: deploy release docs and update `latest`

In repository settings, set **Pages Source** to **GitHub Actions**.
