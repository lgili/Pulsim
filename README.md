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

## Documentation

- Documentation site: [https://lgili.github.io/Pulsim/](https://lgili.github.io/Pulsim/)
- Getting started guide: [`docs/getting-started.md`](docs/getting-started.md)
- Netlist format (including `simulation.control`): [`docs/netlist-format.md`](docs/netlist-format.md)
- Electrothermal workflow: [`docs/electrothermal-workflow.md`](docs/electrothermal-workflow.md)
- Configuration guide (solver/control/thermal): [`docs/configuration.md`](docs/configuration.md)
- API reference: [`docs/api-reference.md`](docs/api-reference.md)
- Benchmarks and parity: [`docs/benchmarks-and-parity.md`](docs/benchmarks-and-parity.md)

## Closed-Loop + Thermal Quickstart

Use the backend-ready closed-loop buck example:

```bash
PYTHONPATH=build/python python3 - <<'PY'
import pulsim as ps

parser = ps.YamlParser(ps.YamlParserOptions())
circuit, options = parser.load("examples/09_buck_closed_loop_loss_thermal_validation_backend.yaml")
options.newton_options.num_nodes = int(circuit.num_nodes())
options.newton_options.num_branches = int(circuit.num_branches())

sim = ps.Simulator(circuit, options)
result = sim.run_transient(circuit.initial_state())

print("ok:", result.success)
print("max_temp:", result.thermal_summary.max_temperature)
print("components:", len(result.component_electrothermal))
print("thermal_trace_keys:", [k for k in result.virtual_channels if k.startswith("T(")])
PY
```

Generate a plot with electrical/control channels plus switch thermal trace (`T(M1)`):

```bash
PYTHONPATH=build/python python3 examples/plot_buck_closed_loop_thermal_backend.py
```

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
