# PulsimCore

High-performance power-electronics simulation with a Python-first runtime.

## Supported Product Surface

Pulsim is now **Python-only** for user-facing usage:

- Supported: `import pulsim` APIs and YAML netlist workflows.
- Not supported as product surface: legacy CLI, gRPC server/client docs, JSON netlist loading.
- Core execution engine: unified `v1` kernel.

See:

- `docs/user-guide.md`
- `docs/migration-guide.md`
- `openspec/changes/refactor-python-only-v1-hardening/tasks.md`

## Quick Start

### 1) Build local Python bindings

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPULSIM_BUILD_PYTHON=ON
cmake --build build -j
```

### 2) Run with Python package from local build

```bash
PYTHONPATH=build/python python3 - <<'PY'
import pulsim as ps

parser_opts = ps.YamlParserOptions()
parser = ps.YamlParser(parser_opts)
circuit, sim_opts = parser.load("benchmarks/circuits/rc_step.yaml")

sim_opts.newton_options.num_nodes = int(circuit.num_nodes())
sim_opts.newton_options.num_branches = int(circuit.num_branches())

sim = ps.Simulator(circuit, sim_opts)
result = sim.run_transient(circuit.initial_state())
print("success:", result.success, "steps:", result.total_steps)
PY
```

## Benchmarks and Validation

```bash
# Benchmark matrix (Python runtime path)
PYTHONPATH=build/python python3 benchmarks/validation_matrix.py --output-dir benchmarks/matrix

# External parity (ngspice)
PYTHONPATH=build/python python3 benchmarks/benchmark_ngspice.py \
  --backend ngspice \
  --output-dir benchmarks/ngspice_out

# External parity (LTspice, explicit executable path required)
PYTHONPATH=build/python python3 benchmarks/benchmark_ngspice.py \
  --backend ltspice \
  --ltspice-exe "/Applications/LTspice.app/Contents/MacOS/LTspice" \
  --output-dir benchmarks/ltspice_out

# Tiered stress suite (A/B/C)
PYTHONPATH=build/python python3 benchmarks/stress_suite.py --output-dir benchmarks/stress_out
```

## Documentation Site

Build docs locally:

```bash
python3 -m pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in your browser.

GitHub Pages deploy is automated by `.github/workflows/docs.yml` on pushes to `main`.

## Notes

- YAML schema is `pulsim-v1`.
- Benchmark/parity artifacts are machine-readable (`results.json`, `parity_results.json`, `stress_results.json`).
- For migration details (removed APIs and timeline), see `docs/migration-guide.md`.
