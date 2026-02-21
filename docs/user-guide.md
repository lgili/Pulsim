# Pulsim User Guide (Python-Only)

Pulsim is a power-electronics simulator with a unified v1 kernel exposed through the Python package.

## 1. Scope

Supported:

- Python runtime (`pulsim`)
- YAML netlists (`schema: pulsim-v1`)
- Benchmark/parity/stress tooling under `benchmarks/`

Not supported as user-facing product surface:

- Legacy CLI workflows
- gRPC remote workflows
- JSON netlist loading

## 2. Build and Runtime Setup

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPULSIM_BUILD_PYTHON=ON
cmake --build build -j
```

Use local bindings:

```bash
export PYTHONPATH=build/python
```

## 3. YAML Netlist Example

```yaml
schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-3
  dt: 1e-6
components:
  - type: voltage_source
    name: V1
    nodes: [in, 0]
    waveform: {type: dc, value: 5.0}
  - type: resistor
    name: R1
    nodes: [in, out]
    value: 1k
  - type: capacitor
    name: C1
    nodes: [out, 0]
    value: 1u
```

## 4. Python Simulation Flow

```python
import pulsim as ps

parser_opts = ps.YamlParserOptions()
parser = ps.YamlParser(parser_opts)
circuit, options = parser.load("circuit.yaml")

options.newton_options.num_nodes = int(circuit.num_nodes())
options.newton_options.num_branches = int(circuit.num_branches())

sim = ps.Simulator(circuit, options)
result = sim.run_transient(circuit.initial_state())

print(result.success, result.total_steps)
```

## 5. Validation and Parity

```bash
# Core benchmark run
python3 benchmarks/benchmark_runner.py --output-dir benchmarks/out

# Full scenario matrix
python3 benchmarks/validation_matrix.py --output-dir benchmarks/matrix

# External parity (ngspice or ltspice)
python3 benchmarks/benchmark_ngspice.py --backend ngspice --output-dir benchmarks/ngspice_out
python3 benchmarks/benchmark_ngspice.py --backend ltspice --ltspice-exe "/path/to/LTspice" --output-dir benchmarks/ltspice_out

# Tiered stress suite
python3 benchmarks/stress_suite.py --output-dir benchmarks/stress_out
```

## 6. Output Artifacts

- Benchmark: `results.csv`, `results.json`, `summary.json`
- Parity: `parity_results.csv`, `parity_results.json`, `parity_summary.json`
- Stress: `stress_results.csv`, `stress_results.json`, `stress_summary.json`

## 7. Migration and Compatibility

For removed API surfaces and deprecation timeline, see `docs/migration-guide.md`.
