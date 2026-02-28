# Getting Started

This guide takes you from a clean checkout to your first backend simulation.

## Prerequisites

- Python 3.10+
- CMake 3.20+
- Ninja
- A C++ toolchain compatible with project requirements

## Option A: Build From Source (Recommended for backend development)

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPULSIM_BUILD_PYTHON=ON
cmake --build build -j
```

Confirm that the package is importable from the local build tree:

```bash
PYTHONPATH=build/python python3 -c "import pulsim as ps; print(ps.__version__)"
```

## Option B: Install Package

```bash
python3 -m pip install --upgrade pip
python3 -m pip install pulsim
```

## First Simulation (RC Step)

```bash
PYTHONPATH=build/python python3 - <<'PY'
import pulsim as ps

parser = ps.YamlParser(ps.YamlParserOptions())
circuit, options = parser.load("benchmarks/circuits/rc_step.yaml")

options.newton_options.num_nodes = int(circuit.num_nodes())
options.newton_options.num_branches = int(circuit.num_branches())
options.step_mode = ps.StepMode.Variable

sim = ps.Simulator(circuit, options)
result = sim.run_transient(circuit.initial_state())

print("success:", result.success)
print("steps:", result.total_steps)
print("samples:", len(result.time))
PY
```

Expected outcome:

- `success: True`
- non-zero `steps` and `samples`
- monotonic RC charging behavior on `V(out)`

## Run Core Validation Commands

```bash
# Python runtime tests
PYTHONPATH=build/python pytest python/tests -v --ignore=python/tests/validation

# C++ core tests (if build already includes tests)
ctest --test-dir build --output-on-failure
```

## Next Steps

1. Read [User Guide](user-guide.md) for canonical backend usage patterns.
2. Use [Examples and Results](examples-and-results.md) for converter scenarios.
3. Add CI quality checks from [Benchmarks and Parity](benchmarks-and-parity.md).
