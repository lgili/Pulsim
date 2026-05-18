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

The recommended starting pattern uses `SimulationOptions.from_preset(...)`
to materialise a fully-tuned configuration with one decision — see
[Numerical Configuration](numerical-configuration.md) for the full
preset menu.

```bash
PYTHONPATH=build/python python3 - <<'PY'
import pulsim as ps

# Build (or load) a circuit
parser = ps.YamlParser(ps.YamlParserOptions())
circuit, _ = parser.load("benchmarks/circuits/rc_step.yaml")

# One-call numerical configuration. `Preset.Auto` is a good default
# for new circuits; pick `Preset.Fast` for pure-switching converters,
# `Preset.HighFidelity` for parity validation runs.
options = ps.SimulationOptions.from_preset(ps.Preset.Auto,
                                            dt=1e-6, tstop=1e-3)
options.newton_options.num_nodes    = int(circuit.num_nodes())
options.newton_options.num_branches = int(circuit.num_branches())

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

The same preset is available via YAML:

```yaml
simulation:
  preset: auto         # auto | fast | robust | high_fidelity
  tstop: 1e-3
  dt: 1e-6
```

## Run Core Validation Commands

```bash
# Python runtime tests
PYTHONPATH=build/python pytest python/tests -v --ignore=python/tests/validation

# C++ core tests (if build already includes tests)
ctest --test-dir build --output-on-failure
```

## Next Steps

1. Read [Numerical Configuration](numerical-configuration.md) to
   understand the four presets and when to override individual fields.
2. Read [User Guide](user-guide.md) for canonical backend usage patterns.
3. Use [Examples and Results](examples-and-results.md) for converter scenarios.
4. Add CI quality checks from [Benchmarks and Parity](benchmarks-and-parity.md).
5. For multilevel converters (NPC, MMC, flying-cap), see
   [Multilevel Converters](multilevel-converters.md).
