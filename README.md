# Pulsim

Power-electronics circuit simulator — C++23 kernel with a Python-first API.

Pulsim is a header-only C++ simulation engine (`pulsim`) wrapped by a
flat Python module (`import pulsim`). It is built around a PWL
state-space cache (for fast switched-converter dynamics), a Newton
refinement on top of the cached linear factor (for non-linear devices),
and a built-in event detector for diode/MOSFET commutations.

If you have legacy code from the pre-1.0 cycle (the old
``Circuit`` / ``Simulator`` / ``YamlParser`` surface), see
[`docs/migration-guide.md`](docs/migration-guide.md) for the
device / analysis mapping table.

## Why Pulsim

- **PLECS-style PWL cache** — switched-converter steady-state in
  milliseconds instead of minutes.
- **Header-only C++23 kernel** — drop `pulsim/` into your own CMake
  target via `pulsim::core`; no static-library link step.
- **Python-first ergonomics** — `CircuitBuilder` API takes string node
  names and SI-unit parameters, returns the same `SimulationResult` whether
  you run a transient, an AC sweep, or a parameter sweep.
- **Mixed-domain composable control** — `MixedDomainBlockChain` runs
  PI/PID, comparators, rate limiters, op-amps, FOC blocks and thermal
  networks at kernel speed (no Python interpreter cost per step).
- **Frequency-domain analysis included** — small-signal MNA Bode +
  swept-sine FRA + closed-loop GM/PM measurement, all in the same surface.

## Quick start

```bash
git clone https://github.com/lgili/Pulsim.git
cd Pulsim

# Build kernel + Python extension
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPULSIM_BUILD_PYTHON=ON
cmake --build build -j

# Use Pulsim from the source tree (no `pip install` required)
export PYTHONPATH="$(pwd)/build/python:$PYTHONPATH"

# First run — open-loop buck
python3 examples/scripts/run_buck.py

# Print available components + helpers
python3 -c "import pulsim as p; p.catalog()"
```

### First simulation — 8 lines

```python
import pulsim as p

b = p.CircuitBuilder()
b.add_voltage_source("Vin", "n0", "gnd", 5.0)
b.add_resistor      ("R1", "n0", "vc",   1000.0)
b.add_capacitor     ("C1", "vc", "gnd",  1e-6)

res = p.simulate(b, t_end=5e-3, dt=1e-5)
p.scope(b, res, signals=["vc"])           # one-liner plot
```

### What ships

- **Builder**: `CircuitBuilder` with 20+ helpers covering passives,
  sources, MOSFETs (SH1), IGBTs (Level 1), saturable inductors,
  transformers, op-amps, three-phase sources, …
- **YAML loader**: same surface, 13 ready-made example circuits in
  `examples/`.
- **Solver**: PWL state-space cache + Newton refresh + event detection +
  optional state-aware `step_observer(t, x)` callback for closed loops.
- **Control library**: `PIController`, `PIDController`, `Comparator`,
  `RateLimiter`, `FirstOrderLowPass`, …
- **AC analysis**: swept-sine Bode + auto-tuning (`tune_pi_from_bode`)
  with phase-margin / gain-margin extraction.
- **Frequency-response analyser (FRA)**: closed-loop / nonlinear-Bode
  via the time-domain swept-sine path.
- **Plot helpers**: `p.scope()`, `p.plot_bode()` — one-line waveform +
  Bode plots with sensible defaults.

### Where to learn more

- 6 narrative tutorials: [`docs/tutorials/`](docs/tutorials/)
- Mental model: [`docs/mental-model.md`](docs/mental-model.md)
- API reference: [`docs/api-reference.md`](docs/api-reference.md)
- Gotchas: [`docs/gotchas.md`](docs/gotchas.md)
- 20 runnable scripts: [`examples/scripts/`](examples/scripts/)

## Validation and performance

```bash
# Python runtime tests
PYTHONPATH=build/python pytest python/tests -v

# C++ kernel tests
ctest --test-dir build --output-on-failure

# Benchmark suite
PYTHONPATH=build/python python3 benchmarks/benchmark_runner.py --output-dir benchmarks/out
```

## Documentation

- Documentation site: [https://lgili.github.io/Pulsim/](https://lgili.github.io/Pulsim/)
- Migration guide (from the pre-1.0 surface): [`docs/migration-guide.md`](docs/migration-guide.md)

## Docs deployment (GitHub Pages)

Docs are published by `.github/workflows/docs.yml` using MkDocs Material + mike:

- PR: strict docs build
- `main`: deploy `dev` docs channel
- `vX.Y.Z` tag: deploy release docs and update `latest`

In repository settings, set **Pages Source** to **GitHub Actions**.
