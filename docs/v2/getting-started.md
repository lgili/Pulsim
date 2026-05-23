# Getting started

## Prerequisites

- **macOS or Linux** (Windows is best-effort via WSL2)
- **CMake ≥ 3.24**, **Ninja**
- **Clang 16+** or **GCC 13+** (we use C++23)
- **Python 3.11+** with `pip`
- A package manager for the third-party dependencies: KLU (SuiteSparse), Eigen, yaml-cpp, Catch2, pybind11. On macOS, install via Homebrew.

```bash
brew install cmake ninja llvm suite-sparse eigen yaml-cpp catch2 pybind11
```

## Build the C++ core + Python bindings

```bash
git clone https://github.com/lgili/pulsim
cd pulsim
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

A successful build produces:
- `build/core/pulsim_v2_*_tests` — Catch2 binaries (run with `ctest --output-on-failure`).
- `build/python/pulsim/_pulsim.cpython-3X-darwin.so` — the Python extension.

For local Python development without `pip install`:

```bash
export PYTHONPATH="$(pwd)/build/python:$PYTHONPATH"
python -c "import pulsim.v2 as p; print(p.__version__)"
```

## Your first transient — a 3-line RC

```python
import pulsim.v2 as p

b = p.CircuitBuilder()
b.add_voltage_source("Vin", "n0", "gnd", 5.0)
b.add_resistor       ("R",  "n0", "vc",  1000.0)   # 1 kΩ
b.add_capacitor      ("C",  "vc", "gnd", 1e-6)     # 1 µF; τ = 1 ms

res = p.simulate(b, t_end=5e-3, dt=1e-5)
vc_idx = b.node_id_of("vc")

# Print v_C at t = 1τ (should be ≈ V·(1 − 1/e) ≈ 3.16 V).
k_1tau = int(1e-3 / 1e-5)
print(f"v_C(1τ) = {res.states[k_1tau][vc_idx]:.3f} V")
```

Expected output:
```
v_C(1τ) = 3.162 V
```

That's the entire flow. `simulate()` (proposal #3.3) hides:
1. Building the PWL state-space cache (`PwlStateSpaceCache`).
2. Constructing the `SimulationOptions`.
3. Defaulting the `switch_fn` to "all switches closed" (here there are no switches, so it's a no-op).
4. Auto-detecting nonlinear devices and enabling the Newton refresh if needed.
5. Calling `run_transient`.

## Plotting

`SimulationResult` has parallel `times` and `states` arrays. You can hand them to matplotlib:

```python
import matplotlib.pyplot as plt

times = res.times                      # list[float], length N
v_c   = [s[vc_idx] for s in res.states]
plt.plot(times, v_c)
plt.xlabel("t [s]"); plt.ylabel("v_C [V]")
plt.show()
```

For circuits with **branch-current unknowns** (voltage sources, inductors), the corresponding indices are obtained via `pool.branch_var_id_for_source(branch_id, graph)` or `branch_var_id_for_inductor(...)`.

## Where to go next

- [Mental model](mental-model.md) — what `Graph`, `DevicePool`, `PwlStateSpaceCache` actually are.
- [Tutorial 01 — RC charging from a pulse source](tutorials/01-rc-charging.md) — same example, but driven by `PulseVoltageSource` (V12).
