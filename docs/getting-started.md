# Getting started

## Install

```bash
pip install pulsim
```

Pulsim ships binary wheels for **CPython 3.10, 3.11, 3.12 and 3.13** on

- macOS on Apple silicon (arm64),
- Linux x86_64 (manylinux_2_28),
- Windows x64.

No compiler is needed. The only runtime dependency is **NumPy**. Optional
features live in extras, so the base install stays small:

| Extra | Pulls in | Needed for |
|---|---|---|
| `pulsim[jupyter]` | matplotlib, plotly, ipywidgets | `p.scope(...)`, `p.plot.*`, notebook widgets |
| `pulsim[data]` | pandas, xarray | `SweepResult.to_dataframe()` |
| `pulsim[lib]` | PyYAML | reading/writing part files in `pulsim.lib` |
| `pulsim[fast]` | numba | `@p.fast_block` JIT-compiled control blocks |
| `pulsim[schematic]` | schemdraw, networkx, cairosvg, anthropic | schematic rendering |

```bash
pip install "pulsim[jupyter]"          # e.g. to plot
```

Check the install:

```bash
python -c "import pulsim as p; print(p.__version__)"
```

```
2.0.0
```

## Your first transient — an RC in a few lines

```python
import numpy as np
import pulsim as p

b = p.CircuitBuilder()
b.add_voltage_source("Vin", "n0", "gnd", 5.0)
b.add_resistor       ("R",  "n0", "vc",  1000.0)   # 1 kΩ
b.add_capacitor      ("C",  "vc", "gnd", 1e-6)     # 1 µF; τ = 1 ms

res = p.simulate(b, t_end=5e-3, dt=1e-5)

# v_C at t = 1τ; the analytical value is V·(1 − 1/e) = 3.161 V.
v_c = res.v("vc")                      # numpy array, one value per sample
print(f"v_C(1τ) = {np.interp(1e-3, res.times, v_c):.3f} V  [{res.engine_used}]")
```

Expected output:
```
v_C(1τ) = 3.151 V  [pwl]
```

Passing `dt` asks for the **fixed-step** trapezoidal engine (`pwl`). It
starts from an all-zero state, so the 5 V source is switched on during
the first step; the response is the exact exponential delayed by half a
step (`dt/2 = 5 µs`), which is the 0.3 % gap to 3.161 V. Shrink `dt` and
the gap shrinks with it.

Leave `dt` out and the default `engine='auto'` picks the engine — here
the variable-step TR-BDF2 engine, which lands on the analytical value:

```python
res_auto = p.simulate(b, t_end=5e-3)
print(f"v_C(1τ) = {np.interp(1e-3, res_auto.times, res_auto.v('vc')):.3f} V"
      f"  [{res_auto.engine_used}]")
```

```
v_C(1τ) = 3.161 V  [trbdf2]
```

`res.engine_used` says which engine ran and `res.engine_route_reason`
says why `auto` did not take the variable-step one. A variable-step run
has a non-uniform time grid: interpolate with `np.interp` or average
with `np.trapezoid(y, t) / (t[-1] - t[0])` rather than indexing samples
or taking a plain `.mean()`.

That is the entire flow. `simulate()` hides:

1. Choosing the engine (`dt` given → `pwl`; otherwise `auto` routes).
2. Building the PWL state-space cache (`PwlStateSpaceCache`) for the
   fixed-step engine — lazily, one factorisation per switch state the
   run actually visits.
3. Constructing the `SimulationOptions`.
4. Auto-detecting nonlinear devices and enabling the Newton refresh if
   needed.
5. Running the transient.

## Reading results

`res.times` is the list of sample times and `res.states` a read-only
`numpy` array of shape `(num_samples, state_size)`. You rarely need to
index it by hand: ask by name.

```python
v_c = res.v("vc")      # node voltage trace
i_r = res.i("R")       # branch current through R, from n0 to vc
print(f"i_R(0+) = {i_r[1] * 1e3:.3f} mA,  i_R(end) = {i_r[-1] * 1e6:.3f} µA")
```

```
i_R(0+) = 4.975 mA,  i_R(end) = 33.858 µA
```

## Plotting

Plotting needs the `[jupyter]` extra (matplotlib). One line plots
named signals and returns the matplotlib `Figure` (add `show=True` to
open a window, or `save="rc.png"` to write a file):

<!-- doctest: requires=matplotlib -->
```python
fig = p.scope(b, res, signals=["vc"])
```

Or hand the arrays to matplotlib yourself:

<!-- doctest: requires=matplotlib -->
```python
import matplotlib.pyplot as plt

plt.plot(res.times, res.v("vc"))
plt.xlabel("t [s]"); plt.ylabel("v_C [V]")
plt.show()
```

## Where to go next

- [Mental model](mental-model.md) — what `Graph`, `DevicePool`, `PwlStateSpaceCache` actually are.
- [Tutorial 01 — RC charging from a pulse source](tutorials/01-rc-charging.md) — same example, but driven by `PulseVoltageSource`.
- [API reference](api-reference.md) — every builder method and helper signature.

## Building from source

Only needed if you want to work on the C++ kernel or the bindings.

**Requirements.** A C++23 compiler (AppleClang 15+ / Clang 17+ /
GCC 13+), CMake ≥ 3.20 and Python ≥ 3.10. Ninja is optional. Nothing
else has to be installed by hand: Eigen 3.4 is taken from the system if
CMake finds it and downloaded otherwise; yaml-cpp, Catch2 (tests) and
pybind11 (if the build environment does not provide it) are fetched at
configure time.

**Python package, editable.** `pip` drives the CMake build through
scikit-build-core:

```bash
git clone https://github.com/lgili/Pulsim.git
cd Pulsim
pip install -e .
```

**Plain CMake** — for the C++ tests, or a build-tree Python package:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPULSIM_BUILD_PYTHON=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

`PULSIM_BUILD_TESTS` (default `ON`) builds the Catch2 binaries
`build/core/pulsim_core_*_tests`. `PULSIM_BUILD_PYTHON` (default `OFF`)
builds the extension into `build/python/pulsim/`, a full copy of the
package; use it without installing:

```bash
export PYTHONPATH="$(pwd)/build/python:$PYTHONPATH"
python -c "import pulsim as p; print(p.__version__)"
```
