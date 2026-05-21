# Layer 7 — Python bindings for v2

V6 shipped the `CircuitBuilder` C++ API. V7 wraps it in
pybind11 so Python scripts can build, run, and analyze v2
simulations without writing any C++. Bindings sit under
`pulsim.v2`, a clean namespace that re-exports the
`pulsim._pulsim.v2_kernel` C++ submodule.

## Quick start

```python
import pulsim.v2 as p

# Build a circuit.
b = p.CircuitBuilder()
b.add_voltage_source("Vin", "n0", "gnd", 5.0)
b.add_resistor      ("R1",  "n0", "n1", 100.0)  # ohms
b.add_capacitor     ("C1",  "n1", "gnd", 1e-6)  # farads

# Pre-factor the MNA matrix per switch state.
cache = p.PwlStateSpaceCache(b.graph, b.pool)
cache.build(dt=1e-5)

# Run a transient.
opts = p.SimulationOptions(t_start=0.0, t_end=1e-3, dt=1e-5)
res = p.run_transient(
    cache, b.graph, b.pool, opts,
    switch_fn=lambda t: p.SwitchStateMask(0),
)

print(f"recorded {res.num_steps()} samples")
print("final state:", res.states[-1])   # numpy array
```

## Bound symbols

| Symbol | Purpose |
|--------|---------|
| `CircuitBuilder` | Layer 6's fluent builder; main entry. |
| `Graph` | Opaque topology handle (built by the builder). |
| `DevicePool` | Opaque parameter pool. |
| `SwitchStateMask(int)` | Bit-vector for switch state. |
| `PwlStateSpaceCache(graph, pool)` | PWL cache constructor. |
| `SimulationOptions(t_start, t_end, dt)` | Inputs to `run_transient`. |
| `SimulationResult` | Output: `times`, `states`, `commutation_events`, `event_iteration_count`. |
| `CommutationEvent` | Sub-step event diagnostic. |
| `run_transient(...)` | Fixed-dt transient solver. |
| `IdealDiodeParams` | Smooth-blend diode params. |

## Numpy interop

`SimulationResult.states[k]` returns a 1-D `numpy.ndarray`
directly — no manual conversion. The state-vector layout is
`[v_n0, v_n1, ..., i_src0, i_src1, ..., i_L0, i_L1, ...]`
in node-insertion + source-insertion + inductor-insertion
order.

User-supplied `b_extra_fn(t)` returns a numpy array (or
list/tuple); pybind11 auto-converts to `Eigen::VectorXd`.

```python
import math
import numpy as np

omega = 2 * math.pi * 60.0
state_size = 3   # 2 nodes + 1 source

def b_extra_fn(t):
    bv = np.zeros(state_size, dtype=np.float64)
    bv[2] = -10.0 * math.sin(omega * t)  # V_sine modulation
    return bv

res = p.run_transient(
    cache, b.graph, b.pool, opts,
    switch_fn=lambda t: p.SwitchStateMask(1),
    b_extra_fn=b_extra_fn,
)
```

## SimulationOptions

All option flags are settable from Python:

```python
opts = p.SimulationOptions(t_start=0, t_end=1e-3, dt=1e-5)

# Newton globalization (Layer 4 V4/V5).
opts.enable_newton_line_search = True
opts.enable_newton_lm = False

# Layer 5 V3 substep correction.
opts.enable_substep_state_correction = True

# Event-iteration limit (Layer 5 V2.1).
opts.max_event_iterations = 16

# Newton tuning (Layer 5 V4).
opts.max_newton_iterations = 50
opts.tol_newton_dx = 1e-9
opts.tol_newton_res = 1e-9
```

## What's NOT bound in V0 (use the C++ kernel)

- `solve_with_newton_b_extra` direct call (for users
  writing their own outer loop).
- `continuation_solve`, `pseudo_transient_solve` (Layer 4
  V8 / V10 primitives).
- `make_diode_aware_initial_guess` (Layer 4 V10's helper).
- `refresh_smooth_diodes` for Newton-iterated transients
  (the `run_transient` Newton overload — Python can use
  the `add_nonlinear_diode` builder method and run static
  cache.solve, but the full Newton transient loop requires
  C++).

Power users who need these primitives in Python should
either drop to the C++ kernel for the inner loop or wait
for V1 of these bindings.

## Submodule layout under the hood

```
pulsim._pulsim       (the pybind11 extension shared lib)
├── (v1 bindings)    (top-level, ~258 KB of v1's API)
└── v2_kernel        (def_submodule("v2_kernel"))
    ├── CircuitBuilder
    ├── PwlStateSpaceCache
    └── ...

pulsim/v2.py         (Python wrapper)
    re-exports from _pulsim.v2_kernel under `pulsim.v2`
```

The submodule split prevents class-name collisions
(v1's `Resistor` vs v2's `models::Resistor`) while keeping
everything in a single shared library — Python only loads
one `.so`.

## Test coverage

11 Python tests in
`python/tests/v2/test_v2_python_bindings.py`:

1. Module imports + public symbol presence.
2. `gnd` alias maps to `Graph.ground()`.
3. `node_id_of` throws for unknown names.
4. `SimulationOptions` constructor + flag mutability.
5. `SwitchStateMask` repr contains class name.
6. V_dc + R DC solve roundtrip (every sample == V_dc).
7. Half-wave rectifier built and run entirely from Python.
8. `IdealDiodeParams` defaults match Layer 4 V3
   conventions.
9. `IdealDiodeParams` keyword args.
10. `add_nonlinear_diode` binding works.
11. `Graph` accessors return correct counts.

Run with:

```bash
PYTHONPATH=build/python ./.venv/bin/python3 \
    -m pytest python/tests/v2/ -v
```

## What V0 deliberately does NOT do

- **Type stubs (`.pyi`)** for `pulsim.v2`: V0 ships
  docstrings only.
- **Sphinx docs / API reference**: V0 is markdown-only
  (this file).
- **Jupyter / matplotlib examples**: V0 returns raw numpy
  arrays. Plotting examples are V1.
- **Wheel publishing**: V0 keeps the existing editable-
  install / scikit-build-core flow.
- **Newton-transient binding**: the `run_transient`
  overload that takes a `NonlinearRefreshFn`. V1 will wire
  this through a Python callback after the binding
  infrastructure proves stable.

## Files

- NEW `python/bindings_v2_kernel.cpp`
- MODIFIED `python/bindings.cpp` (added submodule call)
- MODIFIED `python/CMakeLists.txt`
- NEW `python/pulsim/v2.py`
- NEW `python/tests/v2/__init__.py`
- NEW `python/tests/v2/test_v2_python_bindings.py`
- NEW `openspec/changes/pulsim-v2-python-bindings/`
