# Design — `pulsim-v2-python-bindings` (Layer 7 V0)

## The binding stack

```
+--------------------------------------+
| User Python code                     |
|   import pulsim.v2 as p              |
|   b = p.CircuitBuilder()             |
|   ...                                |
+--------------------------------------+
              ↓ re-exports
+--------------------------------------+
| python/pulsim/v2.py (Python wrapper) |
|   from ._pulsim.v2_kernel import (   |
|       CircuitBuilder, Graph, ...)    |
+--------------------------------------+
              ↓ pybind11
+--------------------------------------+
| _pulsim.v2_kernel (C++ extension)    |
|   bindings_v2_kernel.cpp             |
+--------------------------------------+
              ↓ delegates to
+--------------------------------------+
| v2 kernel (Layers 0-6)               |
|   header-only, ~4400 assertions      |
+--------------------------------------+
```

## Why a submodule (not top-level)

The existing `python/bindings.cpp` is ~258 KB of v1
bindings. Those bind v1's CRTP device classes (`Resistor`,
`Capacitor`, etc.) directly into the top-level `_pulsim`
namespace.

My v2 has the SAME class names but in a different C++
namespace (`pulsim::v2::models::...`). Binding both at top
level would collide.

Solution: put the v2 kernel bindings under
`_pulsim.v2_kernel` (a pybind11 submodule). The Python
wrapper `pulsim/v2.py` re-exports each symbol so users see
the clean `pulsim.v2` namespace.

## What's bound

| C++ type | Python |
|----------|--------|
| `builder::CircuitBuilder` | `p.CircuitBuilder` |
| `topology::Graph` | `p.Graph` (opaque; build via builder) |
| `pwl::DevicePool` | `p.DevicePool` (opaque) |
| `topology::SwitchStateMask` | `p.SwitchStateMask(int)` |
| `pwl::PwlStateSpaceCache` | `p.PwlStateSpaceCache(graph, pool)` |
| `solver::SimulationOptions` | `p.SimulationOptions(t_start, t_end, dt)` |
| `solver::SimulationResult` | `p.SimulationResult` (read-only) |
| `solver::CommutationEvent` | `p.CommutationEvent` (read-only) |
| `solver::run_transient` | `p.run_transient(...)` |
| `models::IdealDiode::Params` | `p.IdealDiodeParams` |

## What's NOT bound in V0

- **Newton-iterated `run_transient`**: the overload that
  takes `NonlinearRefreshFn` (Layer 5 V4) requires passing
  a C++ functor that calls into AD-aware stamping. Wiring
  Python callbacks through `evaluate_current_and_jacobian`
  is a V1 add-on.
- **Direct Newton solver primitives**:
  `solve_with_newton_b_extra`, `continuation_solve`,
  `pseudo_transient_solve`, `make_diode_aware_initial_guess`.
  Power users who need these can drop to C++ at this stage.
- **HistoryState / DiodeEventState**: Python users get the
  high-level `run_transient` flow. The underlying state
  trackers are implementation detail.

## Ownership / lifetime

The C++ kernel uses const references heavily:
`PwlStateSpaceCache::PwlStateSpaceCache(const Graph&, const
DevicePool&)` stores refs to the builder's internal members.

pybind11 needs `py::keep_alive<1, 2>()` to ensure the
graph/pool outlive the cache. The bindings declare:

```cpp
py::class_<pwl::PwlStateSpaceCache>(...)
    .def(py::init<const topology::Graph&,
                   const pwl::DevicePool&>(),
          py::keep_alive<1, 2>(),
          py::keep_alive<1, 3>());
```

Builder's `.graph` and `.pool` use
`py::return_value_policy::reference_internal` so Python
holds a reference back to the builder while it has the
graph/pool view.

User code pattern:

```python
b = p.CircuitBuilder()   # builder owns graph + pool
b.add_voltage_source(...)
cache = p.PwlStateSpaceCache(b.graph, b.pool)
# `b` MUST outlive `cache` — typical Python scoping handles
# this naturally (both live in the same function scope).
```

## Numpy / Eigen interop

`pybind11/eigen.h` (already included in v1 bindings) auto-
converts `Eigen::VectorXd` ↔ `numpy.ndarray`. So
`SimulationResult.states[k]` returns a 1-D numpy array
without any explicit conversion in our binding code.

The user-supplied `b_extra_fn` returns a numpy array that
pybind11 auto-converts back to `Eigen::VectorXd`.

## Test plan

11 cases in `python/tests/v2/test_v2_python_bindings.py`:

1. **Module imports**: every public name from `pulsim.v2`.
2. **`gnd` alias**: `b.node("gnd") == b.node("GND") ==
   p.Graph.ground()`.
3. **`node_id_of` validation**: throws for unknown names.
4. **`SimulationOptions` constructor**: positional + kwarg
   forms; flag mutability.
5. **`SwitchStateMask` repr**: `repr(p.SwitchStateMask(5))`
   contains the class name.
6. **V_dc + R DC solve**: 5V → 1Ω → GND should give v_n0=5V
   for every recorded step.
7. **Half-wave rectifier from Python**: builder + run_transient
   produce the expected half-wave with > 90 % pos-half +
   neg-half tracking.
8. **`IdealDiodeParams` defaults**: V_F0=0.7, R_d=0.01,
   G_off=1e-9, kappa=20.0.
9. **`IdealDiodeParams` named args**: keyword construction
   sets fields correctly.
10. **`add_nonlinear_diode` binding**: builds without
    crashing; `num_branches` increments.
11. **`Graph` accessors**: `num_nodes` / `num_branches`
    return correct counts.

## What V0 deliberately does NOT do

- **Type stubs (`.pyi`)** for the v2 module: V0 ships
  docstrings only; mypy-strict users can wait for V1.
- **Sphinx docs**: pulsim's existing docs don't yet cover
  v2 at all. V8 (showcases) will document examples that
  exercise these bindings.
- **Jupyter / matplotlib helpers**: V0 returns raw numpy
  arrays. Plotting helpers are V1.
- **Wheel publishing**: V0 keeps the existing
  `pulsim._pulsim` extension layout; wheels built via
  `pip install -e .` automatically include the new
  `pulsim.v2.py`. Pip-publishable wheels are V1.

## Files

- NEW `python/bindings_v2_kernel.cpp` (~270 LOC)
- MODIFIED `python/bindings.cpp` (+15 LOC for the submodule
  registration call in `PYBIND11_MODULE`)
- MODIFIED `python/CMakeLists.txt` (+1 LOC source addition)
- NEW `python/pulsim/v2.py` (~50 LOC, Python wrapper)
- NEW `python/tests/v2/__init__.py` (empty)
- NEW `python/tests/v2/test_v2_python_bindings.py` (~250 LOC)
- NEW `docs/pulsim-v2/layer7-python-bindings.md`
