## Why

After Layer 6 shipped `CircuitBuilder`, v2's surface is
finally usable without manually wiring `Graph + DevicePool`.
But all of it is still C++ — adoption from Python (Pulsim's
primary user-facing language for SMPS / motor-drive scripting)
requires reimplementing every test in C++ or writing ctypes
boilerplate.

V7 ships first-class Python bindings for the v2 kernel via
pybind11 (same machinery used by v1's `pulsim._pulsim`
extension). Python users get a clean `pulsim.v2` module with:

```python
import pulsim.v2 as p

b = p.CircuitBuilder()
b.add_voltage_source("Vin", "n0", "gnd", 5.0)
b.add_resistor("R1", "n0", "n1", 100.0)
b.add_capacitor("C1", "n1", "gnd", 1e-6)

cache = p.PwlStateSpaceCache(b.graph, b.pool)
cache.build(dt=1e-5)

opts = p.SimulationOptions(t_start=0.0, t_end=1e-3, dt=1e-5)
res = p.run_transient(
    cache, b.graph, b.pool, opts,
    switch_fn=lambda t: p.SwitchStateMask(0),
)
```

This is what destrava real-world adoption of v2 — anyone
writing pulsim scripts, Jupyter notebooks, parameter sweeps,
or test harnesses gets v2's PWL cache architecture without
leaving Python.

## What Changes

**Scope decision — Layer 7 V0** (pybind11 bindings for v2):

- New file `python/bindings_v2_kernel.cpp` exposing:
  - `CircuitBuilder` (Layer 6) — full method coverage.
  - `Graph`, `DevicePool` — opaque handles owned by the
    builder.
  - `SwitchStateMask` — constructible from int.
  - `PwlStateSpaceCache` — constructor, `build(dt)`, `dt()`.
  - `SimulationOptions` — all flags + constructor with
    `t_start/t_end/dt` kwargs.
  - `SimulationResult`, `CommutationEvent` — read-only
    accessors for `times`, `states`, `commutation_events`,
    `event_iteration_count`, `num_steps()`.
  - `run_transient` — full signature with `switch_fn`,
    `b_extra_fn`, `start_from_dc_op`.
  - `IdealDiodeParams` — for nonlinear-diode use cases.

- Wired into the existing `_pulsim` extension as a
  submodule `_pulsim.v2_kernel` (avoids name collision with
  v1's existing top-level bindings).

- Python wrapper `python/pulsim/v2.py` re-exports every
  symbol under a clean `pulsim.v2` namespace.

- Python tests in `python/tests/v2/test_v2_python_bindings.py`:
  - Module import + symbol presence.
  - `"gnd"` alias maps to ground.
  - `node_id_of` throws for unknown names.
  - `SimulationOptions` constructor + flag mutability.
  - V_dc + R DC solve via `run_transient`.
  - Half-wave rectifier built and run entirely from Python.
  - `IdealDiodeParams` defaults + keyword args.
  - `add_nonlinear_diode` binding.
  - `Graph` `num_nodes` / `num_branches` accessors.

## Impact

- **Affected specs**: ADDED requirement on
  `kernel-v2-solver` for `pulsim.v2` Python bindings.
- **Affected code** (~400 LOC across):
  - NEW `python/bindings_v2_kernel.cpp`
  - MODIFIED `python/bindings.cpp` (added submodule
    registration in `PYBIND11_MODULE`)
  - MODIFIED `python/CMakeLists.txt` (added
    `bindings_v2_kernel.cpp` to `_pulsim` sources)
  - NEW `python/pulsim/v2.py`
  - NEW `python/tests/v2/__init__.py`
  - NEW `python/tests/v2/test_v2_python_bindings.py`
- **Migration**: zero. Existing v1 bindings unchanged.
  New `pulsim.v2` namespace is additive.
- **Risk**: low. Pure binding wrapper; the underlying
  C++ kernel has 4348 assertions of test coverage.
  Python tests verify the binding plumbing only (since
  every value-add is already kernel-tested).
