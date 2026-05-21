## Why

V6's `CircuitBuilder` exposes generic primitives
(`add_switch`, `add_diode`, `add_resistor`, ...). For SMPS
workloads users have to manually wire:

```cpp
// Buck converter switch element: MOSFET + body diode
b.add_switch("Q1", "vin", "sw", 1e3, 1e-9);
b.add_diode ("Q1_BD", "sw", "vin", 1e3, 1e-9, V_th=0.7);
```

Two branches, two add calls, easy to forget the body diode,
and the user has to know the conductance conventions instead
of thinking in MOSFET terms (R_on / R_off / V_F).

V1 of Layer 2 ships SMPS-aware convenience methods:

```cpp
b.add_mosfet           ("Q1", "vin", "sw", R_on=1e-3, R_off=1e9);
b.add_mosfet_with_body_diode("Q2", "sw", "gnd", R_on=1e-3, R_off=1e9,
                              V_F=0.7);
b.add_igbt             ("T1", "vin", "sw", R_on=10e-3, R_off=1e9);
```

These map to the existing kernel branches but encode:
- **MOSFET R_on / R_off in ohms** instead of `g_on / g_off`.
- **Body-diode wiring** baked in (anti-parallel from
  `source → drain`).
- **SMPS-realistic defaults** so a buck/boost/flyback
  prototype works with zero parameter-tuning.

This is the lowest-risk, highest-ergonomics win for SMPS
adoption of v2.

## What Changes

**Scope decision — Layer 2 V1** (MOSFET/IGBT helpers):

- Extend `CircuitBuilder` with three new methods:
  - `add_mosfet(name, drain, source, R_on=1e-3, R_off=1e9)`
    — single controlled switch with MOSFET-default
    resistances. No body diode.
  - `add_mosfet_with_body_diode(name, drain, source,
      R_on=1e-3, R_off=1e9, V_F=0.7,
      g_on_diode=1e3, g_off_diode=1e-9)` — adds BOTH a
    controlled switch (drain → source) AND an
    anti-parallel body diode (source → drain) with the
    given forward voltage.
  - `add_igbt(name, collector, emitter, R_on=10e-3,
      R_off=1e9)` — single controlled switch with IGBT-
    default resistances. No anti-parallel diode by default
    (matches typical IGBT module behavior).

- All methods accept SI-unit values (ohms, volts) and
  convert to the kernel's conductance representations
  internally. Sensible defaults make `b.add_mosfet("Q1",
  "vin", "sw")` work out-of-the-box.

- Python bindings: `pulsim.v2.CircuitBuilder` gets the
  same three methods. Wired in `bindings_v2_kernel.cpp`.

- **Tests**:
  - Unit: `add_mosfet` produces 1 branch with correct
    g_on = 1/R_on.
  - Unit: `add_mosfet_with_body_diode` produces 2 branches
    (switch + diode) with the diode anti-parallel.
  - Unit: `add_igbt` produces 1 branch with IGBT defaults.
  - Integration: buck converter using
    `add_mosfet_with_body_diode` parity with manual setup
    (sample-by-sample within 1 µV).
  - Python smoke: builder methods callable from
    `pulsim.v2`.

## Impact

- **Affected specs**: ADDED requirement on
  `kernel-v2-solver` for MOSFET/IGBT builder helpers.
- **Affected code** (~150 LOC):
  - MODIFIED `core/include/pulsim/v2/builder/circuit_builder.hpp`
    (+ 3 methods)
  - MODIFIED `core/tests/v2/builder/test_circuit_builder.cpp`
    (+ unit tests)
  - MODIFIED `python/bindings_v2_kernel.cpp` (+ 3 methods)
  - MODIFIED `python/tests/v2/test_v2_python_bindings.py`
    (+ smoke test)
  - NEW `docs/pulsim-v2/layer2-v1-mosfet-igbt.md`
- **Migration**: zero. Pure additive.
- **Risk**: low. Each helper is 1-2 lines delegating to
  existing primitives.
