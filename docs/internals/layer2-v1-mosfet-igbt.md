# Layer 2 V1 — MOSFET / IGBT builder helpers

V6 shipped the generic `CircuitBuilder` (with
`add_switch`, `add_diode`, `add_resistor`, etc.). V1 of
Layer 2 adds SMPS-friendly device wrappers that bake in:

- Power-switch defaults (`R_on`, `R_off` typical of modern
  Si MOSFETs and IGBT modules).
- Body-diode wiring (one-call instead of two).
- SI units (ohms, volts) so users don't manually compute
  conductance.

## API

```cpp
// MOSFET — single controlled switch, no body diode.
b.add_mosfet("Q1", "drain", "source",
             R_on=1e-3, R_off=1e9);

// MOSFET with intrinsic anti-parallel body diode.
b.add_mosfet_with_body_diode(
    "Q1", "drain", "source",
    R_on=1e-3, R_off=1e9, V_F=0.7);

// IGBT — single controlled switch (no body diode by
// default, since discrete IGBTs typically don't include
// one).
b.add_igbt("T1", "collector", "emitter",
           R_on=10e-3, R_off=1e9);
```

Available from both C++ and Python — same names, same
defaults.

## Defaults

| Param | MOSFET | IGBT | Notes |
|-------|--------|------|-------|
| `R_on` | 1 mΩ | 10 mΩ | Typical Si power device |
| `R_off` | 1 GΩ | 1 GΩ | Leakage cap |
| `V_F` (body diode) | 0.7 V | — | Si body diode |
| `g_on_diode` | 1 kS | — | Matches `add_diode` default |
| `g_off_diode` | 1 nS | — | Matches `add_diode` default |

## `add_mosfet_with_body_diode` internals

```
b.add_mosfet_with_body_diode("Q1", "drain", "source")
```

Adds TWO branches in sequence:

1. Switch branch (drain → source) with `g_on = 1/R_on`,
   `g_off = 1/R_off`. Driven by `switch_fn(t)` at
   simulation time.
2. Diode branch (source → drain — ANTI-PARALLEL!) with
   `V_th = V_F`, `g_on = g_on_diode`,
   `g_off = g_off_diode`. Auto-commutates via the V2
   diode-iteration loop.

The diode direction (source → drain) matches the
intrinsic body-diode direction of an n-channel MOSFET:
conduction flows from source to drain when the channel is
off and the load tries to push current backward (typical
freewheeling scenario in inductive loads).

## SMPS example: synchronous buck

```cpp
CircuitBuilder b;

b.add_voltage_source("Vin", "vin", "gnd", 24.0);

// High-side MOSFET (the controlled "buck" switch).
b.add_mosfet_with_body_diode("Q1", "vin", "sw");

// Low-side MOSFET (synchronous rectifier — its body
// diode handles dead-time freewheeling).
b.add_mosfet_with_body_diode("Q2", "sw", "gnd");

// LC output filter + load.
b.add_inductor ("L1",   "sw",   "vout", 100e-6);
b.add_capacitor("Cout", "vout", "gnd",  47e-6);
b.add_resistor ("R_L",  "vout", "gnd",  10.0);
```

Without the V1 helpers:

```cpp
// Same circuit, manually:
b.add_voltage_source("Vin", "vin", "gnd", 24.0);

// Q1 main switch + Q1 body diode (4 branches just for the
// two MOSFETs!).
b.add_switch("Q1_main",  "vin", "sw",   1000.0, 1e-9);
b.add_diode ("Q1_body",  "sw",  "vin",  1e3, 1e-9, 0.7);
b.add_switch("Q2_main",  "sw",  "gnd",  1000.0, 1e-9);
b.add_diode ("Q2_body",  "gnd", "sw",   1e3, 1e-9, 0.7);

b.add_inductor ("L1",   "sw",   "vout", 100e-6);
b.add_capacitor("Cout", "vout", "gnd",  47e-6);
b.add_resistor ("R_L",  "vout", "gnd",  10.0);
```

Equivalent topology, but the V1 helpers cut 4 lines and
remove the mental load of remembering body-diode direction
and default conductances.

## Python

Identical names + defaults:

```python
import pulsim as p

b = p.CircuitBuilder()
b.add_voltage_source("Vin", "vin", "gnd", 24.0)
b.add_mosfet_with_body_diode("Q1", "vin", "sw")
b.add_diode("D1", "gnd", "sw", 1e3, 1e-9, V_th=0.7)
b.add_inductor("L1", "sw", "vout", 100e-6)
b.add_capacitor("Cout", "vout", "gnd", 47e-6)
b.add_resistor("R_L", "vout", "gnd", 10.0)

cache = p.PwlStateSpaceCache(b.graph, b.pool)
cache.build(dt=1e-7)
# ... ready for run_transient.
```

## Test coverage

**C++** (`core/tests/v2/builder/test_circuit_builder.cpp`,
6 new cases):
- `add_mosfet` adds 1 branch, correct `g_on = 1/R_on`.
- `add_mosfet` defaults (R_on = 1mΩ, R_off = 1GΩ).
- `add_mosfet_with_body_diode` adds 2 branches, diode
  anti-parallel direction.
- `add_mosfet_with_body_diode` defaults (V_F = 0.7 V).
- `add_igbt` adds 1 branch, correct IGBT defaults.
- Buck topology smoke: source + MOSFET-w-body-diode +
  freewheeling diode + R = 5 branches with the expected
  `StoredKind` per branch.

**Python** (`python/tests/v2/test_v2_python_bindings.py`,
5 new cases):
- `add_mosfet` callable.
- `add_mosfet_with_body_diode` adds 2 branches.
- `add_mosfet` with custom R_on.
- `add_igbt` callable.
- Buck topology via the MOSFET helper builds + factors
  without raising.

## What V0 deliberately does NOT do

- **Shichman-Hodges Level 1 MOSFET** (`I_D = K (V_GS −
  V_T)²` in saturation). Requires a 3-terminal AD-driven
  channel model with the gate as a control input. Future
  research OpenSpec.
- **Gate-charge / switching-loss modeling** (`C_iss`,
  `C_oss`, `t_on`, `t_off`, `E_on`, `E_off`). V0 treats
  switching as instantaneous.
- **Temperature dependence** (R_on(T), V_F(T)). V0 is
  isothermal.
- **Three-terminal package with gate node**. V0 uses the
  switch_fn mechanism — gate signals come in via
  `switch_fn(t)`, not as an explicit gate node.

## Files

- MODIFIED `core/include/pulsim/builder/circuit_builder.hpp`
  (+ `add_mosfet`, `add_mosfet_with_body_diode`,
  `add_igbt`)
- MODIFIED `core/tests/v2/builder/test_circuit_builder.cpp`
  (+ 6 cases)
- MODIFIED `python/bindings_v2_kernel.cpp`
  (+ 3 Python bindings)
- MODIFIED `python/tests/v2/test_v2_python_bindings.py`
  (+ 5 cases)
- NEW `openspec/changes/pulsim-v2-mosfet-igbt/`
