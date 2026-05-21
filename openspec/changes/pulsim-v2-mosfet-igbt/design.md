# Design — `pulsim-v2-mosfet-igbt` (Layer 2 V1)

## Why MOSFET ≠ "just a switch"

V6's `add_switch` covers the topology — a controlled
2-terminal device with on/off conductance. But SMPS engineers
think of MOSFETs and IGBTs as DEVICES, not primitives:

- **MOSFET (n-channel power)**: R_DS_on (a few mΩ), R_DS_off
  (typically MΩ-GΩ), an intrinsic anti-parallel BODY DIODE
  with forward voltage ~0.5–1.0 V. The body diode matters
  during freewheeling and dead-time intervals.
- **IGBT**: Higher R_on (mΩ–tens of mΩ), no body diode in
  most discrete packages (anti-parallel diode is added
  externally in modules). Saturation voltage ~1–2 V.

A `CircuitBuilder` user shouldn't have to:
1. Convert R_on to g_on = 1/R_on by hand.
2. Remember to add the body diode as a SEPARATE branch.
3. Get the body-diode direction right (source → drain for
   n-channel MOSFETs).

V1 ships three convenience methods that bake this in.

## API

```cpp
CircuitBuilder& add_mosfet(
    std::string name, std::string drain, std::string source,
    Real R_on  = Real{1e-3},
    Real R_off = Real{1e9});

CircuitBuilder& add_mosfet_with_body_diode(
    std::string name, std::string drain, std::string source,
    Real R_on        = Real{1e-3},
    Real R_off       = Real{1e9},
    Real V_F         = Real{0.7},
    Real g_on_diode  = Real{1e3},
    Real g_off_diode = Real{1e-9});

CircuitBuilder& add_igbt(
    std::string name, std::string collector,
    std::string emitter,
    Real R_on  = Real{10e-3},
    Real R_off = Real{1e9});
```

## Semantics

### `add_mosfet`

Single controlled switch on branch (drain → source) with
`g_on = 1/R_on`, `g_off = 1/R_off`. Equivalent to:

```cpp
add_switch(name, drain, source, 1/R_on, 1/R_off);
```

The MOSFET defaults (1 mΩ / 1 GΩ) are SMPS-typical for
modern Si power devices.

### `add_mosfet_with_body_diode`

Adds TWO branches in sequence:

1. Main switch (drain → source) with `g_on = 1/R_on`,
   `g_off = 1/R_off`.
2. Body diode (source → drain) — anti-parallel — with
   `g_on = g_on_diode`, `g_off = g_off_diode`, `V_th = V_F`.

The body diode automatically conducts during freewheeling
intervals and clamps when the switch is OFF and the load
forces reverse current.

Equivalent to:

```cpp
add_switch(name + "_main", drain, source, 1/R_on, 1/R_off);
add_diode (name + "_body", source, drain,
            g_on_diode, g_off_diode, V_F);
```

The implementation uses `name + "_main"` and `name + "_body"`
internally; users may pass their preferred name as the
single `name` argument.

### `add_igbt`

Single controlled switch on branch (collector → emitter).
Same shape as `add_mosfet` but with IGBT defaults (10 mΩ /
1 GΩ). No body diode added — IGBT modules typically include
an external anti-parallel diode that the user wires
explicitly with `add_diode` if needed.

## Default values — engineering rationale

| Param | Default | Reasoning |
|-------|---------|-----------|
| MOSFET R_on | 1 mΩ | Typical n-channel power MOSFET (e.g. IRFB4127) |
| MOSFET R_off | 1 GΩ | Practical leakage cap; doesn't dominate quiescent current |
| MOSFET V_F (body) | 0.7 V | Si Schottky / body diode |
| MOSFET g_on (body) | 1e3 S | Matches `add_diode` default |
| IGBT R_on | 10 mΩ | Typical IGBT module on-state |

These are V0 defaults — power-electronics engineers override
freely. The point is to make `add_mosfet("Q1", "vin", "sw")`
work without forcing users to think about typical numbers.

## Test plan

In `core/tests/v2/builder/test_circuit_builder.cpp`:

1. **`add_mosfet` smoke**: 1 branch added, kind = Switch,
   `switch_g_on(branch_id) == 1/R_on`.
2. **`add_mosfet_with_body_diode` smoke**: 2 branches
   added. Branch 0 = Switch (drain → source); branch 1 =
   Diode with the `Switch` kind but the `Diode` stored
   variant in the pool, antiparallel direction (source →
   drain), V_th == V_F.
3. **`add_igbt` smoke**: 1 branch, switch defaults.
4. **Buck converter parity**: build the V1.5 buck via
   `add_mosfet_with_body_diode` for the high-side MOSFET +
   `add_diode` for the low-side freewheeling diode; verify
   sample-by-sample parity with manual setup.

In `python/tests/v2/test_v2_python_bindings.py`:
5. **Python smoke**: `b.add_mosfet(...)` and
   `b.add_mosfet_with_body_diode(...)` callable; correct
   `num_branches` post-call.

## What V0 deliberately does NOT do

- **Shichman-Hodges Level 1 MOSFET** (I_D = K (V_GS − V_T)²
  in saturation): this would require a 3-terminal AD-driven
  model with the gate as a control input. V0 keeps the
  switch-with-defaults wrapping; the AD-driven channel
  model is a future research OpenSpec.
- **Gate-charge / switching-loss modeling**: V0 treats the
  device as instant-switching. C_iss, C_oss, t_on, t_off,
  E_on, E_off are simulation-time-dependent and require
  the substep state correction wired into the Newton path
  (Layer 5 V3 covers the trap-companion substep; Newton
  substep is V1).
- **Temperature dependence**: V0 is isothermal.
- **Three-terminal package** (gate as a control node): V0
  uses the switch_fn mechanism (gate signals come in via
  the user's `switch_fn(t)`), not an explicit gate node.

## Files

- MODIFIED `core/include/pulsim/v2/builder/circuit_builder.hpp`
- MODIFIED `core/tests/v2/builder/test_circuit_builder.cpp`
- MODIFIED `python/bindings_v2_kernel.cpp`
- MODIFIED `python/tests/v2/test_v2_python_bindings.py`
- NEW `docs/pulsim-v2/layer2-v1-mosfet-igbt.md`
