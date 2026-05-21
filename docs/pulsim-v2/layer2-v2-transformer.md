# Layer 2 V2 — Two-winding transformer (coupled inductors)

V2 already shipped R / L / C / sources / diodes / switches /
MOSFETs / IGBTs (Layer 2 V1). V2 of Layer 2 adds **magnetic
coupling** — the missing piece for isolated SMPS
topologies (flyback, forward, push-pull, full-bridge).

## API

```cpp
b.add_transformer(
    "T1",
    "p+", "p-",         // primary terminals
    "s+", "s-",         // secondary terminals
    /*L_p=*/100e-6,     // primary self-inductance, H
    /*L_s=*/25e-6,      // secondary self-inductance, H
    /*k=*/0.95);        // coupling coefficient ∈ [0, 1]
```

Identical signature from Python:

```python
b.add_transformer("T1", "p+", "p-", "s+", "s-",
                   L_p=100e-6, L_s=25e-6, k=0.95)
```

## Model

```
        i_p →    M    ← i_s
       ┌────┐ ─ ─ ─ ─ ┌────┐
   v_p │ L_p│          │ L_s│ v_s
       └────┘ ─ ─ ─ ─ └────┘
```

Constitutive equations:

```
v_p = L_p · di_p/dt + M · di_s/dt
v_s = M  · di_p/dt + L_s · di_s/dt

with M = k · √(L_p · L_s),   k ∈ [0, 1]
```

| k value | Behaviour |
|---------|-----------|
| `k = 0` | Independent inductors (no transformer action). |
| `0 < k < 1` | Real transformer with leakage. |
| `k = 1` | Ideal coupling (no leakage). |

## How it's stamped

Internally, `add_transformer` creates two ordinary inductor
branches AND registers a "coupling pair" in `DevicePool`.
The assemble + history passes apply cross-coupling on top:

```
J[p_constraint_row, s_branch_var_col] += -(2M/dt)
J[s_constraint_row, p_branch_var_col] += -(2M/dt)
b_extra[p_constraint_row] += (2M/dt) · i_s_prev
b_extra[s_constraint_row] += (2M/dt) · i_p_prev
```

The single-inductor self-terms come from each branch's
existing trap-companion stamp. The mutual-coupling
overlay is added as a SECOND PASS after the per-branch
loop, keeping the change to `assemble.hpp` minimal.

For `dt = 0` (static path), couplings have no effect —
ideal transformers are open circuits at DC, matching the
standalone inductor's static-path behaviour.

## Use cases

### Isolated SMPS (flyback)

```cpp
b.add_voltage_source("Vin", "vin",      "gnd",      24.0);
b.add_mosfet_with_body_diode(
    "Q1",  "vin",     "sw");
b.add_transformer ("T1",
                    "sw",     "gnd",         // primary
                    "sec_pos","sec_neg",     // secondary
                    /*L_p=*/100e-6,
                    /*L_s=*/25e-6,
                    /*k=*/0.95);
b.add_diode       ("D1", "sec_pos", "vout", 1e3, 1e-9, 0.7);
b.add_capacitor   ("Cout", "vout",  "sec_neg", 47e-6);
b.add_resistor    ("R_L",  "vout",  "sec_neg", 5.0);
b.add_resistor    ("Rgnd", "sec_neg", "gnd", 1e-6);
```

`Q1` switches the primary; the transformer transfers energy
to the secondary; `D1` rectifies; `Cout` smooths; `R_L` is
the load.

### Current-sense transformer

```cpp
b.add_transformer("CT1",
                   "pri_in", "pri_out",   // primary in series
                   "sec_high", "sec_low",
                   /*L_p=*/1e-9,    // primary ~0 turns
                   /*L_s=*/10e-6,
                   /*k=*/0.98);
b.add_resistor("R_burden", "sec_high", "sec_low", 100.0);
```

A small primary inductance (1 nH) effectively acts as a
"current sensor" — primary current is mirrored to the
secondary through the coupling, and `R_burden` converts
it to a voltage signal.

## Test coverage

**C++** — 7 cases in
`core/tests/v2/layer5_v1/test_transformer.cpp`:

1. `mutual_inductance` computes M = k·√(L_p·L_s) (with
   k clamping to [0, 1]).
2. `cross_dt` is symmetric under L_p ↔ L_s swap.
3. `add_transformer` smoke: 2 branches + 1 coupling.
4. Builder chaining with other devices.
5. Ideal 1:1 transformer: states stay finite and
   bounded under a V_dc step.
6. k=0 isolation: secondary stays at quiescent voltage
   even with primary excitation (< 1µV deviation).
7. Realistic flyback parameters (100µH:25µH, k=0.95): no
   NaN/Inf, states bounded.

**Python** — 2 cases in
`python/tests/v2/test_v2_python_bindings.py`:

8. `add_transformer` callable from Python; 2 branches.
9. Transformer topology builds and KLU-factors.

## What V0 deliberately does NOT do

- **Saturation modeling**: V0 treats L as constant.
  Ferrite cores saturate at the knee; that's a future
  AD-driven L(i) model.
- **Multi-winding transformer (>2 windings)**: V0 is
  two-winding. Three-winding (push-pull centre-tap, or
  flyback with bias winding) is a straightforward
  extension via more coupling pairs; V1.
- **Frequency-dependent core losses**: V0 is lossless
  apart from the optional leakage from k < 1.
- **Hysteresis**: not modeled (would need a B-H
  curve and history tracking on the flux).
- **Magnetic-equivalent-circuit** (reluctance) approach:
  V0 uses inductor-pair stamping. The reluctance
  formulation is more natural for multi-leg cores but
  adds machinery.

## Files

- NEW `core/include/pulsim/v2/models/transformer.hpp`
- MODIFIED `core/include/pulsim/v2/pwl/device_pool.hpp`
  (+ coupling registry)
- MODIFIED `core/include/pulsim/v2/pwl/assemble.hpp`
  (+ cross-term pass)
- MODIFIED `core/include/pulsim/v2/pwl/history_state.hpp`
  (+ cross-current history pass)
- MODIFIED `core/include/pulsim/v2/builder/circuit_builder.hpp`
  (+ add_transformer)
- MODIFIED `python/bindings_v2_kernel.cpp`
  (+ Python add_transformer)
- NEW `core/tests/v2/layer5_v1/test_transformer.cpp`
  (7 cases)
- MODIFIED `python/tests/v2/test_v2_python_bindings.py`
  (+ 2 cases)
- MODIFIED `core/CMakeLists.txt`
  (transformer test added to layer5_v1 target)
- NEW `openspec/changes/pulsim-v2-transformer/`
