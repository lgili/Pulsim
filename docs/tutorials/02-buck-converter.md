# Tutorial 02 — Buck converter (YAML + switch_fn)

**Goal.** Build an open-loop buck converter, drive its high-side switch with a 100 kHz / 50 % duty PWM, verify that the output voltage equals `D · V_in` in steady state.

**Concepts introduced:**
- Switch + free-wheeling diode topology.
- `make_pwm_switch_fn` (Layer 2 V5).
- Settling vs. ripple: how to extract the mean output voltage from a transient trace.

**Reference YAML:** [`examples/buck.yaml`](https://github.com/lgili/Pulsim/blob/main/examples/buck.yaml).

## Topology

```
     Vin ────┬─── Q1 ───┬─── L ───┬───── vout ── load
             │  (HS switch)       │
             │           ▲        │
             │           │ D1     │  C
             │     (free-wheel)   │
             ⏚         ⏚        ⏚
```

Components:

| Name | Type | Value |
|---|---|---|
| `Vin` | DC source | 24 V |
| `Q1`  | Switch (drain→source) | `R_on = 1 mΩ`, `R_off = 1 GΩ` |
| `D1`  | SwitchedDiode (gnd → sw) | `g_on = 1 kS`, `V_th = 0.7 V` (commutation threshold) |
| `L`   | Inductor (sw → vout) | 100 µH |
| `Cout`| Capacitor (vout → gnd) | 47 µF |
| `R_L` | Load resistor (vout → gnd) | 10 Ω |

PWM: 100 kHz, 50 % duty → expected steady-state `V_out ≈ D · V_in = 0.5 · 24 = 12 V`.

`V_th` on a switched diode is the voltage at which it *commutates*, not a
forward drop: once ON it is a 1 mΩ conductance (`g_on = 1e3` S) with no
offset, and `R_on = 1 mΩ` on Q1 drops ~1 mV at 1.2 A. The losses are
therefore negligible and the run lands on `D · V_in` itself. Model a real
forward drop with a device that has one (e.g. a Shockley diode) if you
want to see it in `V_out`.

## Python build

```python
import pulsim as p

V_IN  = 24.0
F_PWM = 100e3
DUTY  = 0.50

b = p.CircuitBuilder()
b.add_voltage_source        ("Vin", "vin", "gnd", V_IN)
b.add_mosfet_with_body_diode("Q1",  "vin", "sw")
b.add_diode                 ("D1",  "gnd", "sw", 1e3, 1e-9, V_th=0.7)
b.add_inductor              ("L",   "sw", "vout", 100e-6)
b.add_capacitor             ("Cout","vout", "gnd", 47e-6)
b.add_resistor              ("R_L", "vout", "gnd", 10.0)
```

Note: `add_mosfet_with_body_diode` creates BOTH a Switch branch (drain→source) AND an anti-parallel `SwitchedDiode`, so the inductor current has somewhere to go during the OFF interval.

## Switch driver

```python
# Ask the builder for Q1's bit instead of hard-coding it. Switch bits
# follow the builder-call order of every switching branch -- here
# Q1 (bit 0), Q1's body diode (bit 1) and the free-wheel diode D1
# (bit 2) -- so inserting or reordering a device silently moves the
# bit. `graph.num_switches` sizes the mask.
q1_bit = b.switch_index_of("Q1")
n_sw   = b.graph.num_switches
pwm = p.make_pwm_switch_fn(F_PWM, DUTY, q1_bit, n_sw, phase=0.0)
```

`make_pwm_switch_fn` returns a `t -> SwitchStateMask` callable that turns bit `q1_bit` ON during the first 50 % of each cycle and OFF for the rest. The body diode and free-wheel diode bits are managed by the event-detection loop inside `run_transient` (no manual driving needed).

## Run

```python
import numpy as np

res = p.simulate(b, t_end=5e-3, dt=2e-7, switch_fn=pwm)

vout_idx = b.node_id_of("vout")
t    = np.asarray(res.times)
vout = np.array([s[vout_idx] for s in res.states])
# Skip the first 2 ms transient and time-average the rest
# (np.trapezoid stays correct if the run is ever variable-step).
m = t >= 2e-3
v_mean = np.trapezoid(vout[m], t[m]) / (t[m][-1] - t[m][0])
print(f"V_out (steady-state mean) = {v_mean:.2f} V (target D·V_in = 12.0 V)")
```

Expected output:
```
V_out (steady-state mean) = 12.00 V (target D·V_in = 12.0 V)
```

The output filter (100 µH / 47 µF / 10 Ω) is lightly damped (ζ ≈ 0.07), so
at 2–5 ms it is still ringing from the start-up. The window mean is already
on target, but do not read the ripple from this run: `make_pwm_switch_fn`
compares a floating-point phase, so on the `dt` grid some gate edges land
one step early or late, and that duty jitter keeps the filter ringing. At
9–10 ms of a 10 ms run it still shows ≈ 0.12 V peak-to-peak, where a PWM
computed in whole steps (`round(t / dt) % 50 < 25`) shows the ≈ 16 mV
switching ripple that `ΔI / (8 f C)` predicts. See trap 10 and section 3.2
of `python -m pulsim.guide`.

## What's going on internally

1. **Cache.** `Q1`, its body diode and `D1` are three switching branches, so there are up to 2³ = 8 mask combinations. The cache is lazy: it factors the MNA matrix of a combination the first time the run reaches it and reuses it on every later visit.
2. **PWM scheduling.** At each `dt`, `pwm(t)` returns the mask for Q1's state. The cache looks up the matching pre-factored entry in O(1).
3. **Inductor commutation.** When `Q1` opens, `i_L` must keep flowing. `D1` switches ON automatically (event detection on `v_D1` changing sign) and the integration restarts from the new switch state at the exact bisected commutation time.

## Reading the YAML

[`examples/buck.yaml`](https://github.com/lgili/Pulsim/blob/main/examples/buck.yaml) declares the same power stage (with a 5 Ω load and the free-wheel diode named `D_FW`). The block below carries that file's content inline, so it runs without a clone of the repository; with a clone, `p.load_yaml_file("examples/buck.yaml")` (path relative to the repository root) gives the same result.

The YAML holds the circuit and the run settings, not the gate drive.
`p.load_yaml_string(...)` / `p.load_yaml_file(...)` return a `LoadedCircuit`
with `.builder` and `.options`; build the PWM from the loaded builder
exactly as above:

```python
BUCK_YAML = """
circuit:
  devices:
    - {type: voltage_source,         name: Vin,  from: vin, to: gnd, V: 24.0}
    - {type: mosfet_with_body_diode, name: Q1,   drain: vin, source: sw,
       R_on: 1.0e-3, R_off: 1.0e9, V_F: 0.7}
    - {type: diode,                  name: D_FW, anode: gnd, cathode: sw,
       g_on: 1.0e3, g_off: 1.0e-9, V_th: 0.7}
    - {type: inductor,               name: L1,   from: sw, to: vout, L: 100.0e-6}
    - {type: capacitor,              name: Cout, from: vout, to: gnd, C: 47.0e-6}
    - {type: resistor,               name: R_L,  from: vout, to: gnd, R: 5.0}
simulation:
  t_start: 0.0
  t_end: 5.0e-3
  dt: 1.0e-7
"""
lc  = p.load_yaml_string(BUCK_YAML)
pwm = p.make_pwm_switch_fn(100e3, 0.5, lc.builder.switch_index_of("Q1"),
                           lc.builder.graph.num_switches)
res = p.simulate(lc.builder, t_end=lc.options.t_end, dt=lc.options.dt, switch_fn=pwm)
```

## Try this next

- **Change duty.** Set `DUTY = 0.25` → `V_out ≈ 5.82 V`; `DUTY = 0.75` → `V_out ≈ 17.82 V` (means over 9–10 ms of a 10 ms run). Both sit 0.18 V below `D · V_in` because the fixed-step engine evaluates `switch_fn` on the `dt` grid: an on-time of `0.25 · 10 µs = 2.5 µs` is 12.5 steps of 0.2 µs and is not representable. Halve `dt` to `1e-7` (25 steps) and the means become 5.99 V and 17.99 V. Pick `dt` so that `D · T` is a whole number of steps.
- **Vary frequency.** Drop to `F_PWM = 20 kHz` and watch the output ripple grow.
- **Continuous → discontinuous conduction.** Raise the load resistance to 100 Ω. The inductor current goes to zero each cycle; the diode bit switches OFF too. The cache handles this without any code change.
- **Add closed-loop control.** Replace the constant `DUTY` with a duty-cycle command that depends on `v_out` (via your own switch_fn lambda). Welcome to compensator design.

## Going to [Tutorial 03](03-flyback-isolated.md)

The next tutorial introduces the transformer (Layer 2 V2) and shows the classic 2-switch-state flyback converter.
