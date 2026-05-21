# Tutorial 02 — Buck converter (YAML + switch_fn)

**Goal.** Build an open-loop buck converter, drive its high-side switch with a 100 kHz / 50 % duty PWM, verify that the output voltage equals `D · V_in` in steady state.

**Concepts introduced:**
- Switch + free-wheeling diode topology.
- `make_pwm_switch_fn` (Layer 2 V5).
- Settling vs. ripple: how to extract the mean output voltage from a transient trace.

**Reference YAML:** [`examples/v2/buck.yaml`](https://github.com/lgili/Pulsim/blob/main/examples/v2/buck.yaml).

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
| `D1`  | SwitchedDiode (gnd → sw) | `V_F = 0.7 V` |
| `L`   | Inductor (sw → vout) | 100 µH |
| `Cout`| Capacitor (vout → gnd) | 47 µF |
| `R_L` | Load resistor (vout → gnd) | 10 Ω |

PWM: 100 kHz, 50 % duty → expected steady-state `V_out ≈ 0.5 · 24 = 12 V` (minus diode drop / R_on losses ≈ 11.6 V).

## Python build

```python
import pulsim.v2 as p

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
# Switch index 0 is the HS MOSFET. Bit 1 is the body diode (handled
# automatically by the cache + event detection). The free-wheel
# diode D1 is a SwitchedDiode (kind == Switch from a topology view)
# — its bit is bit 2.
pwm = p.make_pwm_switch_fn(
    switch_index=0, frequency=F_PWM, duty=DUTY, phase=0.0,
)
```

`make_pwm_switch_fn` returns a `t -> SwitchStateMask` callable that flips bit 0 ON during the first 50 % of each cycle and OFF for the rest. The body diode and free-wheel diode bits are managed by the event-detection loop inside `run_transient` (no manual driving needed).

## Run

```python
res = p.simulate(b, t_end=5e-3, dt=2e-7, switch_fn=pwm)

vout_idx = b.node_id_of("vout")
vout_samples = [s[vout_idx] for s in res.states]
# Skip the first 2 ms transient and average over the rest.
k_skip = int(2e-3 / 2e-7)
v_mean = sum(vout_samples[k_skip:]) / (len(vout_samples) - k_skip)
print(f"V_out (steady-state mean) = {v_mean:.2f} V (target 11.5–12.0 V)")
```

Expected output:
```
V_out (steady-state mean) = 11.79 V (target 11.5–12.0 V)
```

## What's going on internally

1. **Cache enumeration.** `Q1` and `D1` give 2 switches × 2 states = 4 reachable mask combinations. The cache pre-factors 4 MNA matrices.
2. **PWM scheduling.** At each `dt`, `pwm(t)` returns the mask for Q1's state. The cache looks up the matching pre-factored entry in O(1).
3. **Inductor commutation.** When `Q1` opens, `i_L` must keep flowing. `D1` switches ON automatically (event detection on `v_D1` changing sign) and the integration restarts from the new switch state at the exact bisected commutation time.

## Reading the YAML

[`examples/v2/buck.yaml`](https://github.com/lgili/Pulsim/blob/main/examples/v2/buck.yaml) declares the same circuit + PWM with:

```yaml
sources:
  - name: pwm0
    type: pwm
    switch_index: 0
    frequency: 100e3
    duty: 0.5

components:
  - {type: voltage_source, name: Vin, from: vin, to: gnd, V: 24}
  - {type: switch,         name: Q1,  from: vin, to: sw}
  - {type: diode,          name: D1,  from: gnd, to: sw, V_th: 0.7}
  - {type: inductor,       name: L,   from: sw,  to: vout, L: 100e-6}
  ...
```

Load it from Python with `p.load_yaml_file(...)`. Both `LoadedCircuit.builder` and `LoadedCircuit.switch_fn` are ready to pass straight to `simulate()`.

## Try this next

- **Change duty.** Set `DUTY = 0.25` → `V_out ≈ 6 V`. Set `DUTY = 0.75` → `V_out ≈ 18 V`. The output tracks `D · V_in` minus losses.
- **Vary frequency.** Drop to `F_PWM = 20 kHz` and watch the output ripple grow.
- **Continuous → discontinuous conduction.** Raise the load resistance to 100 Ω. The inductor current goes to zero each cycle; the diode bit switches OFF too. The cache handles this without any code change.
- **Add closed-loop control.** Replace the constant `DUTY` with a duty-cycle command that depends on `v_out` (via your own switch_fn lambda). Welcome to compensator design.

## Going to [Tutorial 03](03-flyback-isolated.md)

The next tutorial introduces the transformer (Layer 2 V2) and shows the classic 2-switch-state flyback converter.
