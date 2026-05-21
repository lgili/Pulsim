# Tutorial 04 — Three-phase voltage-source inverter (SPWM)

**Goal.** Build a 6-switch 3-leg VSI driving a balanced RL load with sinusoidal PWM. Verify that the three line-to-line voltages have the correct 120° relative phase and equal magnitude.

**Concepts introduced:**
- 6-switch enumeration in `SwitchStateMask` (`num_switches = 6` → 64 cache entries).
- `make_three_phase_spwm_fn` + `ThreePhaseLegIndices` (Layer 2 V8).
- Why the cache stays manageable even with many switches (only states reachable from the schedule are built).

**Reference YAML:** [`examples/v2/three_phase_inverter.yaml`](https://github.com/lgili/Pulsim/blob/main/examples/v2/three_phase_inverter.yaml).

## Topology

```
   V_dc ─┬── Q1_H ──┬── Q2_H ──┬── Q3_H ──
         │          │          │
         │ phase A  │ phase B  │ phase C
         │          │          │
   gnd ──┴── Q1_L ──┴── Q2_L ──┴── Q3_L ──
                │          │          │
              (phase outputs go to L-R load)
```

Three half-bridge legs. Each leg has a high-side and a low-side switch driven complementarily by an SPWM modulator with the appropriate 120° phase shift.

Load: a 3-phase wye-connected RL (5 Ω + 5 mH per phase, common neutral).

## Python build

```python
import pulsim.v2 as p
import math

V_DC = 600.0
F_CARRIER = 10e3   # PWM carrier
F_MOD     = 60.0   # AC output fundamental
M         = 0.85   # modulation index ∈ [0, 1]

b = p.CircuitBuilder()
b.add_voltage_source("Vdc", "dc_pos", "gnd", V_DC)

# 6 switches.
b.add_mosfet("Q1_H", "dc_pos", "a")
b.add_mosfet("Q1_L", "a",      "gnd")
b.add_mosfet("Q2_H", "dc_pos", "b")
b.add_mosfet("Q2_L", "b",      "gnd")
b.add_mosfet("Q3_H", "dc_pos", "c")
b.add_mosfet("Q3_L", "c",      "gnd")

# Wye load (3 × (R + L) to neutral n).
for phase in ("a", "b", "c"):
    b.add_resistor(f"R_{phase}", phase, f"{phase}_int", 5.0)
    b.add_inductor(f"L_{phase}", f"{phase}_int", "n", 5e-3)
# Neutral pinned to gnd by a small resistor (numerical convenience).
b.add_resistor("R_n", "n", "gnd", 1e-3)
```

## SPWM driver

```python
legs = p.ThreePhaseLegIndices(
    a_high=0, a_low=1,
    b_high=2, b_low=3,
    c_high=4, c_low=5,
)
spwm = p.make_three_phase_spwm_fn(
    leg_indices=legs,
    carrier_frequency=F_CARRIER,
    modulating_frequency=F_MOD,
    modulation_index=M,
    phase=0.0,
)
```

The helper emits a `SwitchStateMask(6)` that:
- Compares a triangular carrier at `F_CARRIER` with three sinusoidal modulators at `F_MOD`, each phase-shifted by 120°.
- Drives each leg's HS-LS pair complementarily.

## Run + analyse

```python
res = p.simulate(b, t_end=100e-3, dt=2e-6, switch_fn=spwm)

a_idx = b.node_id_of("a")
b_idx = b.node_id_of("b")
c_idx = b.node_id_of("c")
v_ab = [s[a_idx] - s[b_idx] for s in res.states]
v_bc = [s[b_idx] - s[c_idx] for s in res.states]
v_ca = [s[c_idx] - s[a_idx] for s in res.states]

# After ~3 cycles, extract one full fundamental period.
import math
T_mod = 1.0 / F_MOD
k0 = int(50e-3 / 2e-6)
k1 = k0 + int(T_mod / 2e-6)

# Fundamental RMS via mean(v²)·2.
def rms(samples):
    return math.sqrt(sum(v*v for v in samples) / len(samples))

print(f"V_ab fundamental RMS ≈ {rms(v_ab[k0:k1]):.1f} V (target ~{0.612 * M * V_DC:.0f} V)")
```

For `M = 0.85` and `V_DC = 600`, expected fundamental line-to-line RMS is `(√3/2) · M · V_DC / √2 ≈ 312 V` (sinusoidal modulation, not third-harmonic-injected).

## Cache size

This circuit has **6 switches → 64 possible masks**. The cache is built lazily — only masks actually reached by the SPWM scheduler are factored, which is usually ~12-16 of the 64. You can check `cache.dt()` and the runtime stamping cost stays bounded.

## Try this next

- **Three-harmonic injection.** Replace the pure-sine modulator with `sin + (1/6)·sin(3·)`, the classic technique to lift `V_phase_max` from `0.5 · V_DC` to `0.577 · V_DC` without overmodulating.
- **Space-vector PWM.** SVPWM is harder to express with `make_three_phase_spwm_fn`; you can either write a custom `switch_fn` lambda OR build a `make_combined_switch_fn(...)` from per-leg helpers. See V10.
- **Field-oriented control.** Drive a permanent-magnet synchronous motor model (via the saturable-inductor + transformer pair) and close the dq-loop in Python.

## Going to [Tutorial 05](05-ldo-feedback.md)

Next we leave switching and look at a linear-regulator topology: a high-gain VCVS + pass MOSFET + RC feedback.
