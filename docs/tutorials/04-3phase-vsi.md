# Tutorial 04 — Three-phase voltage-source inverter (SPWM)

**Goal.** Build a 6-switch 3-leg VSI driving a balanced RL load with sinusoidal PWM. Verify that the three line-to-line voltages have the correct 120° relative phase and equal magnitude.

**Concepts introduced:**
- Switch-bit bookkeeping in `SwitchStateMask`: 6 MOSFETs with body diodes are 12 switching branches (`num_switches = 12` → 4096 possible masks).
- `make_three_phase_spwm_fn` + `ThreePhaseLegIndices` (Layer 2 V8).
- Why the cache stays manageable even with many switches (only states reachable from the schedule are built).

**Reference YAML:** [`examples/three_phase_inverter.yaml`](https://github.com/lgili/Pulsim/blob/main/examples/three_phase_inverter.yaml).

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
import pulsim as p
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

# Wye load (3 × (R + L) to neutral n). The neutral floats, as in a
# real motor or wye load: do NOT tie it to gnd. gnd is the DC-bus
# negative rail, so a neutral-to-gnd tie gives every phase a 300 V
# (V_DC/2) common-mode bias and drives ~60 A of DC through each
# phase resistor. The node is still referenced: it reaches the rails
# through the load and the switches.
for phase in ("a", "b", "c"):
    b.add_resistor(f"R_{phase}", phase, f"{phase}_int", 5.0)
    b.add_inductor(f"L_{phase}", f"{phase}_int", "n", 5e-3)
```

## SPWM driver

```python
# add_mosfet() adds each switch AND its body diode (`<name>_body`),
# and both are switching branches, so the six MOSFETs land on bits
# 0, 2, 4, 6, 8, 10 -- not 0..5. Ask the builder; never count by hand.
legs = p.ThreePhaseLegIndices(
    b.switch_index_of("Q1_H"), b.switch_index_of("Q1_L"),   # hs_a, ls_a
    b.switch_index_of("Q2_H"), b.switch_index_of("Q2_L"),   # hs_b, ls_b
    b.switch_index_of("Q3_H"), b.switch_index_of("Q3_L"),   # hs_c, ls_c
)
spwm = p.make_three_phase_spwm_fn(
    F_CARRIER,               # carrier_frequency
    F_MOD,                   # modulation_frequency
    M,                       # modulation_index
    legs,
    b.graph.num_switches,    # 12: six switches + six body diodes
    0.0,                     # dead_time [s]
)
```

The helper returns a `t -> SwitchStateMask(12)` callable that:
- Compares one carrier at `F_CARRIER` with three sinusoidal references at `F_MOD` (0°, −120°, −240° on legs A/B/C).
- Drives each leg's HS-LS pair complementarily, with the requested symmetric dead time (shoot-through is prevented by construction).
- Leaves the body-diode bits OFF; event detection turns them on when they must conduct.

`dead_time = 0` here because the fixed-step engine evaluates `switch_fn`
on the `dt = 2 µs` grid: a dead time shorter than `dt` is not
representable. To study dead-time distortion, shrink `dt` well below
the dead time first.

## Run + analyse

```python
res = p.simulate(b, t_end=100e-3, dt=2e-6, switch_fn=spwm)

a_idx = b.node_id_of("a")
b_idx = b.node_id_of("b")
c_idx = b.node_id_of("c")
import numpy as np

t = np.asarray(res.times)
X = np.asarray(res.states)
v_ab = X[:, a_idx] - X[:, b_idx]
v_bc = X[:, b_idx] - X[:, c_idx]
v_ca = X[:, c_idx] - X[:, a_idx]

# One full fundamental period after the start-up transient (from 50 ms).
win = (t >= 50e-3) & (t <= 50e-3 + 1.0 / F_MOD)
tw  = t[win]
T   = tw[-1] - tw[0]

def fundamental(v):
    """RMS and phase [deg] of the F_MOD Fourier component (time-weighted)."""
    vw = v[win]
    a = 2 / T * np.trapezoid(vw * np.cos(2 * np.pi * F_MOD * tw), tw)
    b = 2 / T * np.trapezoid(vw * np.sin(2 * np.pi * F_MOD * tw), tw)
    return math.hypot(a, b) / math.sqrt(2), math.degrees(math.atan2(a, b))

target = math.sqrt(3) / 2 * M * V_DC / math.sqrt(2)
for name, v in (("V_ab", v_ab), ("V_bc", v_bc), ("V_ca", v_ca)):
    rms1, ph = fundamental(v)
    print(f"{name}: fundamental {rms1:6.1f} V rms at {ph:+6.1f} deg (target {target:.1f} V)")
total = math.sqrt(np.trapezoid(v_ab[win] ** 2, tw) / T)
print(f"V_ab total RMS (fundamental + PWM harmonics) = {total:.1f} V")
```

Expected output:
```
V_ab: fundamental  313.3 V rms at  +30.1 deg (target 312.3 V)
V_bc: fundamental  313.0 V rms at  -90.0 deg (target 312.3 V)
V_ca: fundamental  312.5 V rms at +150.1 deg (target 312.3 V)
V_ab total RMS (fundamental + PWM harmonics) = 411.2 V
```

The three line voltages have equal fundamentals within 0.3 % and sit
120° apart (V_ab leads its phase-A reference by the textbook 30°). The
target is `(√3/2) · M · V_DC / √2 ≈ 312 V` for plain sinusoidal PWM
(no third-harmonic injection). Note the total RMS: a PWM line voltage
is a train of ±600 V pulses, so its RMS (411 V) is far above the
fundamental — take the Fourier component, not `sqrt(mean(v²))`, when
you want the fundamental.

## Cache size

This circuit has **12 switching branches → 4096 possible masks**. The cache is built lazily — only the masks the run actually reaches are factored. Pass `on_cache=` to `simulate()` to get the cache back and count them:

```python
caches = []
p.simulate(b, t_end=100e-3, dt=2e-6, switch_fn=spwm, on_cache=caches.append)
print(f"factored masks: {caches[0].num_built_segments()} of {2 ** b.graph.num_switches}")
```

Expected: `factored masks: 8 of 4096` — the 2³ leg states. The body
diodes never conduct here because, with zero dead time, the opposite
switch of each leg always carries the reversed load current.

## Try this next

- **Three-harmonic injection.** Replace the pure-sine modulator with `sin + (1/6)·sin(3·)`, the classic technique to lift `V_phase_max` from `0.5 · V_DC` to `0.577 · V_DC` without overmodulating.
- **Space-vector PWM.** SVPWM is harder to express with `make_three_phase_spwm_fn`; you can either write a custom `switch_fn` lambda OR build a `make_combined_switch_fn(...)` from per-leg helpers. See V10.
- **Field-oriented control.** Drive a permanent-magnet synchronous motor model (via the saturable-inductor + transformer pair) and close the dq-loop in Python.

## Going to [Tutorial 05](05-ldo-feedback.md)

Next we leave switching and look at a linear-regulator topology: a high-gain VCVS + pass MOSFET + RC feedback.
