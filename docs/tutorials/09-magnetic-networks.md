# Tutorial 09 — Magnetic networks: a three-limb transformer and a coupled inductor

A core with several legs is a circuit of its own. Pulsim models it in
gyrator–capacitor form, on magnetic nodes beside the electrical ones:

| Magnetic quantity | Circuit quantity |
|---|---|
| MMF `F` [A-turn] | node potential |
| flux rate `dΦ/dt` [V] | branch current |
| permeance `P` [H] (`add_permeance`, `add_air_gap`) | capacitor |
| saturable limb (`add_saturable_permeance`) | nonlinear capacitor |
| eddy loss (`add_magnetic_resistance`) | resistor |
| winding of `N` turns (`add_winding`) | gyrator `g = 1/N` |

`add_winding(name, e_pos, e_neg, m_pos, m_neg, N)` sets
`F(m_pos) − F(m_neg) = N·i`, with `i` into `e_pos`. Each magnetic
island must touch `gnd`, or the circuit is refused. All parameters
below are synthetic illustrations, not a catalogue core. The model has
no fringing and no core-shape library: give each section its own area,
length and permeability.

For one flux path with a magnetising branch, the T-model
(`add_saturable_transformer`, `add_transformer`) is enough. Use a
network when legs share flux: multi-limb transformers, integrated
magnetics, or a gap you want to see.

## A three-limb transformer under unbalanced load

Each limb carries a 1000-turn primary and a 100-turn secondary in
series on its flux path, and a saturable section to the top yoke `T`.
The bottom yoke is `gnd`. Zero-sequence flux has no iron return on a
three-limb core: it closes through the air, modelled as a 20 nH
permeance from `T` to `gnd`.

```python
import math
import numpy as np
import pulsim as p

F_LINE, V_PK = 50.0, 325.0          # 230 V rms per phase
N_P, N_S = 1000, 100
AREA, LENGTH = 10e-4, 0.25          # limb section: 10 cm^2, 25 cm
MU_R, B_SAT = 2500.0, 1.5

def three_limb(loads):
    b = p.CircuitBuilder()
    for k, ph in enumerate("abc"):
        b.add_sine_voltage_source(f"V{ph}", f"g{ph}", "gnd", 0.0, V_PK,
                                  F_LINE, -2 * math.pi * k / 3)
        b.add_resistor(f"Rp{ph}", f"g{ph}", f"p{ph}", 2.0)
        b.add_winding(f"Wp{ph}", f"p{ph}", "gnd", f"y{ph}", "gnd", N_P)
        b.add_winding(f"Ws{ph}", f"s{ph}", "gnd", f"x{ph}", f"y{ph}", N_S)
        b.add_resistor(f"R{ph}", f"s{ph}", "gnd", loads[k])
        b.add_saturable_permeance(f"Limb{ph}", f"x{ph}", "T", AREA, LENGTH,
                                  mu_r=MU_R, B_sat=B_SAT)
    b.add_permeance("Air", "T", "gnd", 20e-9)
    return b

b = three_limb(loads=(5.0, 10.0, 50.0))
print(b.magnetic_nodes())
```

```text
['ya', 'xa', 'T', 'yb', 'xb', 'yc', 'xc']
```

### Start from the steady-state flux

Switching a transformer on at zero flux leaves a DC flux offset in each
limb that decays slowly through the winding resistance: the inrush.
Run from zero, limb a still averages 0.53 T over 60–100 ms, and its
primary current peaks above 50 A. To study the steady state, start each
limb at its steady-state flux `−Φ_pk·cos φ`. The initial state of a
saturable section is its MMF; invert its law
(`magnetic_element(...)["law"].charge`) by bisection:

```python
law = b.magnetic_element("Limba")["law"]

def mmf_for(phi):
    lo, hi = -1e5, 1e5
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if law.charge(mid) < phi else (lo, mid)
    return 0.5 * (lo + hi)

phi_pk = V_PK / (2 * math.pi * F_LINE * N_P)
x0 = np.zeros(len(b.initial_state()))
for k, ph in enumerate("abc"):
    F0 = mmf_for(-phi_pk * math.cos(-2 * math.pi * k / 3))
    x0[b.node_id_of(f"x{ph}")] = F0     # the limb's MMF ...
    x0[b.node_id_of(f"y{ph}")] = F0     # ... carried by the primary
```

`set_initial` does not set a saturable section (it records capacitor
and inductor values only): pass the vector as `initial_state=`.

### Flux per limb

`res.b_field(name)` is the flux of a section divided by its area;
`res.flux(name)` and `res.h_field(name)` give `Φ` and `F/length`.

```python
res = p.simulate(b, t_end=0.1, dt=2e-5, engine="pwl", initial_state=x0)
t = np.asarray(res.times)

def rms(y):
    return math.sqrt(p.time_average(t, np.asarray(y) ** 2, t_from=0.06))

for ph in "abc":
    B = res.b_field(f"Limb{ph}")[t >= 0.06]
    print(f"{ph}: B_pk {B.max():.2f} T, V_s {rms(res.v('s' + ph)):.1f} V, "
          f"I_s/10 {rms(res.i('R' + ph)) / 10:.3f} A, "
          f"I_p {rms(res.i('Rp' + ph)):.3f} A")
air = np.max(np.abs(res.flux("Air")))
print(f"air-path flux {air * 1e6:.2f} uWb, F_T {air / 20e-9:.0f} A-turn")
```

```text
a: B_pk 1.03 T, V_s 22.9 V, I_s/10 0.458 A, I_p 0.434 A
b: B_pk 1.04 T, V_s 22.9 V, I_s/10 0.229 A, I_p 0.216 A
c: B_pk 1.03 T, V_s 23.0 V, I_s/10 0.046 A, I_p 0.153 A
air-path flux 2.46 uWb, F_T 123 A-turn
```

The secondary voltages stay balanced, but the primary currents are not
the load currents over the turns ratio. The unbalanced load has a
zero-sequence part; its flux finds no iron return and drives the yoke
potential `F_T` through the air path. `F_T` adds to the ampere-turns
of every limb, and the primaries carry the difference. A T-model per
phase cannot show this: it has no shared yoke.

The flux leaving the yoke equals the flux entering it at every step, up
to a per-step residual that accumulates (2.5e-9 of the peak after the
5000 steps here; about 1e-12 for a short linear run):

```python
net = sum(res.flux(f"Limb{ph}") for ph in "abc") - res.flux("Air")
print("yoke flux balanced:", bool(np.max(np.abs(net)) < 1e-8 * phi_pk))
```

```text
yoke flux balanced: True
```

## An integrated coupled inductor

Two windings on the outer legs of an EE core, and the air gap in the
centre leg. Same-facing magnetic ports on parallel legs drive flux in
opposite senses around the window, so the coupling is inverse
(`k < 0`): the arrangement an interleaved buck wants. From geometry,
`L = N²·P_o·(P_o + P_g)/(2·P_o + P_g)` and `k = −P_o/(P_o + P_g)`.

```python
MU0 = 4e-7 * math.pi
N = 10
P_out = MU0 * 2000 * 50e-6 / 40e-3        # outer leg: mu_r 2000, 50 mm^2, 40 mm
P_gap = MU0 * 100e-6 / 0.2e-3             # centre gap: 100 mm^2, 0.2 mm

def ee_core():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "e1", "gnd", 1.0)
    b.add_winding("W1", "e1", "gnd", "x1", "gnd", N)
    b.add_winding("W2", "e2", "gnd", "x2", "gnd", N)
    b.add_resistor("Ropen", "e2", "gnd", 1e9)   # winding 2 open
    b.add_permeance("Leg1", "x1", "T", P_out)
    b.add_permeance("Leg2", "x2", "T", P_out)
    b.add_air_gap("Gap", "T", "gnd", 100e-6, 0.2e-3)
    return b

ee = p.simulate(ee_core(), t_end=10e-6, dt=10e-9, engine="pwl")
i1 = ee.i("W1", port=1)                   # winding current
L = 1.0 / ((i1[-1] - i1[-101]) / (ee.times[-1] - ee.times[-101]))
k = ee.v("e2")[-1] / ee.v("e1")[-1]
L_cf = N**2 * P_out * (P_out + P_gap) / (2 * P_out + P_gap)
print(f"L = {L * 1e6:.2f} uH (closed form {L_cf * 1e6:.2f}), "
      f"k = {k:.4f} (closed form {-P_out / (P_out + P_gap):.4f})")
```

```text
L = 171.36 uH (closed form 171.36), k = -0.8333 (closed form -0.8333)
```

Reverse one winding's magnetic port (`"gnd", "x2"`) and `k` becomes
positive.

## Core loss in the loop

`add_magnetic_resistance(name, a, b, R_m)` between two magnetic nodes
dissipates `R_m·(dΦ/dt)²`: classical eddy loss, inside the solve, so it
damps the circuit. It appears in `p.device_loss_summary` with
`category="core_loss"`. Its value is fitted, not predicted: Pulsim's
core-loss model is not validated against measurement. Hysteresis and
excess loss in the network form are not available yet; use
`add_hysteretic_core_inductor` for one flux path, or a Steinmetz / iGSE
estimate after the run (see the losses page).

## Limits

* A saturable section on a node with no winding makes the DC
  operating point fail by name; start from zero, or from an
  `initial_state` as above.
* A saturable section runs on `pwl` and `trbdf2`; `dsed` and `bdf1`
  refuse it by name, like any nonlinear capacitor.
