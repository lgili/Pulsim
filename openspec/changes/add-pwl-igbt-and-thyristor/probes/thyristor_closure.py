"""Half-wave phase-controlled rectifier, R load, with the Python thyristor
closure (pulsim.make_thyristor_switch_fn). Read-only probe.

Analytic mean load voltage: Vm/(2*pi)*(1+cos(alpha)).
Control: the device conducts forward after the gate (v_load max = Vm).

Measured 2026-11-12 on docs/core-roadmap-openspec, dt = 10 us:
  50 Hz (zero crossing on the grid), i_holding = 0   : 29.700 V (analytic 29.699), no reverse conduction
  60 Hz (zero crossing off the grid), i_holding = 0  : -2.140 V (analytic 29.699), v_load min = -99.99 V
  60 Hz, i_holding = 0.05 A                          : 29.687 V
"""
import math
import warnings

import numpy as np

import pulsim as p

warnings.simplefilter("ignore")
VM, R, DT = 100.0, 10.0, 10e-6


def run(f, alpha_deg, i_hold):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, VM, f)
    b.add_switch("T", "a", "k", 1e3, 1e-9)
    b.add_resistor("R", "k", "gnd", R)
    T = 1.0 / f
    ta = alpha_deg / 360.0 * T
    sf, ob = p.make_thyristor_switch_fn(
        num_switches=b.graph.num_switches, switch_idx=0,
        gate_fn=lambda t: ta <= (t % T) < ta + 1e-3,
        current_probe=lambda t, x: float(x[1]) / R,   # x[1] is node "k" here
        i_holding=i_hold)
    r = p.simulate(b, t_end=3 * T, dt=DT, engine="pwl", switch_fn=sf, step_observer=ob)
    t = np.asarray(r.times)
    v = np.asarray(r.v("k"))
    m = t >= 2 * T
    mean = np.trapezoid(v[m], t[m]) / T
    ana = VM / (2 * math.pi) * (1 + math.cos(math.radians(alpha_deg)))
    return mean, ana, v[m].max(), v[m].min()


for f in (50.0, 60.0):
    for ih in (0.0, 0.05):
        for a in (30.0, 90.0):
            mean, ana, vmax, vmin = run(f, a, ih)
            print(f"f={f:.0f} Hz i_holding={ih}: alpha={a:.0f} deg mean={mean:.3f} V "
                  f"analytic={ana:.3f} V  v_load max={vmax:.2f} min={vmin:.2f}")
