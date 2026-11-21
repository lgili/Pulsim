"""Step response of a lumped pi-section line built from existing L and C primitives.

Read-only probe. Z0 = 50 ohm, delay tau = 100 ns, matched source, open far end.
An ideal lossless line gives exactly 1.000 V at the far end after tau, with no
overshoot.

Measured 2026-11-12 on docs/core-roadmap-openspec (pwl, dt = 25 ps):
  ideal step:      N=10  t50/tau=1.0438 peak=1.2635 | N=20 1.0278 1.2682
                   N=50  t50/tau=1.0152 peak=1.2713 | N=100 1.0096 1.2726
  20 ns rise time: N=20  t50/tau=1.0176 peak=1.1022 | N=50 0.9994 1.0540
  trbdf2, N=50, ideal step: t50/tau=1.0151 peak=1.2715 in 2341 points
"""
import warnings

import numpy as np

import pulsim as p

warnings.simplefilter("ignore")
Z0, TAU = 50.0, 100e-9


def run(n_sec, engine="pwl", dt=25e-12, trise=0.0):
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "s", "gnd", 0.0, 1.0, 10e-9, 1.0, 0.0,
                               max(trise, 1e-12), 1e-12)
    b.add_resistor("Rs", "s", "n0", Z0)
    l_sec, c_sec = Z0 * TAU / n_sec, TAU / (Z0 * n_sec)
    for k in range(n_sec):
        b.add_capacitor(f"Ca{k}", f"n{k}", "gnd", c_sec / 2)
        b.add_inductor(f"L{k}", f"n{k}", f"n{k + 1}", l_sec)
        b.add_capacitor(f"Cb{k}", f"n{k + 1}", "gnd", c_sec / 2)
    b.add_resistor("Rend", f"n{n_sec}", "gnd", 1e9)
    kw = dict(dt=dt) if engine == "pwl" else {}
    r = p.simulate(b, t_end=4 * TAU, engine=engine, **kw)
    t = np.asarray(r.times)
    v = np.asarray(r.v(f"n{n_sec}"))
    k50 = np.argmax(v >= 0.5)
    t50 = np.interp(0.5, [v[k50 - 1], v[k50]], [t[k50 - 1], t[k50]]) - 10e-9 - trise / 2
    return t50 / TAU, v.max(), v[-1], len(t)


for n_sec in (10, 20, 50, 100):
    print("N=%3d step:    t50/tau=%.4f  peak=%.4f  final=%.4f n=%d" % ((n_sec,) + run(n_sec)))
for n_sec in (20, 50):
    print("N=%3d tr=20ns: t50/tau=%.4f  peak=%.4f  final=%.4f n=%d"
          % ((n_sec,) + run(n_sec, trise=20e-9)))
print("N= 50 trbdf2:", run(50, "trbdf2"))
