"""Resonant-mode gates of the restart rule (the review that replaced the BE pair).

    python resonant_gates.py [--on]

A. Series-resonant tank (100 uH, 100 nF, Q = 50) driven at resonance by a
   +-50 V PWM source: peak inductor current over periods 380-400 against the
   converged value (T/1600 with the flag off).
B. Ideal LC (1 mH, 1 uF, i0 = 1 A) beside an ELECTRICALLY ISOLATED PWM source
   into a resistor: energy after 100 periods / energy at t = 0 (exact: 1).
"""
import math
import sys

import numpy as np

import pulsim as p

L, C = 100e-6, 100e-9
F0 = 1 / (2 * math.pi * math.sqrt(L * C))
T = 1 / F0
R = math.sqrt(L / C) / 50


def tank_peak(N, **kw):
    b = p.CircuitBuilder()
    b.add_pwm_voltage_source("V", "a", "gnd", 50.0, -50.0, F0, 0.5)
    b.add_resistor("R", "a", "m", R)
    b.add_inductor("L", "m", "n", L)
    b.add_capacitor("C", "n", "gnd", C)
    r = p.simulate(b, t_end=400 * T, dt=T / N, engine="pwl", **kw)
    t, i = np.asarray(r.times), np.asarray(r.i("L"))
    return float(np.abs(i[t > 380 * T]).max()), r


def lc_energy(**kw):
    w = 1 / math.sqrt(1e-3 * 1e-6)
    Tp = 2 * math.pi / w
    b = p.CircuitBuilder()
    b.add_inductor("L", "a", "gnd", 1e-3, i0=1.0)
    b.add_capacitor("C", "a", "gnd", 1e-6)
    b.add_pwm_voltage_source("Vp", "x", "gnd", 10.0, 0.0, 2.5 / Tp, 0.5)
    b.add_resistor("Rx", "x", "gnd", 10.0)
    r = p.simulate(b, t_end=100 * Tp, dt=Tp / 50, engine="pwl", **kw)
    i, v = np.asarray(r.i("L")), np.asarray(r.v("a"))
    E = 0.5e-3 * i ** 2 + 0.5e-6 * v ** 2
    return float(E[-1] / E[0]), r


if __name__ == "__main__":
    kw = dict(discontinuity_restart=True) if "--on" in sys.argv else {}
    ref = tank_peak(1600)[0]
    print(f"A. tank, converged peak (T/1600, flag off) {ref:.4f} A")
    for N in (50, 100, 200):
        pk, r = tank_peak(N, **kw)
        print(f"   T/{N}: peak {pk:.4f} A, error {100 * (pk / ref - 1):+.3f} %, restarts {r.n_restart_steps}")
    e, r = lc_energy(**kw)
    print(f"B. isolated-PWM LC: E(100 T)/E(0) = {e:.4f}, restarts {r.n_restart_steps}")
