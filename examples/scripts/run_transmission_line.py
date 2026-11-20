#!/usr/bin/env python3
"""Bergeron transmission line: the travelling-wave lattice and a quarter-wave transformer.

A lossless line, Z0 = 50 ohm, one-way delay tau = 100 ns
(``add_transmission_line(..., model="bergeron")``).

1. Lattice. A 1 V step behind 10 ohm into the line, far end open.
   V+ = 50/60 = 5/6 V, Gamma_s = (10 - 50)/60 = -2/3, Gamma_L = +1, so the
   far end steps after (2n+1) tau to 2 V+ * sum_{j<=n} Gamma_s^j:
   5/3, 5/9, 35/27 V. With tau a multiple of dt (here 100 dt) the fixed
   engine reads the stored sample, so the plateaus are exact to rounding.

2. Quarter wave. At f = 1/(4 tau) the line turns a load R_L into
   Z0^2 / R_L at its input: 10 ohm -> 250 ohm. Driven by a 1 mA sine, the
   near-end amplitude settles to 0.25 V once the reflections have died out
   (each round trip multiplies them by Gamma_L = -2/3).
"""

from __future__ import annotations

import numpy as np

import pulsim as p

Z0, TAU = 50.0, 100e-9
T_EDGE = 10e-9


def lattice() -> None:
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "s", "gnd", 0.0, 1.0, T_EDGE, 1.0, 0.0,
                               1e-12, 1e-12)
    b.add_resistor("Rs", "s", "k", 10.0)
    b.add_transmission_line("T", "k", "gnd", "m", "gnd", Z0=Z0, delay=TAU,
                            model="bergeron")
    r = p.simulate(b, t_end=7 * TAU, dt=1e-9, engine="pwl")
    t, vm = np.asarray(r.times), np.asarray(r.v("m"))
    for n, exact in enumerate((5 / 3, 5 / 9, 35 / 27)):
        tq = T_EDGE + (2 * n + 1) * TAU + TAU / 2   # mid plateau
        got = float(np.interp(tq, t, vm))
        print(f"lattice plateau {n}: {got:.12f} V (exact {exact:.12f}), "
              f"relative error {abs(got / exact - 1):.1e}")


def quarter_wave() -> None:
    f_q = 1 / (4 * TAU)
    b = p.CircuitBuilder()
    b.add_sine_current_source("I", "gnd", "k", 0.0, 1e-3, f_q)
    b.add_resistor("Rp", "k", "gnd", 1e6)   # a DC path for the source
    b.add_transmission_line("T", "k", "gnd", "m", "gnd", Z0=Z0, delay=TAU,
                            model="bergeron")
    b.add_resistor("RL", "m", "gnd", 10.0)
    n_per = 400
    r = p.simulate(b, t_end=60 * 4 * TAU, dt=4 * TAU / n_per, engine="pwl")
    vk = np.asarray(r.v("k"))[-n_per:]   # the last period
    amp = 0.5 * (vk.max() - vk.min())
    exact = (Z0 ** 2 / 10.0) * 1e6 / (Z0 ** 2 / 10.0 + 1e6) * 1e-3
    print(f"quarter-wave input amplitude: {amp:.6f} V "
          f"(Z0^2/R_L || Rp: {exact:.6f} V)")


if __name__ == "__main__":
    lattice()
    quarter_wave()
