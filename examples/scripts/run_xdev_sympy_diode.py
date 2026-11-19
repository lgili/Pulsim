#!/usr/bin/env python3
"""A diode declared as equations (SymPy), compiled to C, run as an external device.

   V (20 V, 1 kHz) ── Rs 1 Ohm ── D ──┬── out
                                     C 10 uF ‖ R 100 Ohm
                                     gnd

``pulsim.xdev`` differentiates the law, writes plain C99 against the stable
header, compiles it through the per-user native cache and loads it with
``CircuitBuilder.add_external_device`` (whose registration checks run on the
generated model). The law is the built-in Shockley diode's, written out:
the exponential continued by its own tangent above v_lim = n V_T ln(i_lim /
I_S + 1) (a Piecewise the smoothness lint proves C1), plus G_min.

Expected: the generated model gives the built-in diode's node voltages
(identical law; the C uses expm1 where the built-in writes exp - 1, so the
two agree to rounding, not to the bit — below 1e-9 of the voltage scale on
any platform, 1.6e-14 on macOS arm64). The monotonicity scan finds the diode law
non-decreasing over [-1, 1] V down to 1e-15 V of 0, and finds the textbook
Langevin law coth(u) - 1/u (written without its series) DECREASING near
u = 0, where its two 1/u-sized terms cancel: the defect that once made a
magnetic law non-monotone at 1 pA. The registration checks sample a box and
cannot see it; the scan can.

Needs SymPy (``pip install pulsim[codegen]``) and a C compiler. The
parameters are the built-in's defaults, not a datasheet part.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

import pulsim as p
from pulsim import xdev


def declare_shockley():
    a, k = xdev.pins("anode", "cathode")
    I_S, n, V_T, G_min, i_lim = xdev.params(
        I_S=(1e-12, "A", 1e-30, 1.0), n=(1.0, "", 0.1, 10.0), V_T=(0.025852, "V", 1e-3, 1.0),
        G_min=(1e-12, "S", 0.0, 1.0), i_lim=(1e6, "A", 1e-6, 1e12))
    vte = n * V_T
    v_lim = vte * sp.log(i_lim / I_S + 1)
    e_lim = sp.exp(v_lim / vte)
    u = a - k
    i = sp.Piecewise((I_S * (e_lim - 1) + I_S * e_lim / vte * (u - v_lim), u > v_lim),
                     (I_S * (sp.exp(u / vte) - 1), True)) + G_min * u
    return xdev.compile("sympy_diode", pins=[a, k], f=[i, -i])


def rectifier(diode):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 20.0, 1e3)
    b.add_resistor("Rs", "in", "a", 1.0)
    if diode is None:
        b.add_shockley_diode("D", "a", "out", 1e-12, 1.0, 0.025852, 1e-12)
    else:
        diode.add_to(b, "D", "a", "out")
    b.add_capacitor("C", "out", "gnd", 10e-6)
    b.add_resistor("R", "out", "gnd", 100.0)
    return b


def main() -> None:
    dev = declare_shockley()
    print(f"generated model: {dev.name}, SMOOTH claimed: {dev.smooth}, "
          f"Jacobian pattern {dev.f_pattern}")
    kw = dict(t_end=2e-3, dt=1e-6, device_path="interface")
    builtin = p.simulate(rectifier(None), **kw)
    gen = p.simulate(rectifier(dev), **kw)
    xa, xb = np.asarray(builtin.states), np.asarray(gen.states)
    diff = float(np.max(np.abs(xa - xb)) / np.max(np.abs(xa)))
    print(f"output at 2 ms: built-in {builtin.v('out')[-1]:.6f} V, "
          f"generated {gen.v('out')[-1]:.6f} V")
    print(f"max relative difference against the built-in diode: {diff:.3e}")

    scan = xdev.scan_monotone(dev, "anode", -1.0, 1.0)
    print(f"diode law non-monotone intervals: {len(scan.decreasing)}")

    a, k = xdev.pins("a", "k")
    G = xdev.params(G=(1.0, "S", 1e-9, 1e3))
    u = a - k
    lang = G * (sp.coth(u) - 1 / u)
    raw = xdev.scan_monotone(xdev.compile("langevin_raw", [a, k], [lang, -lang]), "a", -1, 1)
    widest = max(max(abs(lo), abs(hi)) for lo, hi in raw.decreasing)
    print(f"raw Langevin non-monotone intervals: {len(raw.decreasing)} "
          f"(all within |u| < {widest:.1e} V; NaN at u = {raw.nonfinite})")


if __name__ == "__main__":
    main()
