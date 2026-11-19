"""Order of accuracy across a gate-driven threshold crossing.

The power MOSFET's gate is driven by a SMOOTH source (6 − 6·cos ωt,
f = 2 MHz) through R_g into C_gs, C_gd; the drain is a 400 V supply
behind a 40 Ω load. The channel turns on through its own charge
dynamics: no source edge, no switch event. Reference: scipy Radau at
rtol 1e-12 on the same equations (two node ODEs, the Level-1 law of
models/mosfet_level1.hpp and the vdmos_cgd law of
models/nonlinear_capacitor.hpp written out in numpy).

    PYTHONPATH=python python3 probes/threshold_order.py [pwl|trbdf2|both] [linear|vdmos]

`vdmos` makes C_gd the VDMOS law (CGDMAX 1 nF, CGDMIN 50 pF, A 0.5), whose
branch point v_gd = 0 is crossed near the end of the drain fall.

Measured 2026-11-18 (both laws): second order on both engines; the
gate-threshold crossing itself (V_GS = V_T at ~93 ns) is second order at
every dt tried. The largest error sits at the saturation→triode edge
(V_DS = V_GS − V_T, ~215 ns), where the β blend of width 1/κ = 67 mV is
crossed in ~13 ps: with the linear C_gd pwl is pre-asymptotic until dt
resolves it (order 0.28, 1.28 from 125 to 31.25 ps, then 2.00, 2.00 down
to 7.8 ps); with the VDMOS C_gd (slower edge) 2.00 from 250 ps. trbdf2:
error against step count 2.0–2.2 (linear) and 1.6–2.2 (VDMOS) per row.
"""
import math
import sys

import numpy as np
from scipy.integrate import solve_ivp

import pulsim as ps

K, VT, LAM, KAP = 1.43, 3.0, 0.0, 15.0
CGS, CGD, RG, RL, VDD = 1e-9, 100e-12, 10.0, 40.0, 400.0
F, T_END = 2e6, 250e-9


def logistic(z):
    return 0.5 * (1.0 + math.tanh(0.5 * z))


def fwd(vov, vds):
    a = logistic(KAP * vov)
    b = logistic(KAP * (vov - vds))
    clm = 1.0 + LAM * vds
    return a * (b * K * (2 * vov * vds - vds * vds) * clm + (1 - b) * K * vov * vov * clm)


def ich(vg, vd):
    vov = vg - VT
    return -fwd(vov - vd, -vd) if vd < 0 else fwd(vov, vd)


def vdrv(t):
    return 6.0 + 6.0 * math.sin(2 * math.pi * F * t - math.pi / 2)


LAW = sys.argv[2] if len(sys.argv) > 2 else "linear"
CGD_MAX, CGD_MIN, A_GD = 1e-9, 50e-12, 0.5


def cgd(u):
    if LAW == "linear":
        return CGD
    s = (CGD_MAX - CGD_MIN) / (1.0 + math.pi / 2)
    y = CGD_MAX - s
    return y + s * (math.tanh(A_GD * u) if u > 0 else math.atan(A_GD * u))


def rhs(t, y):
    vg, vd = y
    c = cgd(vg - vd)          # charge form: d/dt Q_gd(v_g - v_d) = C_gd(u)·(v̇_g - v̇_d)
    m = np.array([[CGS + c, -c], [-c, c]])
    return np.linalg.solve(m, [(vdrv(t) - vg) / RG, (VDD - vd) / RL - ich(vg, vd)])


def reference():
    """Dense Radau solution: the error is measured at the ENGINE'S OWN
    samples (a linear interpolation of an engine's output between its
    samples would add an O(h²) error of its own)."""
    s = solve_ivp(rhs, (0, T_END), [0.0, VDD], method="Radau", rtol=1e-12, atol=1e-12,
                  dense_output=True)
    return s.sol


def build():
    b = ps.CircuitBuilder()
    b.add_voltage_source("Vdd", "vdd", "gnd", VDD)
    b.add_resistor("RL", "vdd", "d", RL)
    b.add_sine_voltage_source("Vg", "drv", "gnd", 6.0, 6.0, F, -math.pi / 2)
    c_gd = CGD if LAW == "linear" else ps.ChargeLaw.vdmos_cgd(CGD_MAX, CGD_MIN, A_GD)
    b.add_power_mosfet("M", "d", "gnd", "drv", K=K, V_T=VT, lambda_=LAM, kappa=KAP,
                       C_gs=CGS, C_gd=c_gd, C_ds=0.0, R_g=RG)
    names = b.state_var_names()
    x0 = np.array(b.initial_state(), dtype=float).ravel()
    for n in ("V(d)", "V(vdd)"):
        x0[names.index(n)] = VDD
    return b, x0


def run(**kw):
    b, x0 = build()
    r = ps.simulate(b, t_end=T_END, initial_state=x0, **kw)
    t = np.asarray(r.times)
    return t, np.column_stack([np.asarray(r.v(n)) for n in ("M.g", "d")])


def errors(ref, **kw):
    """Max |error| of (v_g, v_d) over the engine's samples, and its step count."""
    t, y = run(**kw)
    return np.max(np.abs(y - ref(t).T), axis=0), t.size - 1


def table(ref, label, cases):
    """Order = log(error ratio) / log(step-count ratio) between rows."""
    print(f"\n{label}")
    prev = None
    for name, kw in cases:
        e, n = errors(ref, **kw)
        order = "" if prev is None else f"  order {math.log(prev[0][1] / e[1]) / math.log(n / prev[1]):5.2f}"
        print(f"  {name:>14}: steps {n:6d}  |e v_g| {e[0]:.3e} V  |e v_d| {e[1]:.3e} V{order}")
        prev = (e, n)


if __name__ == "__main__":
    ref = reference()
    eng = sys.argv[1] if len(sys.argv) > 1 else "both"
    if eng in ("both", "pwl"):
        table(ref, "pwl (fixed step)",
              [(f"dt {dt*1e12:g} ps", dict(engine="pwl", dt=dt))
               for dt in (2.5e-10, 1.25e-10, 6.25e-11, 3.125e-11, 1.5625e-11, 7.8125e-12)])
    if eng in ("both", "trbdf2"):
        # rtol alone moves the step only below the default ceiling t_end/1000
        # (0.25 ns); dt (the ceiling on trbdf2) lifts it so the controller sets every step.
        table(ref, "trbdf2 (rtol per row; order against the step count)",
              [(f"rtol {rt:g}", dict(engine="trbdf2", rtol=rt, atol=rt * 1e-3, dt=T_END))
               for rt in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)])
