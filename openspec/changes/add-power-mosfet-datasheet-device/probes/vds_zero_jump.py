"""The Level-1 channel law is discontinuous at V_DS = 0 near threshold.

forward(V_OV, 0) = logistic(κV_OV)·(1 − logistic(κV_OV))·K·V_OV² is not 0
(the saturation branch leaks through the β blend at V_DS = 0), and the
third quadrant is −forward(V_OV − V_DS, −V_DS), so i jumps by 2·forward(V_OV, 0)
across V_DS = 0 (removed 2026-11-18 by the third-quadrant term). Prints the jump against V_GS (K = 1.43, V_T = 3, κ = 15),
then drives a ±10 mA sine current through the channel with the gate held
at V_GS (1 nF across it) and reports what each engine does.

    PYTHONPATH=python python3 probes/vds_zero_jump.py
"""
import math
import warnings

import numpy as np

import pulsim as ps

K, VT, KAP = 1.43, 3.0, 15.0


def lg(z):
    return 0.5 * (1.0 + math.tanh(0.5 * z))


def fwd(vov, vds):
    a, b = lg(KAP * vov), lg(KAP * (vov - vds))
    return a * (b * K * (2 * vov * vds - vds * vds) + (1 - b) * K * vov * vov)


def law(vgs, vds, fixed=True):
    """`fixed`: the 2026-11-18 third-quadrant term 2·forward(V_OV − V_DS/2, 0)."""
    vov = vgs - VT
    if vds >= 0:
        return fwd(vov, vds)
    return -fwd(vov - vds, -vds) + (2 * fwd(vov - 0.5 * vds, 0.0) if fixed else 0.0)


def run(vgs, engine):
    b = ps.CircuitBuilder()
    b.add_voltage_source("Vg", "g", "gnd", vgs)
    b.add_sine_current_source("I", "gnd", "d", 0.0, 10e-3, 1e3)
    b.add_capacitor("C", "d", "gnd", 1e-9)
    b.add_mosfet_level1("M", "d", "gnd", "g", K=K, V_T=VT, lambda_=0.0, kappa=KAP,
                        with_body_diode=False)
    kw = dict(dt=1e-6) if engine == "pwl" else {}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            r = ps.simulate(b, t_end=2e-3, engine=engine, **kw)
        except Exception as e:  # noqa: BLE001
            return f"raised {type(e).__name__}: {str(e)[:90]}"
    v = np.asarray(r.v("d"))
    return (f"{len(r.times)} samples, v_d in [{v.min():.3e}, {v.max():.3e}] V, "
            f"{len(w)} warnings {[str(x.message)[:60] for x in w]}")


if __name__ == "__main__":
    for vgs in (2.5, 3.2, 3.5, 4.0, 5.0, 6.0):
        old = law(vgs, 1e-15) - law(vgs, -1e-15, fixed=False)
        new = law(vgs, 1e-15) - law(vgs, -1e-15)
        print(f"V_GS {vgs}: jump i(0+) - i(0-) = {old:.3e} A before, {new:.3e} A after")
    for vgs in (3.2, 4.0):
        for eng in ("pwl", "trbdf2"):
            print(f"V_GS {vgs} {eng}: {run(vgs, eng)}")
