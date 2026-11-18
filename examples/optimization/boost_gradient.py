"""Sizing a boost's L and C with exact gradients.

12 V -> 24 V (D = 0.5, open loop), 100 kHz, 20 ohm load (1.2 A). The
design variables are the inductance and the output capacitance; the
targets are an output ripple of 50 mV and an inductor ripple of 0.6 A
(25 % of the 2.4 A input current), at the smallest L and C:

    cost(L, C) = (ripple_v / 50 mV - 1)^2 + (ripple_i / 0.6 A - 1)^2
                 + 1e-3 (L / L0 + C / C0)

Every evaluation is the periodic steady state by Newton shooting on the
exact stepper; ``Evaluator(gradients=...)`` returns d(KPI)/d(L, C) from
the orbit sensitivities (no finite difference), and
``pulsim.value_and_grad`` hands ``(cost, gradient)`` to
``scipy.optimize.minimize(jac=True)``. Without scipy the example falls
back to a grid on the kernel batch evaluator (one GIL-free call for the
whole grid, DECISIONS 26) and takes the best design of the grid.

READ THIS BEFORE QUOTING A NUMBER FROM IT
-----------------------------------------
* The switch and the diode are ideal PWL devices (1 mOhm / 1 GOhm); the
  inductor and capacitor are ideal but for a 10 mOhm DCR. Nothing here
  is a real part; the loss numbers are the circuit's own resistive
  losses, not a semiconductor model.
* What IS checked: ``python/tests/test_example_boost_gradient.py``. The
  gradient at the start equals a central difference within 1e-6; the
  optimiser lowers the cost and lands both ripples within 5 % of their
  targets; the grid fallback's records equal the Python path's and run
  in the kernel. Every other printed number is an OBSERVATION of this
  model on the machine that ran it.

    python examples/optimization/boost_gradient.py [--grid]
"""

from __future__ import annotations

import sys

import numpy as np

import pulsim as p
from pulsim.design import Evaluator, Kpi, SteadySpec

VIN, D, F_SW, R_LOAD, DCR = 12.0, 0.5, 100e3, 20.0, 10e-3
T = 1.0 / F_SW
RIPPLE_V, RIPPLE_I = 0.05, 0.6          # targets: V peak-to-peak, A
L0, C0 = 100e-6, 47e-6                   # scales of the variables
BOUNDS = ((0.2, 5.0), (0.2, 10.0))       # L / L0, C / C0
VARS = {"L": "L.L", "C": "C.C"}


def circuit():
    """The boost as one immutable circuit; a design is ``with_value`` of
    its ``L.L`` and ``C.C``."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", VIN)
    b.add_inductor("L", "vin", "x", L0)
    b.add_resistor("R_dcr", "x", "sw", DCR)
    b.add_switch("S", "sw", "gnd", 1e3, 1e-9)
    b.add_diode("D1", "sw", "vout", 1e3, 1e-9)
    b.add_capacitor("C", "vout", "gnd", C0)
    b.add_resistor("R_load", "vout", "gnd", R_LOAD)
    return b.build()


def schedule(design):
    return p.make_pwm_switch_fn(F_SW, D, 0, 2)


def make_evaluator(gradients=True, base=None):
    """``gradients=True``: d(KPI)/d(L, C) on every record (the Python
    path); ``False``: declarative only, so ``evaluate_many`` runs the
    kernel batch."""
    return Evaluator(
        circuit=base if base is not None else circuit(), fields=VARS,
        schedule=schedule, period=T, loads=("R_load",), output="vout",
        kpis=[Kpi.formula("cost", cost_formula())],
        steady=SteadySpec(method="pss"),
        gradients=dict(VARS) if gradients else None)


def cost_formula():
    return (f"(ripple_v / {RIPPLE_V!r} - 1) ** 2 + "
            f"(ripple_i.L / {RIPPLE_I!r} - 1) ** 2 + "
            f"1e-3 * (L / {L0!r} + C / {C0!r})")


def objective(ev):
    """``values -> (cost, {"L": dcost/dL, "C": dcost/dC})`` from one
    evaluation and its KPI gradients."""
    def f(values):
        r = ev.evaluate(values)
        if r.status not in ("ok", "infeasible"):
            raise RuntimeError(f"{values}: {r.status}: {r.reason}")
        k, g = r.kpis, r.grads
        uv = k["ripple_v"] / RIPPLE_V - 1.0
        ui = k["ripple_i.L"] / RIPPLE_I - 1.0
        grad = {x: 2 * uv * g["ripple_v"][x] / RIPPLE_V
                + 2 * ui * g["ripple_i.L"][x] / RIPPLE_I
                + 1e-3 / (L0 if x == "L" else C0) for x in VARS}
        return k["cost"], grad
    return f


def optimise(ev, x0=(1.0, 1.0)):
    """``scipy.optimize.minimize(jac=True)`` on the scaled variables;
    returns ``(result, evaluations)``."""
    from scipy.optimize import minimize
    vg = p.value_and_grad(objective(ev), list(VARS))
    scale = np.array([L0, C0])
    n = [0]

    def f(u):
        n[0] += 1
        v, g = vg(u * scale)
        return v, g * scale
    res = minimize(f, np.asarray(x0, float), jac=True, method="L-BFGS-B",
                   bounds=BOUNDS)
    return res, n[0]


def grid(n=8, workers=4):
    """The fallback: an ``n x n`` log grid on the kernel batch."""
    ev = make_evaluator(gradients=False)
    ls = L0 * np.geomspace(*BOUNDS[0], n)
    cs = C0 * np.geomspace(*BOUNDS[1], n)
    designs = [{"L": float(a), "C": float(c)} for a in ls for c in cs]
    recs = ev.evaluate_many(designs, workers=workers)
    ok = [r for r in recs if r.status == "ok"]
    return min(ok, key=lambda r: r.kpis["cost"]), recs, ev


def main(argv):
    try:
        import scipy.optimize  # noqa: F401
        have_scipy = "--grid" not in argv
    except ImportError:
        have_scipy = False
    if have_scipy:
        ev = make_evaluator()
        start = ev.evaluate({"L": L0, "C": C0})
        print(f"start  L = {L0 * 1e6:.1f} uH, C = {C0 * 1e6:.1f} uF: "
              f"ripple_v {start.kpis['ripple_v'] * 1e3:.2f} mV, ripple_i "
              f"{start.kpis['ripple_i.L']:.3f} A, cost "
              f"{start.kpis['cost']:.4f}  (observation)")
        res, n = optimise(ev)
        best = ev.evaluate({"L": res.x[0] * L0, "C": res.x[1] * C0})
        print(f"optimum L = {res.x[0] * L0 * 1e6:.2f} uH, C = "
              f"{res.x[1] * C0 * 1e6:.2f} uF after {n} evaluations: "
              f"ripple_v {best.kpis['ripple_v'] * 1e3:.2f} mV, ripple_i "
              f"{best.kpis['ripple_i.L']:.3f} A, efficiency "
              f"{best.kpis['efficiency']:.5f}  (observation; tested: both "
              "ripples within 5 % of target)")
    else:
        best, recs, ev = grid()
        print(f"grid of {len(recs)} designs on the kernel batch "
              f"({ev.last_batch['kernel']} in the kernel): best L = "
              f"{best.design['L'] * 1e6:.2f} uH, C = "
              f"{best.design['C'] * 1e6:.2f} uF, ripple_v "
              f"{best.kpis['ripple_v'] * 1e3:.2f} mV, ripple_i "
              f"{best.kpis['ripple_i.L']:.3f} A  (observation)")


if __name__ == "__main__":
    main(sys.argv[1:])
