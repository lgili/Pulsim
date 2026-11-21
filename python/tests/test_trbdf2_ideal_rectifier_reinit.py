"""An ideal (branch-form, R_on = 0) half-wave rectifier on `trbdf2`: the turn-on enters a
dependent mode (the capacitor tied to the source), whose SDIRK re-initialisation step
locates no crossing. It took a full 1 us step past the 25.01 us turn-off, ended at -7 A,
and turned off there: V(out) 0.066 V low for the rest of the run at every rtol (the ideal
twin of `diode_rectifier` on the exact scoreboard, 0.0044 at rtol 1e-6, 1e-8 and 1e-10).

Oracle: the closed form. ON, V(out) = Vin = A sin(wt); the diode current C Vin' + Vin/R
falls to zero at tan(wt) = -wRC; OFF, V(out) decays with tau = RC until Vin meets it
again, g_off = 1e-9 S included: C v' = -v/R + g_off (Vin - v), a linear ODE in closed form.
"""
import math
import warnings

import numpy as np
import pytest

import pulsim as p

A, F, R, C, G_OFF = 10.0, 50e3, 100.0, 10e-6, 1e-9
W, TAU = 2 * math.pi * F, R * C
AL, BE = (1 / R + G_OFF) / C, G_OFF * A / C          # OFF: v' = -AL v + BE sin(wt)
P, Q = BE * AL / (AL**2 + W**2), -BE * W / (AL**2 + W**2)
T_END = 2e-3   # the defect needs the netlist's span: its 1 us reinit step


def _v_off(t, t0, v0):
    part = lambda u: P * math.sin(W * u) + Q * math.cos(W * u)  # noqa: E731
    return (v0 - part(t0)) * math.exp(-AL * (t - t0)) + part(t)


def _oracle(ts):
    """V(out) at the times ts, ON from t = 0 (uic, Vin(0) = 0 = V(out))."""
    phi_off = math.pi - math.atan(W * TAU)          # ON -> OFF phase (each period)
    out = np.empty_like(ts)
    t_on, period = 0.0, 1.0 / F
    k = 0
    segs = []
    while t_on < T_END:
        t_off = (phi_off + 2 * math.pi * k) / W
        v_off = A * math.sin(W * t_off)
        # next turn-on: A sin(wt) = v_OFF(t), on the rising slope
        f = lambda t, t_off=t_off, v_off=v_off: A * math.sin(W * t) - _v_off(t, t_off, v_off)  # noqa: E731
        lo, hi = t_off + 0.5 * period, t_off + period
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
        segs.append((t_on, t_off, v_off))
        if t_off > T_END:
            break
        t_on = 0.5 * (lo + hi)
        k += 1
    for i, t in enumerate(ts):
        for (a, b, v0), nxt in zip(segs, segs[1:] + [(math.inf, 0, 0)]):
            if a <= t <= b:
                out[i] = A * math.sin(W * t)
                break
            if b < t < nxt[0]:
                out[i] = _v_off(t, b, v0)
                break
    return out


def _run(rtol):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "in", "gnd", 0.0, A, F)
    b.add_diode("D1", "in", "out", None, 1e-9, 0.0, 0.0, form="branch", R_on=0.0)
    b.add_resistor("Rload", "out", "gnd", R)
    b.add_capacitor("Cfilter", "out", "gnd", C)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, T_END, engine="trbdf2", rtol=rtol, atol=1e-12)
    t = np.asarray(r.times)
    v = np.asarray(r.v("out"))
    keep = t > 0
    return float(np.max(np.abs(v[keep] - _oracle(t[keep]))))


@pytest.mark.parametrize("rtol", [1e-5, 1e-6, 1e-8])
def test_reinit_step_does_not_overrun_the_turn_off(rtol):
    """Was 0.0658 V at every rtol (a wrong decision is flat in rtol). Measured after the
    fix, max |V(out) - oracle| over all samples: 9.7e-8 / 1.6e-7 / 2.6e-7 V. The bound is
    the event location's, not the run's accuracy: a turn-on landed within its 1e-4 h
    bracket (h <= 1 us) can be 1e-10 s late, and V(out)'s slope jumps by 0.6 V/us there,
    6e-5 V on that one sample (a 1e-15 parameter draw showed 2.2e-5 V). 5e-4 V is 8x that
    and 130x below the defect."""
    assert _run(rtol) < 5e-4
