"""The floating-bus bridge against its exact reference.

Truth: benchmarks/tools/bridge_ground_truth.py (mpmath, 50 digits;
benchmarks/BRIDGE_GROUND_TRUTH.md): the 30-40 ms time-weighted average of
dp - dn is 311.343881428885841697 V and the mode sequence is unique.

trbdf2 follows the exact 11-event sequence; its error is the integration
error at the default step ceiling (span/1000 = 4e-5 s), O(h^2): -2.47e-3 V
at rtol 2e-5 (7.9e-6 of the bus, inside rtol*|v| = 6.2e-3 V per step),
-1.8e-4 V with a 1e-5 s ceiling, -1.91e-4 V at rtol 1e-7. The 1e-15
rounding fuzz moves the average by <= 9e-6 V although the diode-event count
varies (leakage-scale decisions at the zero-current turn-offs). An average
over the samples t >= 30 ms divided by 10 ms, without clipping the window
at 30 ms, read 311.154 V (a 6 us slice of the window lost): that was the
"-0.19 V" of DECISIONS #19, a metric error.
"""
import numpy as np
import pytest

import pulsim as p

from .test_trbdf2_rounding_fuzz import FUZZ

TRUTH = 311.343881428885841697


def _bridge(d, form):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "s", "gnd", 0.0, 325.0 * (1 + d[0]), 50.0)
    b.add_resistor("Rs", "s", "a", 0.5 * (1 + d[1]))
    b.add_inductor("Ls", "a", "ac1", 1e-3 * (1 + d[2]))
    for nm, a, c in (("D1", "ac1", "dp"), ("D2", "gnd", "dp"),
                     ("D3", "dn", "ac1"), ("D4", "dn", "gnd")):
        if form == "branch":
            b.add_diode(nm, a, c, form="branch", R_on=1e-3, g_off=1e-9)
        else:
            b.add_diode(nm, a, c, 1e3, 1e-9)
    b.add_capacitor("C", "dp", "dn", 470e-6 * (1 + d[3]))
    b.add_resistor("RL", "dp", "dn", 100.0 * (1 + d[4]))
    b.add_resistor("Rtie", "dn", "gnd", 1e6 * (1 + d[5]))
    return b


def _window_avg(r, a=30e-3, b=40e-3):
    """Time-weighted mean over exactly [a, b] (end points interpolated)."""
    t = np.asarray(r.times)
    v = np.asarray(r.v("dp")) - np.asarray(r.v("dn"))
    m = (t > a) & (t < b)
    tt = np.concatenate(([a], t[m], [b]))
    vv = np.concatenate(([np.interp(a, t, v)], v[m], [np.interp(b, t, v)]))
    return float(np.trapezoid(vv, tt)) / (b - a)


@pytest.mark.parametrize("form", ["branch", "conductance"])
@pytest.mark.parametrize("rtol,bound", [(1e-7, 1e-3), (2e-5, 3e-3)])
def test_bus_average_is_the_exact_one_under_rounding_fuzz(form, rtol, bound):
    rng = np.random.default_rng(20261117)
    errs = []
    for k in range(20):
        d = np.zeros(6) if k == 0 else rng.uniform(-1, 1, 6) * FUZZ
        r = p.simulate(_bridge(d, form), t_end=40e-3, engine="trbdf2",
                       rtol=rtol, atol=rtol * 1e-3)
        errs.append(_window_avg(r) - TRUTH)
    errs = np.asarray(errs)
    assert np.abs(errs).max() <= bound, errs
    # rounding moves the answer far less than the engine's own error
    assert np.ptp(errs) <= 1e-4, errs
