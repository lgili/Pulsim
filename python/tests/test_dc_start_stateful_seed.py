"""A run started from the DC operating point starts ON it and stays there.

`simulate(start_from_dc_op=True)` solves the DC point and seeds every
history from it. Two defects (add-high-order-stiff-integrator DECISIONS 17;
add-unified-device-interface DECISIONS #56):

- the run's DC solve did not stamp the Lauritzen diode (an open circuit:
  2.000 V at the anode where the law gives 0.73096 V), on both paths;
- only the linear, PMSM and external histories were seeded: the Lauritzen
  charge, the IGBT tail charge, the nonlinear capacitor's charge and the
  saturable inductor's flux started from zero, so a consistent DC start ran
  a transient away from its own DC point (saturable: the 1 A short
  collapsed, v went 0 → 0.99 V).

Truths: the Lauritzen DC law solved by bisection (i = τ/(τ+T_M)·I_S·
(e^{v/nV_T} − 1) + G_min·v, the charge row at q̇ = 0); for the other kinds
the DC point itself (an inductor is a short, a capacitor is open, the tail
current is 0 at Q = k·i_ss·τ), and a consistent start has no transient.
"""
import math
import warnings

import numpy as np
import pytest

import pulsim as p
from pulsim import _pulsim as k

PATHS = ("legacy", "interface")


def _lauritzen():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 2.0)
    b.add_resistor("R", "in", "a", 1.0)
    b.add_lauritzen_diode("D", "a", "gnd", tau=1e-4, T_M=5e-5)
    return b


def _lauritzen_truth(V=2.0, R=1.0, I_S=1e-12, n=1.0, VT=0.025852, tau=1e-4,
                     T_M=5e-5, G_min=1e-12):
    def res(v):
        return (V - v) / R - (tau / (tau + T_M) * I_S * math.expm1(v / (n * VT))
                              + G_min * v)
    lo, hi = 0.0, V
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if res(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _run(b, path, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, 1e-5, dt=1e-7, engine="pwl", start_from_dc_op=True,
                          device_path=path, **kw)


@pytest.mark.parametrize("path", PATHS)
def test_lauritzen_dc_point_is_the_law(path):
    v_true = _lauritzen_truth()
    assert v_true == pytest.approx(0.73096, abs=5e-6)
    b = _lauritzen()
    dc = k.compute_dc_operating_point(
        b.graph, b.pool, k.SwitchStateMask(b.graph.num_switches),
        device_path=getattr(k.DevicePath, path.capitalize()))
    assert dc.x[b.node_id_of("a")] == pytest.approx(v_true, rel=1e-9)


@pytest.mark.parametrize("path", PATHS)
def test_lauritzen_start_from_dc_op_stays_on_the_law(path):
    v_true = _lauritzen_truth()
    v = np.asarray(_run(_lauritzen(), path).v("a"))
    # before the fix: v[0] = 2.000 V (the open diode), then a recovery
    assert v[0] == pytest.approx(v_true, rel=1e-9)
    assert np.max(np.abs(v - v_true)) < 1e-6


def _saturable():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 1.0)
    b.add_resistor("R", "in", "a", 1.0)
    b.add_saturable_inductor("L", "a", "gnd", 1e-3, 0.5, 1e-4)
    return b


def _nonlinear_cap():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 5.0)
    b.add_resistor("R", "in", "a", 100.0)
    b.add_resistor("R2", "a", "gnd", 100.0)
    b.add_nonlinear_capacitor("C", "a", "gnd", 1e-9, 0.7)
    return b


def _igbt_tail():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_voltage_source("G", "g", "gnd", 15.0)
    b.add_resistor("R", "in", "a", 10.0)
    b.add_igbt_level1("Q", "a", "gnd", "g", tau_tail=1e-5, k_tail=0.3)
    return b


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("case, v_dc, scale", [
    # the saturable short carries 1 A: v(a) = 0 (before: 0 → 0.99 V)
    (_saturable, 0.0, 1.0),
    # the capacitor is open: v(a) = V/2 (before: a 1.05 V excursion)
    (_nonlinear_cap, 2.5, 2.5),
    # the tail current is 0 on the DC curve (before: an 18 mV excursion)
    (_igbt_tail, None, 10.0),
])
def test_stateful_history_seeded_at_the_dc_point(case, v_dc, scale, path):
    b = case()
    dc = k.compute_dc_operating_point(
        b.graph, b.pool, k.SwitchStateMask(b.graph.num_switches),
        device_path=getattr(k.DevicePath, path.capitalize()))
    a = b.node_id_of("a")
    if v_dc is not None:
        assert dc.x[a] == pytest.approx(v_dc, abs=1e-9 * scale)
    v = np.asarray(_run(case(), path).v("a"))
    assert v[0] == pytest.approx(dc.x[a], abs=1e-12 * scale + 1e-12)
    assert np.max(np.abs(v - dc.x[a])) < 1e-8 * scale
