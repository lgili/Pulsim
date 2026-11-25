"""F-S-6: a jump across a nonlinear (charge-form) capacitor under the restart rule.

What an impulsive event conserves is charge; the law q(v) maps it to voltage, and the
energy the impulse dissipates is the law's, never 1/2 C dV^2 with a frozen C. Every truth
here is a closed form or an mpmath quadrature of the law; no number comes from a Pulsim run.

Hand case: C1 (charged to V1 = 100 V through 10 Ohm and Ssrc, which then opens) and C2 (at 0 V),
two different junction laws C_k(v) = C0_k / (1 + v/V0_k)^m_k, shared through a 1 mOhm
switch S at t_c (tau ~ ps << dt). Charge on the node set {n1, n2} is conserved:
    Q1(v+) + Q2(v+) = Q1(V1) + Q2(0),
and the energy lost is what is no longer stored:
    E = W1(V1) - W1(v+) - W2(v+),   W_k(v) = int_0^v u C_k(u) du.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p

mp = pytest.importorskip("mpmath")
mp.mp.dps = 30

V1, T_O, T_C, DT = 100.0, 1e-6, 2e-6, 1e-8
LAWS = {"C1": (1e-9, 25.0, 0.5), "C2": (2e-9, 10.0, 1.0 / 3.0)}


def _cap(name):
    c0, v0, m = LAWS[name]
    return lambda v: mp.mpf(c0) / (1 + mp.mpf(v) / v0) ** m


def _q(name, v):
    return mp.quad(_cap(name), [0, v])


def _w(name, v):
    c = _cap(name)
    return mp.quad(lambda u: u * c(u), [0, v])


def _truth():
    q_tot = _q("C1", V1)
    vp = mp.findroot(lambda v: _q("C1", v) + _q("C2", v) - q_tot, V1 / 3)
    e = _w("C1", V1) - _w("C1", vp) - _w("C2", vp)
    return float(vp), float(e)


class _Gates:
    """Ssrc closed before T_O, S closed from T_C."""

    def __init__(self, b):
        self.n, self.src, self.s = b.graph.num_switches, b.switch_index_of("Ssrc"), b.switch_index_of("S")

    def __call__(self, t):
        m = p.SwitchStateMask(self.n)
        m.set(self.src, t < T_O - 1e-15)
        m.set(self.s, t >= T_C - 1e-15)
        return m

    def next_edge_after(self, t):
        return next((e for e in (T_O, T_C) if t < e - 1e-15), float("inf"))


def _run(**kw):
    b = p.CircuitBuilder()
    # ramped: a consistent start at 0 V
    b.add_pulse_voltage_source("Vs", "vs", "gnd", 0.0, V1, 0.0, 1.0, 0.0, 1e-7, 0.0)
    # through 10 Ohm (tau ~ dt: no stale trapezoidal ring of a 1 ps RC at the ramp's end)
    b.add_resistor("Rs", "vs", "a", 10.0)
    b.add_switch("Ssrc", "a", "n1", 1e3, 1e-12)
    b.add_switch("S", "n1", "n2", 1e3, 1e-12)
    for name, node in (("C1", "n1"), ("C2", "n2")):
        c0, v0, m = LAWS[name]
        b.add_nonlinear_capacitor(name, node, "gnd", c0, v0, m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=3e-6, dt=DT, switch_fn=_Gates(b), **kw)


@pytest.mark.parametrize("split", [False, True])
def test_charge_sharing_between_two_nonlinear_capacitors(split):
    vp, e = _truth()
    assert 20.0 < vp < 40.0 and e > 1e-6           # a real jump, a real loss
    r = _run(engine="pwl", discontinuity_restart=True, enable_substep_state_correction=split)
    t = np.asarray(r.times)
    v1, v2 = np.asarray(r.v("n1")), np.asarray(r.v("n2"))
    after = t > T_C + 1.5 * DT
    # charge conserved: both nodes at v+ (the 1e-12 S leakage moves 1e-16 C)
    assert np.max(np.abs(v1[after] - vp)) < 1e-6 * V1
    assert np.max(np.abs(v2[after] - vp)) < 1e-6 * V1
    recs = list(r.reinit_records)
    assert len(recs) == 1, [(x.t, list(x.states), x.dW) for x in recs]
    rec = recs[0]
    assert rec.t == pytest.approx(T_C, abs=1e-15)
    assert set(rec.states) == {"C1", "C2"}
    # the energy from the laws (1/2 C dV^2 with C at the pre-event voltages: -3 %)
    assert rec.dW == pytest.approx(e, rel=1e-6)
    # the moved charge, on the dominant capacitor
    dq = {"C1": float(_q("C1", vp) - _q("C1", V1)), "C2": float(_q("C2", vp))}
    assert any(rec.lambda_ == pytest.approx(d, rel=1e-6) for d in dq.values())


def test_no_record_without_the_switch():
    """The source opening at T_O (no current through it) is no jump: no record."""
    r = _run(engine="pwl", discontinuity_restart=True)
    assert [x.t for x in r.reinit_records if x.t < T_C - DT] == []


@pytest.mark.parametrize("i2", [0.224, 0.230, 0.236, 0.244])
def test_a_constant_current_through_a_steep_law_is_no_jump(i2):
    """Reviewer's case: -i2 into a tanh-law capacitor (C 10 nF -> 0.1 nF across the knee),
    an unrelated switch edge at 150 ns. q(t) is a straight line (no jump in charge); in
    voltage the stages curve 100x and a voltage test read a jump (the base refused it by
    name). The test runs in charge: no record."""
    b = p.CircuitBuilder()
    b.add_pwl_current_source("I", "c", "gnd", [0, 1e-9, 1e-7, 1e-7 + 1e-9, 1.0],
                             [0, 0.2, 0.2, -i2, -i2])
    b.add_resistor("Rbig", "c", "gnd", 1e9)
    b.add_nonlinear_capacitor("C", "c", "gnd", p.ChargeLaw.tanh(10e-9, 0.1e-9, 1.0))
    b.add_voltage_source("V", "s", "gnd", 1.0)
    b.add_switch("S", "s", "r", 1e3, 1e-12)
    b.add_resistor("R", "r", "gnd", 1e3)
    n, k = b.graph.num_switches, b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, t >= 150e-9 - 1e-15)
        return m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=200e-9, dt=1e-8, switch_fn=fn, engine="pwl",
                       discontinuity_restart=True)
    assert list(r.reinit_records) == []
