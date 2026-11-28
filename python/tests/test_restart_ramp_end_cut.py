"""A current that ramps to zero inside a restart piece is cut there, not dropped as an impulse.

A boost cell: 12 V, L = 50 uH, a 0.25 us gate pulse each 10 us (i_pk = 60 mA), an exponential
(Shockley, Newton) freewheel diode into C = 1 uF || 1 kOhm held at v0. After the turn-off the
current falls at r = (12 - v0 - V_D)/L and ends at theta = i_pk/|r| inside the piece [10.25,
10.5] us (h = 0.25 us, gamma*h = 73 ns). For theta < 1.185*gamma*h the restart probe's jump test
reads that ramp as a jump: it used to book a `cut_set` and discard the 60 mA (the charge
1/2*i*theta never reached the output). It is now cut at theta (piece A on the slope, piece B
from rest) while theta >= gamma^2*h (what the probe can tell from a fast mode); below that the
impulse stays (the rule's own resolution), and a genuine cut still books 1/2*L*i^2.

Charge bound (derived): the scheme integrates a linear ramp exactly; the ramp's curvature is
the diode's forward voltage moving over the conduction, at most n*V_T*ln(i_pk/I_S) against
|v_L|, so |Q - Q_fine| <= 2*(n*V_T*ln(i_pk/I_S)/|v_L|) * 1/2*i_pk*theta (a factor 2 margin), and
the step-end output voltage moves by that over C.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import pulsim as p

L, C, R, VIN, IS, N = 50e-6, 1e-6, 1e3, 12.0, 1e-12, 1.2
DT, TE, TB = 0.5e-6, 10.25e-6, 10.5e-6
IPK = VIN * 0.25e-6 / L


def _run(v0, k, diode=True):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", VIN)
    b.add_inductor("L", "vin", "sw", L)
    b.add_switch("S", "sw", "gnd", 50.0, 1e-7)
    if diode:
        b.add_shockley_diode("D", "sw", "out", I_S=IS, n=N)
    b.add_capacitor("C", "out", "gnd", C, v0)
    b.add_resistor("R", "out", "gnd", R)
    fn = p.make_pwm_switch_fn(100e3, 0.025, b.switch_index_of("S"), b.graph.num_switches)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, 11e-6, engine="pwl", dt=DT / k, switch_fn=fn,
                          discontinuity_restart=True, enable_substep_state_correction=True)


def _edge(r):
    t = np.asarray(r.times)
    v = np.asarray(r.v("out"))
    m = (t >= TE - 0.25e-6 - 1e-13) & (t <= TB + 1e-13)
    q = C * (v[m][-1] - v[m][0]) + np.trapezoid(v[m], t[m]) / R   # into C || R over [10, 10.5]
    cs = [rec for rec in r.reinit_records if rec.kind == "cut_set" and TE - 0.3e-6 < rec.t < TB]
    return q, float(np.interp(TB, t, v)), cs


def _bound(v0):
    vl = v0 + 0.7 - VIN   # |v_L| during the ramp
    theta = IPK * L / vl
    return 2 * (N * 0.02585 * math.log(IPK / IS) / vl) * 0.5 * IPK * theta, theta


def test_a_ramp_ending_at_0p3_h_is_cut_not_dropped():
    v0 = 52.0   # theta = 0.31 h: inside the jump test's window, above gamma^2*h
    q, vb, cs = _edge(_run(v0, 1))
    q_f, vb_f, cs_f = _edge(_run(v0, 200))
    bq, theta = _bound(v0)
    assert 0.25 < theta / 0.25e-6 < 0.35 and not cs_f
    assert not cs   # no impulse: the current reached the output
    assert abs(q - q_f) <= bq   # the base delivered 0.089 nC of 2.24
    assert abs(q - 0.5 * IPK * theta) <= 0.1 * 0.5 * IPK * theta
    assert abs(vb - vb_f) <= bq / C


@pytest.mark.xfail(strict=True, reason="a ramp ending at 0.72 h is outside the jump test's window: "
                   "SDIRK integrates across the kink, +17 % charge (docs/known-limitations.md)")
def test_a_ramp_ending_at_0p7_h_is_integrated_within_the_bound():
    v0 = 28.4
    q, _, cs = _edge(_run(v0, 1))
    q_f, _, _ = _edge(_run(v0, 200))
    assert not cs
    assert abs(q - q_f) <= _bound(v0)[0]


def test_a_ramp_shorter_than_gamma2_h_stays_an_impulse():
    v0 = 200.0   # theta = 0.07 h < gamma^2*h = 0.086 h: unresolved, the rule's impulse
    _, _, cs = _edge(_run(v0, 1))
    assert len(cs) == 1 and cs[0].t == pytest.approx(TE, abs=1e-12)
    assert cs[0].dW == pytest.approx(0.5 * L * IPK**2, rel=1e-3)


def test_a_genuine_cut_still_books_its_energy():
    _, _, cs = _edge(_run(36.0, 1, diode=False))
    assert len(cs) == 1
    assert cs[0].dW == pytest.approx(0.5 * L * IPK**2, rel=1e-3)
