"""``engine='dsed'`` with ``integrator='auto'`` takes the exact stepper.

Before this change the auto scheduler sent every mode its stiffness
detector flagged to fixed-step BDF2 (h = 1 µs) without asking whether
the mode had a closed-form propagator, and BDF2 started each segment
with a Crank-Nicolson step, which is not L-stable. Circuit: 100 V,
10 Ω, a 1 mΩ switch to ground with 1 nF across it, 100 kHz, open for
the first half of each period (τ = 10 ns while open):

    integrator  samples  v(sw) range            max error, open interval
    'auto'      42       -99.98 ... 196.08 V    96.07 V
    'rk45'      42        0.010 ... 100.000 V   1.0e-6 V
    'bdf2'      402      -99.98 ... 166.67 V    66.66 V
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p

G_ON, G_OFF, CS, T = 1e3, 1e-9, 1e-9, 1e-5
TAU = 10.0 * CS        # open switch: R = 10 ohm, C = 1 nF


def _run(nper=4, **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 100.0)
    b.add_resistor("R", "in", "sw", 10.0)
    b.add_switch("S", "sw", "gnd", G_ON, G_OFF)
    b.add_capacitor("C", "sw", "gnd", CS)
    n = b.graph.num_switches
    k = b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(n)
        fr = t / T - np.floor(t / T + 1e-9)
        m.set(k, bool(fr >= 0.5 - 1e-9))   # open first half, closed second
        return m

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=nper * T, engine="dsed", switch_fn=fn, **kw)


def test_auto_takes_the_exact_stepper():
    res = _run()
    t = np.asarray(res.times)
    v = np.asarray(res.v("sw"))
    m = (t > T + 1e-9) & (t < 1.5 * T - 1e-9)
    exact = 100.0 - (100.0 - 0.01) * np.exp(-(t[m] - T) / TAU)
    assert np.abs(v[m] - exact).max() <= 1e-5          # was 96.07 V
    assert v.min() >= 0.0 and v.max() <= 100.001       # was -99.98 ... 196.08 V
    assert res.n_exact_steps > 0
    assert res.n_bdf2_steps == 0


def test_bdf2_starts_each_segment_with_backward_euler():
    res = _run(integrator="bdf2")
    t = np.asarray(res.times)
    v = np.asarray(res.v("sw"))
    for kper in range(4):
        i = np.searchsorted(t, kper * T + 1e-12)        # first sample after the opening
        h = t[i] - t[i - 1]
        z = h / TAU
        assert v[i] == pytest.approx(100.0 * z / (1.0 + z), abs=0.01), (kper, h)  # was 166.67 V
    assert v.max() <= 102.8
    assert v.min() >= 0.0                               # was -99.98 V


def test_a_mode_without_an_exact_stepper_still_runs_on_auto():
    # A user b_extra_fn has no closed-form propagator (sine, pulse and PWM
    # sources do), so 'auto' integrates numerically (DOPRI5/BDF2) and must
    # still match the fixed engine. The callable adds zero: the physics is
    # the pulse-driven RC the pwl run sees.
    def build():
        b = p.CircuitBuilder()
        b.add_pulse_voltage_source("V", "in", "gnd", 0.0, 10.0, 0.2e-3,
                                   0.5e-3, 1e-3, 50e-6, 50e-6)
        b.add_resistor("R", "in", "o", 100.0)
        b.add_capacitor("C", "o", "gnd", 1e-6)
        return b
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bd = build()
        zero = [0.0] * bd.pool.state_size(bd.graph)
        rd = p.simulate(bd, t_end=2e-3, engine="dsed", b_extra_fn=lambda _t: zero,
                        allow_implicit_segments=True)
        rp = p.simulate(build(), t_end=2e-3, engine="pwl", dt=1e-7)
    assert rd.n_exact_steps == 0
    assert rd.n_rk45_steps + rd.n_bdf2_steps > 0
    td, vd = np.asarray(rd.times), np.asarray(rd.v("o"))
    tp, vp = np.asarray(rp.times), np.asarray(rp.v("o"))
    span = vp.max() - vp.min()
    assert np.abs(np.interp(td, tp, vp) - vd).max() <= 1e-3 * span
