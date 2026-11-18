"""add-branch-form-ideal-switch task 8 on the exact engine: the controlled ideal
switch (R_on = 0, g_off = 0) as a branch unknown, against the closed forms of
`tests/test_branch_form_reinit.py` / `oracles/ideal_switch_reference.py`.

dsed solves each mode exactly, so its numbers are the closed forms themselves (no
step-size error, unlike the pwl oracles at h = tau/20): RL 10 V / 1 ohm / 1 mH,
i(10 ms) = 10(1 - e^-10); the same opened at 5 ms records L*i(5 ms) and
L*i^2/2 with i(5 ms) = 10(1 - e^-5); 1 uF at 10 V onto 3 uF at 2 V lands both on
4 V moving 6 uC and removing 24 uJ, booked to the switch as impulse loss.
"""
import math
import warnings

import numpy as np
import pytest

import pulsim as p
from pulsim.losses import impulse_losses


def _rl(open_at=None):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S", "vin", "a", 0.0, 0.0)
    b.add_resistor("R", "a", "b", 1.0)
    b.add_inductor("L", "b", "gnd", 1e-3)
    k = b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(1)
        m.set(k, open_at is None or t < open_at)
        return m
    return b, fn


def _run(b, t_end, fn, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=t_end, engine="dsed", switch_fn=fn, **kw)


def test_rl_closed_reaches_the_closed_form():
    b, fn = _rl()
    r = _run(b, 10e-3, fn)
    i = np.asarray(r.i("L"))
    assert i[-1] == pytest.approx(10.0 * (1.0 - math.exp(-10.0)), rel=1e-9)
    assert np.asarray(r.i("S"))[-1] == pytest.approx(i[-1], rel=1e-9)   # the solved unknown


def test_rl_cut_set_is_reinitialised_and_recorded_exactly():
    b, fn = _rl(open_at=5e-3)
    r = _run(b, 6e-3, fn)
    t, i = np.asarray(r.times), np.asarray(r.i("L"))
    (rec,) = r.reinit_records
    # the polled switch_fn's edge is located by bisection: judge the record at its own t
    assert rec.t == pytest.approx(5e-3, abs=1e-9)
    i5 = 10.0 * (1.0 - math.exp(-rec.t / 1e-3))
    assert rec.kind == "cut_set" and list(rec.states) == ["L"] and list(rec.devices) == ["S"]
    assert abs(rec.lambda_) == pytest.approx(1e-3 * i5, rel=1e-9)
    assert rec.dW == pytest.approx(0.5e-3 * i5 ** 2, rel=1e-9)
    after = t > rec.t + 1e-9
    assert after.any() and np.max(np.abs(i[after])) <= 1e-9 * i5
    e = impulse_losses(b, r)
    assert list(e) == ["S"] and e["S"]["n_impulses"] == 1
    assert e["S"]["E_impulse_total"] == pytest.approx(0.5e-3 * i5 ** 2, rel=1e-9)


def test_capacitor_loop_lands_on_four_volts_and_books_24_uj_to_the_switch():
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "x", "gnd", 1e-6, 10.0)
    b.add_capacitor("C2", "y", "gnd", 3e-6, 2.0)
    b.add_ideal_switch("S", "x", "y", 0.0, 0.0)
    b.add_resistor("RL", "y", "gnd", 1e9)        # a reference path, tau = 4 ks
    m = p.SwitchStateMask(1)
    m.set(0, True)
    r = _run(b, 1e-6, lambda t: m)
    (rec,) = r.reinit_records
    assert rec.kind == "capacitor_loop" and sorted(rec.states) == ["C1", "C2"]
    assert list(rec.devices) == ["S"]
    assert abs(rec.lambda_) == pytest.approx(6.0e-6, rel=1e-9)
    assert rec.dW == pytest.approx(24.0e-6, rel=1e-9)
    v1, v2 = np.asarray(r.v("x")), np.asarray(r.v("y"))
    np.testing.assert_allclose(v1[1:], 4.0, rtol=0, atol=1e-8)   # 1e-15 /s leak over 1 us
    np.testing.assert_allclose(v2[1:], v1[1:], rtol=0, atol=1e-9)
    e = impulse_losses(b, r)
    assert e["S"]["E_impulse_total"] == pytest.approx(24.0e-6, rel=1e-9)


def test_two_open_ideal_switches_in_series_float_their_node():
    """The pwl circuit of test_branch_form_switch, refused by the same name."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S1", "vin", "mid", 0.0, 0.0)
    b.add_ideal_switch("S2", "mid", "gnd", 0.0, 0.0)
    b.add_resistor("R", "vin", "gnd", 10.0)
    m = p.SwitchStateMask(2)
    with pytest.raises(p.UnsupportedError) as ei:
        _run(b, 1e-5, lambda t: m)
    e = ei.value
    assert (e.feature, e.kind, e.device) == ("floating_node", "Switch", "S1, S2")
    assert "mid" in str(e)


def test_two_closed_ideal_switches_in_parallel_across_a_source():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S1", "vin", "gnd")
    b.add_ideal_switch("S2", "vin", "gnd")
    m = p.SwitchStateMask(2)
    m.set(0, True)
    m.set(1, True)
    with pytest.raises(p.UnsupportedError) as ei:
        _run(b, 1e-5, lambda t: m)
    e = ei.value
    assert (e.feature, e.kind, e.device) == ("ideal_voltage_loop", "Switch", "S1")


def test_a_branch_form_diode_is_still_refused_by_name():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 10.0)
    b.add_diode("D", "a", "k", form="branch")
    b.add_resistor("R", "k", "gnd", 1.0)
    with pytest.raises(p.UnsupportedError) as ei:
        _run(b, 1e-6, None)
    assert ei.value.feature == "branch_form"
