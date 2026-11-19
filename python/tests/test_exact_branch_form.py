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
    assert rec.kind == "cut_set" and list(rec.states) == ["L"] and rec.devices[0] == "S"
    assert abs(rec.lambda_) == pytest.approx(1e-3 * i5, rel=1e-9)
    assert rec.dW == pytest.approx(0.5e-3 * i5 ** 2, rel=1e-9)
    after = t > rec.t + 1e-9
    assert after.any() and np.max(np.abs(i[after])) <= 1e-9 * i5
    e = impulse_losses(b, r)
    assert list(e) == ["S"] and e["S"]["n_impulses"] == 1
    assert e["S"]["E_impulse_total"] == pytest.approx(0.5e-3 * i5 ** 2, rel=1e-9)


@pytest.mark.parametrize("v_rl, recorded", [(1e-7, False), (1e-3, True)])
def test_a_cut_set_is_recorded_only_above_1e_9_of_the_current_scale(v_rl, recorded):
    """The record threshold: 1e-9 of the circuit's current scale (here 1e4 A, set
    by an independent current source). A 1e-7 A cut is projected without a record;
    a 1e-3 A cut is recorded with lambda = L*i."""
    _, fn = _rl(open_at=5e-3)   # S is switch 0 in both circuits
    b2 = p.CircuitBuilder()
    b2.add_voltage_source("V", "vin", "gnd", v_rl)
    b2.add_ideal_switch("S", "vin", "a", 0.0, 0.0)
    b2.add_resistor("R", "a", "b", 1.0)
    b2.add_inductor("L", "b", "gnd", 1e-3)
    b2.add_current_source("Ibig", "gnd", "big", 1e4)
    b2.add_resistor("Rbig", "big", "gnd", 1e-3)
    r = _run(b2, 6e-3, fn)
    i5 = v_rl * (1.0 - math.exp(-5.0))
    recs = list(r.reinit_records)
    i = np.asarray(r.i("L"))
    assert np.max(np.abs(i[np.asarray(r.times) > 5e-3 + 1e-9])) <= 1e-9 * i5   # projected either way
    if recorded:
        (rec,) = recs
        assert rec.kind == "cut_set" and abs(rec.lambda_) == pytest.approx(1e-3 * i5, rel=1e-6)
    else:
        assert recs == []


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


def test_a_loop_through_a_source_books_the_impulse_to_the_switch():
    """C (0 V) closed onto the 10 V source V through the ideal S at t = 0: the record lists
    the switch first, so the 50 uJ (C*V^2/2) is the switch's, not the source's."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S", "vin", "x", 0.0, 0.0)
    b.add_capacitor("C", "x", "gnd", 1e-6, 0.0)
    m = p.SwitchStateMask(1)
    m.set(0, True)
    r = _run(b, 1e-6, lambda t: m)
    (rec,) = r.reinit_records
    assert rec.devices[0] == "S"
    assert rec.dW == pytest.approx(0.5e-6 * 100.0, rel=1e-9)
    assert list(impulse_losses(b, r)) == ["S"]


# --- ideal (branch-form) diodes on the exact engine (exact engine session 18) ---
# One complementarity pair (i, V_f - v) per device, decided on its own variables:
# ON reads the solved current, OFF the voltage. With g_off = 0 an inductor in
# series with an OFF diode is a dependent state (i_L = 0).

def _ideal_diode(b, name, a, k, g_off=0.0):
    b.add_diode(name, a, k, None, g_off, 0.0, 0.0, form="branch", R_on=0.0)


def _halfwave(L=None, Vm=10.0, f=50.0, R=10.0):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "s", "gnd", 0.0, Vm, f)
    _ideal_diode(b, "D", "s", "k")
    if L is None:
        b.add_resistor("R", "k", "gnd", R)
    else:
        b.add_resistor("R", "k", "m", R)
        b.add_inductor("L", "m", "gnd", L)
    return b


def test_an_ideal_half_wave_rectifier_is_the_clipped_sine():
    Vm, f = 10.0, 50.0
    r = _run(_halfwave(), 0.05, None)
    t, v = np.asarray(r.times), np.asarray(r.v("k"))
    np.testing.assert_allclose(v, np.maximum(Vm * np.sin(2 * np.pi * f * t), 0.0), rtol=0, atol=1e-9)


def test_an_ideal_half_wave_rl_rectifier_matches_its_closed_form():
    """i = (Vm/Z)(sin(wt - phi) + sin(phi) e^(-t/tau)) while D conducts, from rest at
    t = 0; D turns OFF at the extinction instant beta (i(beta) = 0) and L is then a
    dependent state (i_L = 0), until the source turns D ON again at t = T."""
    from scipy.optimize import brentq
    Vm, f, R, L = 10.0, 50.0, 10.0, 0.1
    w, T = 2 * np.pi * f, 1 / f
    Z, phi, tau = np.hypot(R, w * L), np.arctan2(w * L, R), L / R

    def i_on(t):
        return Vm / Z * (np.sin(w * t - phi) + np.sin(phi) * np.exp(-t / tau))

    beta = brentq(i_on, 0.5 * T, T, xtol=1e-16, rtol=1e-15)
    r = _run(_halfwave(L), 0.9 * T, None)
    t, i = np.asarray(r.times), np.asarray(r.i("L"))
    ref = np.where(t < beta, i_on(np.minimum(t, beta)), 0.0)
    np.testing.assert_allclose(i, ref, rtol=0, atol=1e-9)
    # the turn-off is located on the current itself: the event instant is beta
    assert np.min(np.abs(np.asarray(r.event_times) - beta)) < 1e-12
    # a located zero of i followed by i = 0 is not impulsive: no record
    assert len(r.reinit_records) == 0


def test_an_ideal_boost_in_dcm_matches_its_closed_form():
    """Ideal switch and ideal diode (R_on = 0, g_off = 0): ON i_L' = Vin/L; OFF the
    L-C-R pair until i_L = 0; then every switch is open and i_L is a dependent state
    (i_L = 0) while C discharges into R. Ten periods from 30 V against expm."""
    from scipy.linalg import expm
    from scipy.optimize import brentq
    Vin, L, C, R, f, D, v0 = 12.0, 100e-6, 47e-6, 500.0, 1e5, 0.5, 30.0
    T = 1 / f
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", Vin)
    b.add_inductor("L", "vin", "sw", L)
    b.add_ideal_switch("S", "sw", "gnd", 0.0, 0.0)
    _ideal_diode(b, "D", "sw", "out")
    b.add_capacitor("C", "out", "gnd", C, v0)
    b.add_resistor("R", "out", "gnd", R)
    sf = p.make_pwm_switch_fn(f, D, b.switch_index_of("S"), b.graph.num_switches)
    r = _run(b, 10 * T, sf)
    # closed form, affine states [i, v, 1]
    A_off = np.array([[0, -1 / L, Vin / L], [1 / C, -1 / (R * C), 0], [0, 0, 0]])
    a_rc = 1 / (R * C)
    v, t_off = v0, []
    for k in range(10):
        i = Vin / L * D * T
        v *= np.exp(-a_rc * D * T)
        x0 = np.array([i, v, 1.0])
        tz = brentq(lambda s: (expm(A_off * s) @ x0)[0], 1e-12, (1 - D) * T, xtol=1e-18)
        v = (expm(A_off * tz) @ x0)[1] * np.exp(-a_rc * ((1 - D) * T - tz))
        t_off.append((k + D) * T + tz)
    assert np.asarray(r.v("out"))[-1] == pytest.approx(v, abs=1e-9)
    et = np.asarray(r.event_times)
    for tz in t_off:
        assert np.min(np.abs(et - tz)) < 1e-12
    assert len(r.reinit_records) == 0   # ten located turn-offs, none impulsive


def test_an_ideal_diode_forward_across_a_source_is_refused_by_name():
    """ON, the diode closes a loop of ideal elements with V: no mode can pose it."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 10.0)
    _ideal_diode(b, "D", "a", "gnd")
    b.add_resistor("R", "a", "gnd", 1.0)
    with pytest.raises(p.UnsupportedError) as ei:
        _run(b, 1e-6, None)
    assert ei.value.feature == "ideal_voltage_loop" and "D" in ei.value.device


def test_a_branch_form_diode_with_its_own_r_on_divides_the_sine():
    """R_on = 1 ohm in series with R = 10 ohm: v_k = max(V, 0)·R/(R + R_on)."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "s", "gnd", 0.0, 10.0, 50.0)
    b.add_diode("D", "s", "k", None, 0.0, 0.0, 0.0, form="branch", R_on=1.0)
    b.add_resistor("R", "k", "gnd", 10.0)
    r = _run(b, 0.05, None)
    t, v = np.asarray(r.times), np.asarray(r.v("k"))
    ref = np.maximum(10.0 * np.sin(2 * np.pi * 50.0 * t), 0.0) * 10.0 / 11.0
    np.testing.assert_allclose(v, ref, rtol=0, atol=1e-9)


def _doubler(form):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "ac", "gnd", 0.0, 12.0, 100.0)
    b.add_capacitor("C1", "ac", "cp", 100e-6, 0.0)
    for name, a, c in (("D2", "gnd", "cp"), ("D1", "cp", "out")):
        if form == "branch":
            b.add_diode(name, a, c, None, 1e-6, form="branch", R_on=1 / 200.0)
        else:
            b.add_diode(name, a, c, 200.0, 1e-6)
    b.add_capacitor("Cout", "out", "gnd", 470e-6, 0.0)
    b.add_resistor("R", "out", "gnd", 1e3)
    return b


def test_a_branch_diode_with_r_on_is_the_conductance_diode():
    """R_on > 0: v_D = V_f + R_on*i is read across the nodes, as the conductance form
    reads it. Read as its current, D2's turn-on (i = 0 tied with the OFF side, a
    5e-7 s R_on*C1 pole) chattered 32 times at 3.4557 ms."""
    rb, rc = (_run(_doubler(f), 0.05, None, dt=50e-6) for f in ("branch", "conductance"))
    tb, tc = np.asarray(rb.times), np.asarray(rc.times)
    vb = np.interp(tc, tb, np.asarray(rb.v("out")))
    np.testing.assert_allclose(vb, np.asarray(rc.v("out")), rtol=0, atol=1e-6)


@pytest.mark.parametrize("mode_selection", [None, "lcp"])
def test_two_ideal_diodes_in_parallel_split_by_an_admissible_mode(mode_selection):
    """Both ON is a loop of closed ideal elements (no mode to pose); one ON and one
    OFF is admissible (the OFF one sits at v_D = 0): the run takes it, it is not
    refused. The split itself is not unique."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, 10.0, 50.0)
    _ideal_diode(b, "D1", "a", "k")
    _ideal_diode(b, "D2", "a", "k")
    b.add_resistor("R", "k", "gnd", 10.0)
    kw = {} if mode_selection is None else {"mode_selection": mode_selection}
    r = _run(b, 0.02, None, **kw)
    t = np.asarray(r.times)
    np.testing.assert_allclose(np.asarray(r.v("k")),
                               np.maximum(10 * np.sin(2 * np.pi * 50 * t), 0.0), rtol=0, atol=1e-9)
    i1, i2 = np.asarray(r.i("D1")), np.asarray(r.i("D2"))
    np.testing.assert_allclose(i1 + i2, np.asarray(r.v("k")) / 10.0, rtol=0, atol=1e-9)
    assert min(i1.min(), i2.min()) > -1e-9


def test_an_ideal_clamp_onto_a_lower_source_is_refused_by_name():
    """D1 from the sine and D2 clamping onto 3 V: above 3 V no mode is admissible
    (both ON short the two sources; either OFF breaks its law)."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, 10.0, 50.0)
    _ideal_diode(b, "D1", "a", "k")
    b.add_resistor("R", "k", "gnd", 10.0)
    b.add_voltage_source("V2", "m", "gnd", 3.0)
    _ideal_diode(b, "D2", "k", "m")
    with pytest.raises(p.UnsupportedError) as ei:
        _run(b, 0.02, None)
    assert ei.value.feature == "ideal_voltage_loop"


def test_the_periodic_steady_state_refuses_an_ideal_diode_by_name():
    """The exact period map reads a diode's current as g*v: a branch-form diode is
    refused there by name (it was a 'grazing' refusal: R_on = 0 zeroes g*v)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_inductor("L", "vin", "sw", 100e-6)
    b.add_ideal_switch("S", "sw", "gnd", 0.0, 0.0)
    _ideal_diode(b, "D", "sw", "out")
    b.add_capacitor("C", "out", "gnd", 47e-6, 30.0)
    b.add_resistor("R", "out", "gnd", 500.0)
    sf = p.make_pwm_switch_fn(1e5, 0.5, b.switch_index_of("S"), b.graph.num_switches)
    with pytest.raises(p.UnsupportedError) as ei:
        p.periodic_steady_state(b, period=1e-5, switch_fn=sf)
    assert ei.value.feature == "branch_form" and ei.value.device.startswith("D")

