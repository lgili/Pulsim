"""Sensitivities through a closed loop on the exact path
(add-parameter-sensitivities group 8, add-compiled-control-graph: the
tick Jacobian).

The loop of ``bind_pi_to_switch`` on the closed-loop buck of
``test_closed_loop_reference.py`` (voltage-mode PI, CCM, 100 kHz): at
each tick the controller's state and the duty move by the exact Jacobian
of the tick map (the block laws on the AD scalar,
``CxxBlockChain.tick_jacobian``), and the duty moves the gate's OFF edge
by the modulator's ``d t_off / d duty = T`` inside the event jump.

Truths: central differences of the full simulation (each run the same
exact map, rtol 1e-11) — the step-size sweep (h/p = 1e-2 ... 1e-7, macOS
arm64, 2026-11-19) shows the O(h^2) slope down to a plateau at h/p ~ 1e-5
where AD and FD agree to <= 7e-9 relative on every parameter; the tests
pin h/p = 1e-5 and assert 1e-6 (>= 100x margin). Closed form: integral
action makes v(kT) -> V_ref exactly, so at steady state dV/dV_ref = 1 and
dV/dK_p = dV/dK_i = d/d(plant) = 0.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p
import pulsim.sensitivity as sens

T = 1e-5
PMAP = {"pi.Kp": "Kp", "pi.Ki": "Ki", "pi.setpoint": "Vref", "L1.L": "L",
        "R.R": "R", "pi.output_max": "hi"}
BASE = dict(Kp=0.05, Ki=200.0, Vref=5.0, L=10e-6, R=10.0, hi=1.0)


def _build(Kp, Ki, Vref, L, R, hi):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 10.0)
    b.add_switch("SH", "vin", "sw", 100.0, 1e-6)
    b.add_diode("D", "gnd", "sw", 100.0, 1e-6)
    b.add_resistor("R_dcr", "sw", "n_l", 0.2)
    b.add_inductor("L1", "n_l", "n_esr", L)
    b.add_resistor("R_esr", "n_esr", "out", 0.05)
    b.add_capacitor("C1", "out", "gnd", 100e-6)
    b.add_resistor("R", "out", "gnd", R)
    pi = p.PIController(Kp=Kp, Ki=Ki, output_min=0.0, output_max=hi)
    loop = p.bind_pi_to_switch(b, pi=pi, measured="out", setpoint=Vref,
                               switch="SH", freq=1.0 / T)
    return b, loop


def _run(t_end, names, **kw):
    b, loop = _build(**{**BASE, **kw})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end, engine="dsed", switch_fn=loop.switch_fn,
                          step_observer=loop.step_observer,
                          sensitivities=names)


def _at(res, k):
    t = np.asarray(res.times)
    j = int(np.searchsorted(t, k * T * (1 - 1e-12)))
    assert abs(t[j] - k * T) <= 1e-12 * k * T + 1e-18   # a tick sample
    return j


def _fd(t_end, name, ks, rel=1e-5, **kw):
    key = PMAP[name]
    v0 = {**BASE, **kw}[key]
    h = rel * abs(v0)
    rp = _run(t_end, ["R.R"], **{**kw, key: v0 + h})
    rm = _run(t_end, ["R.R"], **{**kw, key: v0 - h})
    vp, vm = np.asarray(rp.v("out")), np.asarray(rm.v("out"))
    return np.array([(vp[_at(rp, k)] - vm[_at(rm, k)]) / (2 * h) for k in ks])


def _ad(res, name, ks):
    s = res.sens("out", name)
    return np.array([s[_at(res, k)] for k in ks])


def test_the_tick_jacobian_of_the_pi_graph_is_its_closed_form():
    b, loop = _build(**BASE)
    cxx = loop.switch_fn._graph._cxx_chain
    x = np.zeros(int(b.pool.state_size(b.graph)))
    x[b.node_id_of("out")] = 4.0
    j = cxx.tick_jacobian(1, 0.0, x, ["pi.Kp", "pi.Ki", "pi.setpoint"], True)
    rows, cols, J = j["rows"], j["cols"], np.asarray(j["J"])
    r, c = rows.index, cols.index
    e, dt, Kp, Ki = 1.0, T, 0.05, 200.0
    duty = r("sig:duty")
    assert J[duty, c("pi.Kp")] == e                                  # du/dKp = e
    assert J[duty, c("pi.Ki")] == pytest.approx(dt * 0.5 * e, rel=1e-15)
    assert J[r("pi.integrator_state"), c("pi.Ki")] == pytest.approx(dt * 0.5 * e, rel=1e-15)
    probe = c(f"x[{b.node_id_of('out')}]")
    assert J[duty, probe] == pytest.approx(-(Kp + Ki * dt * 0.5), rel=1e-15)
    assert J[duty, c("pi.setpoint")] == pytest.approx(Kp + Ki * dt * 0.5, rel=1e-15)
    assert J[duty, c("pi.integrator_state")] == 1.0
    # commit ran the graph's own tick: the duty its gate now reads
    assert cxx.sink_value("duty") == j["values"][duty]
    assert all(k["live"] and not k["side"] or k["law"] in ("pl_min", "pl_max")
               for k in j["kinks"])
    # an unknown name is refused naming what exists
    with pytest.raises(ValueError, match="pi.Kp"):
        cxx.tick_jacobian(1, 0.0, x, ["pi.Kq"], False)


@pytest.mark.parametrize("name", list(PMAP)[:5])
def test_closed_loop_transient_matches_central_differences(name):
    ks = [10, 30, 60, 99]
    res = _run(1e-3, list(PMAP)[:5])
    ad, fd = _ad(res, name, ks), _fd(1e-3, name, ks)
    assert np.all(np.abs(ad) > 1e-6 * np.abs(ad).max())     # a live path
    np.testing.assert_allclose(ad, fd, rtol=1e-6)


def test_steady_state_of_integral_control_is_the_closed_form():
    # v(kT) -> V_ref exactly: dV/dV_ref = 1, every other parameter 0
    res = _run(20e-3, list(PMAP)[:5])
    k = 1999
    assert res.sens("out", "pi.setpoint")[_at(res, k)] == pytest.approx(1.0, abs=1e-9)
    early = _at(res, 30)
    for nm in ("pi.Kp", "pi.Ki", "L1.L", "R.R"):
        s = res.sens("out", nm)
        assert abs(s[_at(res, k)]) <= 1e-9 * abs(s[early])


def test_a_saturating_duty_is_right_on_both_sides_of_the_saturation():
    # Kp = 0.3, output_max = 0.5: the duty is clamped for ticks 0..4
    kw = dict(Kp=0.3, hi=0.5)
    names = ["pi.Kp", "pi.setpoint", "L1.L", "pi.output_max"]
    res = _run(2e-3, names, **kw)
    sides = res.sensitivity.control["sides"]
    duty = res.sensitivity.control["duty"][:, 0]
    assert np.all(duty[:5] == 0.5) and np.all(duty[5:20] < 0.5)
    assert [i for i in range(1, len(sides)) if sides[i] != sides[i - 1]] == [5]
    ks = [3, 6, 8, 50, 150]
    for nm in names:
        ad, fd = _ad(res, nm, ks), _fd(2e-3, nm, ks, **kw)
        np.testing.assert_allclose(ad, fd, rtol=1e-6, atol=1e-12 * np.abs(ad).max())
    # clamped: the duty does not see Kp, so neither does v(3T) (a reason)
    assert _ad(res, "pi.Kp", [3])[0] == 0.0
    assert _ad(res, "pi.output_max", [3])[0] != 0.0


def test_dropping_the_edge_term_or_the_loop_term_breaks_the_agreement(monkeypatch):
    ks = [30, 60]
    fd = _fd(1e-3, "pi.Kp", ks)
    monkeypatch.setattr(sens._Loop, "edge_dtau", lambda self, ev: 0.0)
    no_edge = _ad(_run(1e-3, ["pi.Kp"]), "pi.Kp", ks)
    assert np.all(np.abs(no_edge - fd) > 1e-2 * np.abs(fd))
    monkeypatch.undo()
    real = sens._Loop._jac

    def blind(self, t, xf, params, commit):     # the controller never sees the plant
        j = real(self, t, xf, params, commit)
        J = np.asarray(j["J"]).copy()
        for c, name in enumerate(j["cols"]):
            if name.startswith("x["):
                J[:, c] = 0.0
        j["J"] = J
        return j
    monkeypatch.setattr(sens._Loop, "_jac", blind)
    no_loop = _ad(_run(1e-3, ["pi.Kp"]), "pi.Kp", ks)
    assert np.all(np.abs(no_loop - fd) > 1e-2 * np.abs(fd))


def test_a_tick_on_a_kink_is_refused_by_name():
    # output_max equal to the first tick's unclamped output: u > hi is
    # decided ON its switch point
    e = BASE["Vref"]
    u = BASE["Kp"] * e + (0.0 + BASE["Ki"] * T * 0.5 * (e + 0.0))
    with pytest.raises(p.UnsupportedError) as ex:
        _run(1e-4, ["pi.Kp"], hi=u)
    assert ex.value.feature == "control_kink"
    assert "pi" in str(ex.value)


def test_closed_loop_refusals():
    with pytest.raises(p.UnsupportedError) as ex:
        _run(1e-4, ["SH.duty"])
    assert ex.value.feature == "sensitivity_parameter"
    with pytest.raises(ValueError, match="no component named"):
        _run(1e-4, ["pi.Kq"])
    # the gate alone, without its controller's ticks: refused, not run open
    b, loop = _build(**BASE)
    with pytest.raises(p.UnsupportedError) as ex:
        sens.simulate_sensitivities(b, 1e-4, sensitivities=["R.R"],
                                    switch_fn=loop.switch_fn)
    assert ex.value.feature == "closed_loop"


def test_coinciding_gate_edges_shift_once_or_are_refused():
    # two legs on one duty: one OFF event, shifted by T S_d once (not twice)
    lp = object.__new__(sens._Loop)
    lp.legs = [sens._Leg(0, 3, T, 0.0, 0.0, T), sens._Leg(1, 3, T, 0.0, 0.0, T)]
    lp.duty = [0.25, 0.25]
    lp.S_duty = [np.array([2.0, -1.0]), np.array([2.0, -1.0])]
    ev = {"t": 0.25 * T, "old": {0: True, 1: True}, "new": {0: False, 1: False}}
    np.testing.assert_array_equal(lp.edge_dtau(ev), T * np.array([2.0, -1.0]))
    # different duties that happen to meet: the event splits under a perturbation
    lp.S_duty[1] = np.array([1.0, 0.0])
    with pytest.raises(p.UnsupportedError) as ex:
        lp.edge_dtau(ev)
    assert ex.value.feature == "closed_loop"
