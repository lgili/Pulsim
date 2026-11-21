"""The composite MOSFET / IGBT diodes: ``V_F`` is a conduction DROP.

``add_mosfet``, ``add_mosfet_with_body_diode`` and the level-1 body /
freewheeling diodes build ``add_diode(V_th=0, V_f=V_F)`` (PSIM style,
the same mapping as ``add_three_phase_vsi``): while on, v = V_F + i/g_on.
Before this change ``V_F`` was passed as the turn-on THRESHOLD only, so
a conducting body diode dropped i/g_on and nothing else.

Measured before (10 A forced source -> drain, gate off): |v_SD| = 0.010 V,
7 mW where a 0.7 V diode dissipates 7.10 W; buck freewheeling through a
gate-off ``add_mosfet``: V_out 11.990 V and P(QL_body) 0.052 W against
11.640 V and 3.479 W. Level-1 diodes turned on at a hard-coded 0.5 V.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p

I_FORCED = 10.0
T_RUN, DT = 1e-4, 1e-6


def _mask(b, on=()):
    m = p.SwitchStateMask(b.graph.num_switches)
    for name in on:
        m.set(b.switch_index_of(name), True)
    return m


def _forced(add, *, gate_on=False, sw="Q"):
    """10 A pushed into node ``s``; the device sits between ``s`` (its
    source / emitter, the diode anode) and ground (drain / collector)."""
    b = p.CircuitBuilder()
    b.add_current_source("I", "s", "gnd", I_FORCED)
    b.add_resistor("Rb", "s", "gnd", 1e6)
    add(b)
    m = _mask(b, [sw] if gate_on else [])
    return b, (lambda t: m)


def _v_s(b, fn, engine="pwl"):
    kw = {"dt": DT} if engine == "pwl" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, T_RUN, engine=engine, switch_fn=fn, **kw)
    return res, float(np.asarray(res.v("s"))[-1])


def _p_avg(b, res, name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = {e["name"]: e for e in p.device_loss_summary(b, res)}
    return rows[name]["P_avg"]


def _mosfet(V_F=0.7, **kw):
    return lambda b: b.add_mosfet("Q", "gnd", "s", 1e-3, 1e9, True, V_F, **kw)


# 1.1 -----------------------------------------------------------------------
@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_gate_off_body_diode_drops_V_F_plus_ohmic(engine):
    b, fn = _forced(_mosfet())
    res, v = _v_s(b, fn, engine)
    assert v == pytest.approx(0.7 + I_FORCED / 1e3, abs=1e-6)
    i_body = float(np.asarray(res.i("Q_body"))[-1])
    assert v * i_body == pytest.approx(7.10, abs=1e-4)


def test_gate_off_body_diode_drop_at_the_dc_operating_point():
    b, _ = _forced(_mosfet())
    x = p.compute_dc_op(b, mask=_mask(b))
    v = float(x[b.node_id_of("s")])
    assert v == pytest.approx(0.710, abs=1e-6)


# 1.2 -----------------------------------------------------------------------
def test_gate_on_the_channel_carries_it_and_the_diode_stays_off():
    b, fn = _forced(_mosfet(), gate_on=True)
    res, v = _v_s(b, fn)
    assert v == pytest.approx(I_FORCED * 1e-3, abs=1e-6)
    assert abs(float(np.asarray(res.i("Q_body"))[-1])) < 1e-6
    assert _p_avg(b, res, "Q") == pytest.approx(0.1, rel=1e-2)
    assert _p_avg(b, res, "Q_body") < 1e-6


# 1.3 -----------------------------------------------------------------------
@pytest.mark.parametrize("V_F, expected", [(0.0, 0.010), (1.2, 1.210)])
def test_V_F_sets_the_drop(V_F, expected):
    b, fn = _forced(_mosfet(V_F))
    _, v = _v_s(b, fn)
    assert v == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("bad", [-0.1, float("nan")])
@pytest.mark.parametrize("builder", ["add_mosfet", "add_mosfet_with_body_diode"])
def test_bad_V_F_is_refused_by_name(builder, bad):
    b = p.CircuitBuilder()
    with pytest.raises(ValueError, match=r'Q.*V_F'):
        getattr(b, builder)("Q", "d", "gnd", V_F=bad)


def test_bad_V_F_is_refused_on_the_level1_devices():
    b = p.CircuitBuilder()
    with pytest.raises(ValueError, match=r'add_mosfet_level1\("M"\).*V_F'):
        b.add_mosfet_level1("M", "d", "gnd", "g", 5.0, 3.0, V_F=-0.1)
    with pytest.raises(ValueError, match=r'add_igbt_level1\("Z"\).*V_F'):
        b.add_igbt_level1("Z", "c", "gnd", "g", with_fwd=True, V_F=float("nan"))


# 1.4 -----------------------------------------------------------------------
def test_with_body_diode_uses_its_own_slope():
    b, fn = _forced(lambda b: b.add_mosfet_with_body_diode(
        "Q", "gnd", "s", V_F=0.7, g_on_diode=20.0))
    res, v = _v_s(b, fn)
    assert v == pytest.approx(1.200, abs=1e-6)
    assert _p_avg(b, res, "Q_body") == pytest.approx(12.0, rel=1e-2)


# 1.5 -----------------------------------------------------------------------
@pytest.mark.parametrize("kw, expected", [({}, 0.710), ({"V_F": 0.7}, 0.710), ({"V_F": 1.0}, 1.010)])
def test_level1_mosfet_body_diode_drop(kw, expected):
    def add(b):
        b.add_voltage_source("Vg", "g", "s", 0.0)
        b.add_mosfet_level1("M", "gnd", "s", "g", 5.0, 3.0, **kw)
    b = p.CircuitBuilder()
    b.add_current_source("I", "s", "gnd", I_FORCED)
    b.add_resistor("Rb", "s", "gnd", 1e6)
    add(b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, T_RUN, DT, engine="pwl")
    assert float(np.asarray(res.v("s"))[-1]) == pytest.approx(expected, abs=1e-6)


# 1.6 -----------------------------------------------------------------------
@pytest.mark.parametrize("kw", [{}, {"V_F": 0.7}])
def test_igbt_level1_freewheeling_diode_drop(kw):
    b = p.CircuitBuilder()
    b.add_current_source("I", "s", "gnd", I_FORCED)
    b.add_resistor("Rb", "s", "gnd", 1e6)
    b.add_voltage_source("Vg", "g", "s", 0.0)
    b.add_igbt_level1("Z", "gnd", "s", "g", with_fwd=True, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, T_RUN, DT, engine="pwl")
    assert float(np.asarray(res.v("s"))[-1]) == pytest.approx(0.710, abs=1e-6)


# 1.7 -----------------------------------------------------------------------
def _turn_on_voltage(add, diode):
    """Ramp v_SD at 1 V/ms (1 mV per step) through 1 Ohm; the ramp value
    at the first step the diode carries current."""
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("Vr", "r", "gnd", 0.0, 1.0, 0.0, 1.0, 10.0, 1e-3, 1e-3)
    b.add_resistor("Rr", "r", "s", 1.0)
    add(b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, 1e-3, 1e-6, engine="pwl", switch_fn=lambda t: _mask(b))
    t = np.asarray(res.times)
    k = int(np.argmax(np.asarray(res.i(diode)) > 1e-6))
    assert k > 0
    return t[k] / 1e-3


def test_turn_on_is_at_V_F_not_below():
    v_on = _turn_on_voltage(_mosfet(), "Q_body")
    assert 0.700 <= v_on <= 0.702

    def l1(b):
        b.add_voltage_source("Vg", "g", "s", 0.0)
        b.add_mosfet_level1("M", "gnd", "s", "g", 5.0, 3.0)
    v_on = _turn_on_voltage(l1, "M_body")
    assert 0.700 <= v_on <= 0.702  # was 0.5 V


# 1.8 -----------------------------------------------------------------------
T_SW, DUTY = 1e-5, 0.5


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    b.add_switch("QH", "in", "sw", 1e3, 1e-9)
    b.add_mosfet("QL", "sw", "gnd", 1e-3, 1e9, True, 0.7)
    b.add_inductor("L", "sw", "out", 100e-6)
    b.add_capacitor("C", "out", "gnd", 100e-6)
    b.add_resistor("R", "out", "gnd", 1.2)
    n, k = b.graph.num_switches, b.switch_index_of("QH")
    return b, p.make_pwm_switch_fn(1 / T_SW, DUTY, k, n)


@pytest.mark.parametrize("engine", ["pwl", "trbdf2", "dsed"])
def test_buck_freewheeling_through_a_gate_off_mosfet(engine):
    b, fn = _buck()
    kw = {"dt": T_SW / 200} if engine == "pwl" else {}
    if engine == "dsed":
        kw["integrator"] = "rk45"  # 'auto' is being changed elsewhere
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, 400 * T_SW, engine=engine, switch_fn=fn, **kw)
    t = np.asarray(res.times)
    v_out = p.time_average(t, np.asarray(res.v("out")), t_from=300 * T_SW)
    assert v_out == pytest.approx(11.640, abs=0.005)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = {e["name"]: e for e in p.device_loss_summary(b, res, t_from=300 * T_SW)}
    # D = 0.5: the diode carries I_L half the time, so
    # P = (V_F*I_L + I_L^2/g_on)/2 = 3.442 W at I_L = 9.70 A (the 3.479 W
    # first quoted for this was the 0-400-period average, start-up included).
    t, i_l = t[t >= 300 * T_SW], np.asarray(res.i("L"))[t >= 300 * T_SW]
    i_avg = np.trapezoid(i_l, t) / (t[-1] - t[0])
    assert rows["QL_body"]["P_avg"] == pytest.approx(0.5 * (0.7 * i_avg + i_avg**2 / 1e3), rel=2e-3)
    assert rows["QL_body"]["P_avg"] == pytest.approx(3.442, abs=0.01)


# 1.9 -----------------------------------------------------------------------
def test_h_bridge_of_composites_settles_on_dsed():
    """Zero-width band (V_th = 0 < V_f): every dead time hands the load
    current to two body diodes that turn on from zero current. dsed
    settles (measured 2402 steps over 200 periods); a "did not settle"
    error here would reopen Open Question 2 (a V_th keyword)."""
    T, dead = 1e-5, 2e-7
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc", "gnd", 48.0)
    for leg in "ab":
        b.add_mosfet(f"Q{leg}H", "dc", leg, 1e-3, 1e9, True, 0.7)
        b.add_mosfet(f"Q{leg}L", leg, "gnd", 1e-3, 1e9, True, 0.7)
    b.add_resistor("R", "a", "m", 2.0)
    b.add_inductor("L", "m", "b", 1e-3)
    n = b.graph.num_switches
    idx = {k: b.switch_index_of(k) for k in ("QaH", "QaL", "QbH", "QbL")}

    def fn(t):
        ph, m = (t % T) / T, p.SwitchStateMask(n)
        on1, on2 = dead / T < ph < 0.5 - dead / T, 0.5 + dead / T < ph < 1 - dead / T
        for k, on in (("QaH", on1), ("QbL", on1), ("QaL", on2), ("QbH", on2)):
            m.set(idx[k], on)
        return m

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, 200 * T, engine="dsed", integrator="rk45", switch_fn=fn)
    # the body diodes did carry the dead-time current, at a 0.7 V drop
    assert float(np.max(np.asarray(res.i("QaL_body")))) > 1e-3
    assert float(np.min(np.asarray(res.v("a")))) < -0.69


# 1.10 ----------------------------------------------------------------------
def test_yaml_V_F_is_a_drop():
    y = """
circuit:
  devices:
    - {type: current_source, name: I, from: s, to: gnd, I: 10.0}
    - {type: resistor, name: Rb, from: s, to: gnd, R: 1.0e6}
    - {type: mosfet_with_body_diode, name: Q, drain: gnd, source: s, V_F: 0.7}
simulation: {t_start: 0.0, t_end: 1.0e-4, dt: 1.0e-6}
"""
    b = p.load_yaml_string(y).builder
    _, v = _v_s(b, lambda t: _mask(b))
    assert v == pytest.approx(0.710, abs=1e-6)

    y = """
circuit:
  devices:
    - {type: current_source, name: I, from: s, to: gnd, I: 10.0}
    - {type: resistor, name: Rb, from: s, to: gnd, R: 1.0e6}
    - {type: voltage_source, name: Vg, from: g, to: s, V: 0.0}
    - {type: mosfet_level1, name: M, drain: gnd, source: s, gate: g, K: 5.0, V_T: 3.0, V_F: 1.0}
simulation: {t_start: 0.0, t_end: 1.0e-4, dt: 1.0e-6}
"""
    b = p.load_yaml_string(y).builder
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, T_RUN, DT, engine="pwl")
    assert float(np.asarray(res.v("s"))[-1]) == pytest.approx(1.010, abs=1e-6)
