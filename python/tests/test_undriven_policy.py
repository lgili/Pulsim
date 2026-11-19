"""simulate(undriven=...): ONE policy for controlled switches nothing drives
(add-kernel-simulation-object task 11.3, DECISIONS 26). None keeps each
engine's own behaviour (pinned here and in test_cross_cutting_on_every_engine);
'refuse' / 'open' / 'closed' behave the same on pwl, trbdf2, the exact engine
and through engine='auto'."""

import warnings

import numpy as np
import pytest

import pulsim as p
from pulsim import _plan

ENGINES = {"pwl": dict(engine="pwl", dt=1e-8), "trbdf2": dict(engine="trbdf2"),
           "exact": dict(engine="exact"), "auto": dict(engine="auto")}
USED = {"pwl": "pwl", "trbdf2": "trbdf2", "exact": "dsed", "auto": "trbdf2"}
T_END = 5e-6


def _circuit(sw=True, r_sw=1.0):
    # V - S(g_on = 1 S) - out, R || C to ground: tau = 0.5 us closed
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 1.0)
    if sw:
        b.add_switch("S", "in", "out", 1.0, 1e-9)
    else:
        b.add_resistor("Rs", "in", "out", r_sw)
    b.add_resistor("R", "out", "gnd", 1.0)
    b.add_capacitor("C", "out", "gnd", 1e-6)
    return b


def _run(b, engine, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = p.simulate(b, T_END, **ENGINES[engine], **kw)
    return r, [str(x.message) for x in w]


def _v_on(r, t):
    return np.interp(t, np.asarray(r.times), np.asarray(r.v("out")))


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("policy,r_sw", [("open", 1e9), ("closed", 1.0)])
def test_open_and_closed_equal_the_circuit_with_the_switch_replaced(engine, policy, r_sw):
    r, w = _run(_circuit(), engine, undriven=policy)
    assert r.engine_used == USED[engine]
    assert not any("switch_fn" in m for m in w)          # one diagnostic, no warning
    assert r.diagnostics["undriven"] == {"policy": policy, "switches": ["S"]}
    twin, _ = _run(_circuit(sw=False, r_sw=r_sw), engine)
    t = np.linspace(0.0, T_END, 41)
    np.testing.assert_allclose(_v_on(r, t), _v_on(twin, t), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("policy,g", [("open", 1e-9), ("closed", 1.0)])
def test_open_and_closed_agree_across_the_engines(engine, policy, g):
    # every engine against the closed form at its own samples: V g/(g+1) (1 - e^(-t/tau)),
    # tau = C/(g+1); tolerance = the engines' own agreement here (pwl: the trapezoid's
    # first step from the source edge lags by dt/2, ~1 % of V at dt = tau/50)
    r, _ = _run(_circuit(), engine, undriven=policy)
    t = np.asarray(r.times)
    exact = g / (g + 1.0) * (1.0 - np.exp(-t * (g + 1.0) / 1e-6))
    np.testing.assert_allclose(np.asarray(r.v("out")), exact, rtol=0,
                               atol=(1.2e-2 if engine == "pwl" else 1e-4) * g / (g + 1.0))


def _two_switches():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 1.0)
    b.add_switch("S_hi", "in", "sw", 1.0, 1e-9)
    b.add_switch("S_lo", "sw", "gnd", 1.0, 1e-9)
    b.add_resistor("R", "sw", "gnd", 10.0)
    return b


@pytest.mark.parametrize("engine", [*ENGINES, "auto_dt"])
def test_refuse_names_every_undriven_switch(engine):
    kw = dict(engine="auto", dt=1e-8) if engine == "auto_dt" else ENGINES[engine]
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(_two_switches(), T_END, undriven="refuse", **kw)
    assert ei.value.feature == "switch_fn" and ei.value.device == "S_hi"
    assert "S_hi, S_lo" in str(ei.value) and "undriven='open'" in str(ei.value)


@pytest.mark.parametrize("engine", ["pwl", "exact"])
def test_none_keeps_the_engines_own_all_closed_with_a_warning(engine):
    r, w = _run(_circuit(), engine)
    assert sum("no switch_fn was given" in m for m in w) == 1
    assert "undriven" not in (getattr(r, "diagnostics", None) or {})
    twin, _ = _run(_circuit(sw=False, r_sw=1.0), engine)
    np.testing.assert_allclose(np.asarray(r.v("out"))[-1], np.asarray(twin.v("out"))[-1],
                               rtol=1e-9)


def test_none_keeps_trbdf2_and_auto_refusals():
    with pytest.raises(p.UnsupportedError, match="no switch_fn"):
        p.simulate(_circuit(), T_END, engine="trbdf2")
    with pytest.raises(ValueError, match="cannot use the variable-step engine"):
        p.simulate(_circuit(), T_END)
    r, w = _run(_circuit(), "pwl")
    r2 = p.simulate(_circuit(), T_END, dt=1e-8)       # 'auto' with dt: pwl, closed
    assert r2.engine_used == "pwl"
    np.testing.assert_array_equal(np.asarray(r2.v("out")), np.asarray(r.v("out")))


def test_a_driven_or_diode_only_circuit_takes_no_policy():
    fn = p.make_pwm_switch_fn(frequency=1e6, duty=0.5, switch_idx=0, num_switches=1)
    r = p.simulate(_circuit(), T_END, engine="exact", switch_fn=fn, undriven="refuse")
    assert "undriven" not in (getattr(r, "diagnostics", None) or {})
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 1.0)
    b.add_diode("D", "in", "out", 1e3, 1e-9)
    b.add_resistor("R", "out", "gnd", 1.0)
    for engine in ENGINES:
        r = p.simulate(b, T_END, **ENGINES[engine], undriven="refuse")
        assert "undriven" not in (getattr(r, "diagnostics", None) or {})


def test_a_bad_value_is_refused_by_name():
    with pytest.raises(p.OptionsError, match="undriven") as ei:
        p.simulate(_circuit(), T_END, dt=1e-8, undriven="off")
    assert ei.value.option == "undriven"


def test_the_library_default_is_read_only_when_the_call_leaves_it_none(monkeypatch):
    monkeypatch.setattr(_plan, "DEFAULT_UNDRIVEN", "refuse")   # the release value
    with pytest.raises(p.UnsupportedError, match="have no driver: S"):
        p.simulate(_circuit(), T_END, engine="pwl", dt=1e-8)
    r = p.simulate(_circuit(), T_END, engine="pwl", dt=1e-8, undriven="closed")
    assert r.diagnostics["undriven"]["policy"] == "closed"


def test_the_plan_carries_the_one_diagnostic():
    import inspect
    bound = inspect.signature(p.simulate).bind(_circuit(), T_END, 1e-8, undriven="open")
    bound.apply_defaults()
    plan = p._plan_run(**bound.arguments)
    assert plan.undriven == ("open", ("S",))
    (d,) = [d for d in plan.diagnostics if d.code == "undriven"]
    assert "OPEN: S" in d.message
