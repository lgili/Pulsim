"""integrator='esdirk43' on engine='trbdf2' (add-high-order-stiff-integrator,
task group 3): the variable-step engine takes its steps with ESDIRK4(3)6L[2]SA
through the same phases; refused by name where it cannot run."""

import numpy as np
import pytest

import pulsim as p

L, C, V = 100e-6, 100e-9, 10.0
W = 1.0 / np.sqrt(L * C)
T0 = 2.0 * np.pi / W
Z = np.sqrt(L / C)


TD = 0.5 * T0   # the step lands as a source breakpoint: a consistent start


def _lc():
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "a", "gnd", 0.0, V, TD, 1.0, 0.0, 0.0, 0.0)
    b.add_inductor("L", "a", "b", L)
    b.add_capacitor("C", "b", "gnd", C)
    return b


def _lc_run(t_end, **kw):
    return p.simulate(_lc(), engine="trbdf2", t_end=TD + t_end, **kw)


def _lc_err(r):
    t = np.maximum(np.asarray(r.times) - TD, 0.0)
    ev = np.asarray(r.v("b")) - V * (1.0 - np.cos(W * t))
    ei = (np.asarray(r.i("L")) - V / Z * np.sin(W * t)) * Z
    return np.hypot(ev, ei)


def test_lc_is_fourth_order_through_the_engine():
    # fixed ceiling, rtol loose: every step sits at T0/N; error after 4
    # periods (10 periods: 5.17e-3, 3.24e-4, 2.05e-5 V at N = 20, 40, 80; the
    # test body's relative 5.16e-4, 3.24e-5, 2.02e-6 times 10 V)
    e = [_lc_err(_lc_run(4 * T0, integrator="esdirk43", dt=T0 / n,
                         rtol=1.0, atol=1.0))[-1] for n in (20, 40)]
    assert e[0] / e[1] > 12.0, e   # 16 for order 4


@pytest.mark.parametrize("engine,dt", [("pwl", 1e-7), ("exact", None)])
def test_refused_by_name_off_the_variable_step_engine(engine, dt):
    with pytest.raises(p.UnsupportedError) as e:
        p.simulate(_lc(), t_end=1e-6, dt=dt, engine=engine, integrator="esdirk43")
    assert e.value.feature == "integrator"


def test_trbdf2_refuses_another_engines_integrator_by_name():
    with pytest.raises(p.UnsupportedError) as e:
        p.simulate(_lc(), t_end=1e-6, engine="trbdf2", integrator="rk45")
    assert (e.value.engine, e.value.feature) == ("trbdf2", "integrator")


def test_a_stateful_newton_history_is_refused_by_device_and_kind():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_resistor("R", "a", "m", 1.0)
    b.add_saturable_inductor("Ls", "m", "gnd", 1e-3, 20.0)
    with pytest.raises(p.UnsupportedError) as e:
        p.simulate(b, t_end=1e-5, engine="trbdf2", integrator="esdirk43")
    assert e.value.feature == "integrator=esdirk43"
    assert e.value.device == "Ls"
    assert e.value.kind == "SaturableInductor"


def _buck_dcm():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 24.0)
    b.add_switch("S", "in", "sw", 1e3, 1e-6)
    b.add_diode("D", "gnd", "sw", 1e3, 1e-6, 0.0)
    b.add_inductor("L", "sw", "out", 10e-6)
    b.add_capacitor("C", "out", "gnd", 10e-6)
    b.add_resistor("R", "out", "gnd", 20.0)
    fn = p.make_pwm_switch_fn(100e3, 0.3, b.switch_index_of("S"), b.graph.num_switches)
    return b, fn


def test_dcm_buck_events_and_waveform_agree_with_trbdf2():
    runs = {}
    for integ in (None, "esdirk43"):
        b, fn = _buck_dcm()
        r = p.simulate(b, t_end=1e-4, engine="trbdf2", switch_fn=fn,
                       integrator=integ, rtol=1e-8, atol=1e-11)
        runs[integ] = r
    s0, s1 = runs[None]._trbdf2_stats, runs["esdirk43"]._trbdf2_stats
    assert (s1["n_diode_events"], s1["n_gate_events"]) == (
        s0["n_diode_events"], s0["n_gate_events"])
    t = np.linspace(1e-5, 1e-4, 200)
    for sig in ("out", "sw"):
        a = np.interp(t, runs[None].times, runs[None].v(sig))
        e = np.interp(t, runs["esdirk43"].times, runs["esdirk43"].v(sig))
        assert np.median(np.abs(a - e)) < 1e-3 * 24.0, sig
    assert np.all(np.diff(runs["esdirk43"].times) >= 0)
