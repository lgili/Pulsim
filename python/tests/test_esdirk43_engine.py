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


def _loss_buck():
    # the buck of the recorder tests with E = k·v·i tables on S1 and D1 and
    # S1's junction on a thermal RC (synthetic data)
    from .test_device_losses_variable_step import KOFF, KON, _kvi
    from .test_recorder_device_readings import _buck
    b, fn, t_end, _, _ = _buck()
    b.add_voltage_source("Vamb", "amb", "gnd", 25.0)
    b.add_resistor("Rth", "tj", "amb", 5.0)
    b.add_capacitor("Cth", "tj", "amb", 1e-4)
    b.set_device_losses("S1", E_on=_kvi(KON), E_off=_kvi(KOFF), junction="tj")
    b.set_device_losses("D1", E_on=_kvi(KON), E_off=_kvi(KOFF), Tj=25.0)
    return b, fn, t_end, ["S1", "D1"]


def _loss_vsi():
    from .test_device_losses_kernel import _switch_names, _vsi
    b, fn = _vsi()
    tab = p.LossTable(v_axis=[0.0, 800.0], i_axis=[0.0, 60.0], tj_axis=[25.0, 150.0],
                      energy=np.array([[[0.0, 0.0], [0.0, 0.0]],
                                       [[0.0, 0.0], [2e-3, 3e-3]]]))
    names = _switch_names(b)
    for n in names:
        b.set_device_losses(n, Tj=125.0, E_on=tab, E_off=tab)
    return b, fn, 4e-3, names


@pytest.mark.parametrize("case", [_loss_buck, _loss_vsi])
def test_in_kernel_losses_agree_with_trbdf2_at_tight_tolerance(case):
    # integrator='esdirk43' books the same events at the same sites and
    # integrates conduction on its stage weights b: agreement with the
    # TR-BDF2 run at rtol 1e-8 (measured: worst 2.8e-6 of the device's
    # largest energy; rtol 1e-9 grinds TR-BDF2 itself on the buck's
    # capacitance-free switch node). Closed form: test_device_losses_variable_step
    runs = {}
    for integ in (None, "esdirk43"):
        b, fn, t_end, names = case()
        runs[integ] = p.simulate(b, t_end=t_end, engine="trbdf2", switch_fn=fn,
                                 integrator=integ, rtol=1e-8, atol=1e-11)
    for n in names:
        a, e = runs[None].device_energy(n), runs["esdirk43"].device_energy(n)
        assert (e["n_on"], e["n_off"]) == (a["n_on"], a["n_off"]), n
        for k in ("E_on", "E_off", "E_cond_electrical"):
            scale = max(abs(a["E_on"]), abs(a["E_off"]), abs(a["E_cond_electrical"]))
            assert abs(e[k] - a[k]) <= 2e-5 * scale, (n, k, e[k], a[k])
    # the ledger and the junction: test_device_losses_variable_step


def test_in_loop_recorder_runs_and_agrees_with_trbdf2():
    rec = {}
    for integ in (None, "esdirk43"):
        b, fn = _buck_dcm()
        r = p.simulate(b, t_end=1e-4, engine="trbdf2", switch_fn=fn, integrator=integ,
                       rtol=1e-8, atol=1e-11,
                       recorders=[p.Recorder("r", ["v(out)", "i(L)"], period=1e-5)])
        assert r.recorder_feed["mode"] == "in-loop"
        assert r.recorder_feed["n_dropped"] == 0
        rec[integ] = r.recorded["r"]
    for q, scale in (("v(out)", 24.0), ("i(L)", 24.0 / 20.0)):
        a, e = rec[None].periods[q], rec["esdirk43"].periods[q]
        assert a["avg"].size == e["avg"].size == 10
        np.testing.assert_allclose(e["avg"], a["avg"], rtol=0, atol=1e-3 * scale, err_msg=q)
        np.testing.assert_allclose(e["rms"], a["rms"], rtol=0, atol=1e-3 * scale, err_msg=q)
