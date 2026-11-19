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
    # the Lauritzen diode keeps a Newton history whose stage takes no
    # arbitrary base yet (the nonlinear capacitor, the saturable inductor and
    # external devices take the stage form: tests below)
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_resistor("R", "a", "m", 1.0)
    b.add_lauritzen_diode("Dl", "m", "gnd")
    with pytest.raises(p.UnsupportedError) as e:
        p.simulate(b, t_end=1e-5, engine="trbdf2", integrator="esdirk43")
    assert e.value.feature == "integrator=esdirk43"
    assert e.value.device == "Dl"
    assert e.value.kind == "LauritzenDiode"


# ---- stateful kinds through the stage form (βh, ψ) ------------------------
# A lossless LC whose inductor saturates (atan law): smooth and nonlinear.
# Truth: scipy DOP853 at rtol 1e-13 on L(i)·di/dt = V − v, C·dv/dt = i.
SL0, SLR, SIS = 100e-6, 20e-6, 0.1


def _sat_lc():
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "a", "gnd", 0.0, V, TD, 1.0, 0.0, 0.0, 0.0)
    b.add_saturable_inductor("L", "a", "b", SL0, SIS, SLR)
    b.add_capacitor("C", "b", "gnd", C)
    return b


def _sat_truth(t):
    from scipy.integrate import solve_ivp

    def f(_, y):
        return [(V - y[1]) / (SLR + (SL0 - SLR) / (1.0 + (y[0] / SIS) ** 2)), y[0] / C]

    s = solve_ivp(f, (0.0, float(t[-1])), [0.0, 0.0], method="DOP853",
                  rtol=1e-13, atol=1e-16, dense_output=True)
    return s.sol(t)


def _sat_end_err(r):
    i, v = _sat_truth(np.array([0.0, r.times[-1] - TD]))[:, -1]
    return max(abs(r.i("L")[-1] - i) / 0.5, abs(r.v("b")[-1] - v) / 20.0)


def test_saturable_lc_is_fourth_order_against_an_ode_truth():
    # fixed ceiling, rtol loose: every step at T0/N. Measured end errors
    # after 4 periods: 2.03e-3, 8.06e-5, 4.72e-6, 2.94e-7 at T0/20../160
    # (ratios 25.2, 17.1, 16.1: order 4)
    e = [_sat_end_err(_lc_run_sat(n)) for n in (40, 80)]
    assert e[0] / e[1] > 2.0 ** 3.5, e


def _lc_run_sat(n, **kw):
    return p.simulate(_sat_lc(), engine="trbdf2", t_end=TD + 4 * T0,
                      integrator="esdirk43", dt=T0 / n, rtol=1.0, atol=1.0, **kw)


def test_saturable_lc_agrees_with_trbdf2_and_the_truth_at_tight_tolerance():
    runs = {integ: p.simulate(_sat_lc(), engine="trbdf2", t_end=TD + 4 * T0,
                              integrator=integ, rtol=1e-9, atol=1e-12)
            for integ in (None, "esdirk43")}
    # measured: 4.8e-8 (esdirk43, 1789 steps) and 4.8e-6 (TR-BDF2, 23704);
    # esdirk43's floor ~2e-8..8e-8 at every rtol is the landed source
    # step's 7e-14 s sliver (the linear LC shows it too), not the stages
    assert _sat_end_err(runs["esdirk43"]) <= 2e-7
    assert _sat_end_err(runs[None]) <= 2e-5
    # and fewer steps for the same accuracy class
    assert len(runs["esdirk43"].times) < len(runs[None].times)


C0, V0, M, I0 = 1e-9, 0.7, 0.45, 1e-3


def _cap_charge():
    # a junction-law capacitor charged by a constant current from 1 us: its
    # charge is I·t, so v(t) is closed form (Rleak 1e15: below 1e-10)
    b = p.CircuitBuilder()
    b.add_pulse_current_source("I", "c", "gnd", 0.0, I0, 1e-6, 1.0, 0.0, 0.0, 0.0)
    b.add_resistor("Rleak", "c", "gnd", 1e15)
    b.add_nonlinear_capacitor("Cn", "c", "gnd", C0, V0, M)
    return b


def test_nonlinear_capacitor_charge_is_the_closed_form():
    r = p.simulate(_cap_charge(), engine="trbdf2", t_end=2e-5,
                   integrator="esdirk43", rtol=1e-8, atol=1e-11)
    t = np.asarray(r.times)
    q = I0 * np.maximum(t - 1e-6, 0.0)
    v = V0 * ((1.0 + q * (1.0 - M) / (C0 * V0)) ** (1.0 / (1.0 - M)) - 1.0)
    # measured 6.9e-8 of max v, equal to TR-BDF2's (the source-step sliver)
    assert np.max(np.abs(np.asarray(r.v("c")) - v)) <= 3e-7 * np.max(v)


@pytest.mark.parametrize("name", ["boost_mosfet_coss_saturable", "chopper_igbt_vdmos"])
def test_device_circuits_agree_with_trbdf2_at_tight_tolerance(name):
    # Level-1 MOSFET + Coss + saturable inductor + Shockley rectifier; IGBT
    # + the VDMOS C_gd law + saturable freewheel path. Reference: TR-BDF2 at
    # rtol 1e-9 under a 2.5 ns ceiling (at its own steps TR-BDF2 is 8.3e-5 /
    # 3.5e-5 off it; esdirk43 measured 6.3e-6 / 7.2e-6, and moves toward it
    # under a ceiling: 1.6e-6 / 3.4e-7 at 2.5 ns)
    from . import test_device_path_equivalence as Q
    fac = getattr(Q, name)
    te = 4e-5
    ref = p.simulate(fac(), engine="trbdf2", t_end=te, rtol=1e-9, atol=1e-12, dt=2.5e-9)
    run = p.simulate(fac(), engine="trbdf2", t_end=te, integrator="esdirk43",
                     rtol=1e-9, atol=1e-12)
    for k in range(np.asarray(ref.states).shape[1]):
        ya, ye = np.asarray(ref.states)[:, k], np.asarray(run.states)[:, k]
        scale = max(np.max(np.abs(ya)), 1e-9)
        assert abs(ye[-1] - ya[-1]) <= 2.5e-5 * scale, k


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


def _stateful_dcm_buck():
    # the DCM buck with a junction Coss on the switch node and a saturable
    # inductor: diode events change the mask, so each is followed by the
    # forced SDIRK(2,2) step, here with the q rows (ed_sdirk_base)
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 24.0)
    b.add_switch("S", "in", "sw", 1e3, 1e-6)
    b.add_diode("D", "gnd", "sw", 1e3, 1e-6, 0.0)
    b.add_nonlinear_capacitor("Coss", "sw", "gnd", 1e-9, 0.7, 0.45)
    b.add_saturable_inductor("L", "sw", "out", 10e-6, 1.0, 4e-6)
    b.add_capacitor("C", "out", "gnd", 10e-6)
    b.add_resistor("R", "out", "gnd", 20.0)
    fn = p.make_pwm_switch_fn(100e3, 0.3, b.switch_index_of("S"), b.graph.num_switches)
    return b, fn


def test_stateful_kinds_through_diode_events_agree_with_a_fine_trbdf2():
    runs = {}
    for integ, dt in ((None, 2e-9), ("esdirk43", None)):
        b, fn = _stateful_dcm_buck()
        runs[integ] = p.simulate(b, t_end=1e-4, engine="trbdf2", switch_fn=fn,
                                 integrator=integ, rtol=1e-9, atol=1e-12, dt=dt)
    ref, run = runs[None], runs["esdirk43"]
    s0, s1 = ref._trbdf2_stats, run._trbdf2_stats
    assert (s1["n_diode_events"], s1["n_gate_events"]) == (
        s0["n_diode_events"], s0["n_gate_events"])
    assert s1["n_sdirk_steps"] >= s1["n_diode_events"] > 0
    dv = abs(run.v("out")[-1] - ref.v("out")[-1]) / 24.0
    di = abs(run.i("L")[-1] - ref.i("L")[-1]) / np.max(np.abs(ref.i("L")))
    # measured 3.1e-6 / 3.2e-6 against this reference, whose own error is
    # ~4e-6 (a 1 ns ceiling moves it 5.9e-6 / 4.1e-6); TR-BDF2 at its own
    # steps: 3.2e-5 / 3.6e-5 from the 1 ns run
    assert dv <= 2e-5 and di <= 2e-5
