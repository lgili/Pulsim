"""add-branch-form-ideal-switch task 6.4 (last clause): the impulsive
re-initialisation energy is a loss category of its own, booked to the switch
that entered the dependent mode, on every engine that records it (pwl,
trbdf2; dsed refuses the branch form today, its records would be read the
same way through `impulse_losses`).

Energy balance: source energy = resistor + switch conduction + impulsive
energy + change of stored energy. The step that carries the pulse is a
discrete artefact (the current jumps inside it; a linear interpolation of
its samples has no circuit meaning), so the balance is taken around it: the
record's dW is the stored energy that step removes. What is left is the
quadrature error of the samples, O(h^2).
"""
import warnings

import numpy as np
import pytest

import pulsim as p
from pulsim.losses import device_loss_summary, impulse_losses

C1, C2, V1, V2 = 1e-6, 3e-6, 10.0, 2.0


def _caploop(engine, h=1e-7, steps=50):
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "x", "gnd", C1, V1)
    b.add_capacitor("C2", "y", "gnd", C2, V2)
    b.add_ideal_switch("S", "x", "y", 0.0, 1e-9)
    m = p.SwitchStateMask(1)
    m.set(0, True)
    kw = dict(dt=h, discontinuity_restart=True) if engine == "pwl" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=steps * h + 1e-3 * h, engine=engine,
                       switch_fn=lambda t: m, auto_regularize=False, **kw)
    return b, r


def _cutset(engine, N=20):
    tau = 1e-3
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S", "vin", "a", 0.0, 0.0)
    b.add_resistor("R", "a", "b", 1.0)
    b.add_inductor("L", "b", "gnd", 1e-3)
    k = b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(1)
        m.set(k, t < 5e-3 - 1e-9)
        return m
    kw = dict(dt=tau / N, discontinuity_restart=True) if engine == "pwl" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=8e-3, engine=engine, switch_fn=fn, **kw)
    return b, r


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_capacitor_loop_energy_is_booked_to_the_switch_and_closes_the_balance(engine):
    b, r = _caploop(engine)
    (rec,) = r.reinit_records
    s = {e["name"]: e for e in device_loss_summary(b, r)}
    assert s["S"]["E_impulse_total"] == pytest.approx(24e-6, rel=1e-9)
    assert s["S"]["n_impulses"] == 1
    assert s["S"]["P_impulse_avg"] == pytest.approx(
        24e-6 / (s["S"]["t_window_end"] - s["S"]["t_window_start"]), rel=1e-9)
    v1, v2 = r.v("x")[-1], r.v("y")[-1]
    dW = 0.5 * C1 * V1**2 + 0.5 * C2 * V2**2 - 0.5 * C1 * v1**2 - 0.5 * C2 * v2**2
    assert s["S"]["E_impulse_total"] + s["S"]["E_total"] == pytest.approx(dW, rel=1e-10)
    imp = impulse_losses(b, r)
    assert list(imp) == ["S"] and imp["S"]["records"][0] is not None


def _caploop_closing(h, k_close=10, steps=40, **kw):
    # the switch closes at a gate edge mid-run, after leaking g_off while open
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "x", "gnd", C1, V1)
    b.add_capacitor("C2", "y", "gnd", C2, V2)
    b.add_ideal_switch("S", "x", "y", 0.0, 1e-9)
    t_c = (k_close + 0.5) * h

    def fn(t):
        m = p.SwitchStateMask(1)
        m.set(0, t >= t_c)
        return m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=steps * h + 1e-3 * h, engine="pwl", dt=h, switch_fn=fn,
                       discontinuity_restart=True, auto_regularize=False, **kw)
    return b, r


@pytest.mark.parametrize("h", [1e-8, 1e-7, 3e-7, 1e-6, 1e-5])
def test_capacitor_loop_jump_and_conduction_partition_the_pulse_step_at_any_dt(h):
    # DECISIONS #42: the pulse step starts from its post-jump end sample, so the
    # impulsive energy and the conduction integral (the g_off leak before the
    # edge) add up to the stored-energy drop over the whole run, pulse step
    # included, at any step
    b, r = _caploop_closing(h)
    (rec,) = r.reinit_records
    s = {e["name"]: e for e in device_loss_summary(b, r)}
    v1, v2 = r.v("x"), r.v("y")
    W = 0.5 * C1 * np.asarray(v1) ** 2 + 0.5 * C2 * np.asarray(v2) ** 2
    dW = 0.5 * C1 * V1**2 + 0.5 * C2 * V2**2 - W[-1]
    assert s["S"]["E_impulse_total"] == pytest.approx(rec.dW, rel=1e-15)
    assert s["S"]["E_impulse_total"] + s["S"]["E_total"] == pytest.approx(dW, rel=1e-12)


def test_the_pulse_step_conducts_its_end_power_only():
    # the cut-set: R's conduction over the whole run equals the sum of the
    # windows around the pulse step plus that step's end power times h (the
    # current is zero after the jump: nothing), not a line from the
    # pre-jump 9.93 A (R*i^2*h/2 = 2.5e-3 J at tau/20 counted on top of dW)
    b, r = _cutset("pwl")
    t, i = np.asarray(r.times), np.asarray(r.i("R"))
    (rec,) = r.reinit_records
    k = rec.sample if rec.sample > 0 else int(np.searchsorted(t, rec.t, side="right"))
    a, c = t[k - 1], t[k]

    def E(lo=None, hi=None):
        return {e["name"]: e for e in device_loss_summary(b, r, t_from=lo, t_to=hi)}["R"]["E_total"]
    step = 1.0 * i[k] ** 2 * (c - a)
    assert E() == pytest.approx(E(t[0], a) + E(c, t[-1]) + step, rel=1e-12)


def _balance(b, r):
    t, i = np.asarray(r.times), np.asarray(r.i("S"))
    (rec,) = r.reinit_records
    k = rec.sample if rec.sample > 0 else int(np.searchsorted(t, rec.t, side="right"))
    a, c = t[k - 1], t[k]                    # the step that carries the pulse

    def part(lo, hi):
        s = {e["name"]: e for e in device_loss_summary(b, r, t_from=lo, t_to=hi)}
        m = (t >= lo) & (t <= hi)
        return float(np.trapezoid(10.0 * i[m], t[m])), s["R"]["E_total"] + s["S"]["E_total"]

    src1, diss1 = part(t[0], a)
    src2, diss2 = part(c, t[-1])
    E_imp = impulse_losses(b, r)["S"]["E_impulse_total"]
    assert E_imp == pytest.approx(0.5 * 1e-3 * r.i("L")[k - 1] ** 2, rel=1e-9)
    # i_L = 0 at the start and at the end: no stored-energy change overall
    return (src1 + src2 - diss1 - diss2 - E_imp) / (src1 + src2)


def test_cutset_balance_on_pwl_closes_to_second_order():
    e20, e40 = (_balance(*_cutset("pwl", N)) for N in (20, 40))
    # measured -2.57e-5 / -6.48e-6
    assert abs(e20) <= 1.3e-4   # measured -2.57e-5
    assert abs(e20 / e40) >= 3.5


def test_cutset_balance_on_trbdf2():
    assert abs(_balance(*_cutset("trbdf2"))) <= 5e-6    # measured -3.2e-7


def test_the_impulse_counts_in_the_window_its_step_starts_in():
    b, r = _cutset("pwl")
    t_rec = r.reinit_records[0].t
    before = {e["name"]: e for e in device_loss_summary(b, r, t_to=t_rec)}
    after = {e["name"]: e for e in device_loss_summary(b, r, t_from=t_rec)}
    assert before["S"]["n_impulses"] == 0 and before["S"]["E_impulse_total"] == 0.0
    assert after["S"]["n_impulses"] == 1


def test_no_records_the_ideal_element_reports_zero_impulse_explicitly():
    # a run without re-initialisation: the branch-form element still carries the
    # category (0 J, not a missing key); every other entry is as it was
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_ideal_switch("S", "a", "k", 0.0, 0.0)
    b.add_resistor("R", "k", "gnd", 1.0)
    m = p.SwitchStateMask(1)
    m.set(0, True)
    r = p.simulate(b, t_end=1e-5, dt=1e-6, engine="pwl", switch_fn=lambda t: m)
    assert len(r.reinit_records) == 0
    by = {e["name"]: e for e in device_loss_summary(b, r)}
    assert by["S"]["E_impulse_total"] == 0.0 and by["S"]["n_impulses"] == 0
    assert by["S"]["P_impulse_avg"] == 0.0
    assert by["S"]["P_avg"] == pytest.approx(0.0, abs=1e-12)   # ideal: no conduction loss
    assert "E_impulse_total" not in by["R"]


def test_thermal_summary_reports_ideal_elements_with_their_impulse():
    # the capacitor loop: 6 uJ booked to the ideal switch S, no conduction loss;
    # the thermal summary keeps S (it used to drop it: g_on = inf) and heats it
    # with the impulsive energy as a constant offset, as P_sw_avg
    from pulsim.thermal import FosterStage, device_thermal_summary
    b, r = _caploop("pwl")
    loss = {e["name"]: e for e in device_loss_summary(b, r)}["S"]
    th = device_thermal_summary(
        b, r, thermal_specs={"S": {"stages": [FosterStage(R_th_K_per_W=1.0, tau_s=1e-6)]}})
    assert [e["name"] for e in th] == ["S"]
    e = th[0]
    assert e["form"] == "branch"
    assert e["P_cond_avg"] == pytest.approx(loss["P_avg"], rel=1e-9, abs=1e-15)   # thermal E_cond vs loss summary: two summation paths
    assert e["P_impulse_avg"] == loss["P_impulse_avg"] > 0
    assert e["P_total_avg"] == pytest.approx(e["P_cond_avg"] + e["P_impulse_avg"], rel=1e-12)


def test_a_loss_port_on_an_ideal_switch_says_its_impulse_is_not_in_the_junction():
    # the capacitor loop with a loss port on S (a thermal RC on its own nodes)
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "x", "gnd", C1, V1)
    b.add_capacitor("C2", "y", "gnd", C2, V2)
    b.add_ideal_switch("S", "x", "y", 0.0, 1e-9)
    b.add_voltage_source("Vamb", "amb", "gnd", 25.0)
    b.add_resistor("Rth", "tj", "amb", 2.0)
    b.add_capacitor("Cth", "tj", "amb", 1e-3)
    t = p.LossTable(v_axis=[10.0], i_axis=[1.0], tj_axis=[25.0], energy=np.full((1, 1, 1), 1e-9))
    m = p.SwitchStateMask(b.graph.num_switches)
    m.set(b.switch_index_of("S"), True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b.set_device_losses("S", junction="tj", E_on=t, E_off=t)
        r = p.simulate(b, t_end=50e-7 + 1e-10, engine="pwl", dt=1e-7, switch_fn=lambda _: m,
                       discontinuity_restart=True, auto_regularize=False)
    assert r.reinit_records
    with pytest.warns(RuntimeWarning, match="not in its loss port"):
        by = {e["name"]: e for e in device_loss_summary(b, r)}
    assert by["S"]["impulse_in_junction"] is False


def test_a_decimated_pulse_interval_moves_no_row():
    # store_every = 4 stores t = 0, 4h, 8h, 12h, ...: the pulse step (11h, 12h] is
    # the last of the stored interval (8h, 12h]; its first three steps conducted
    # from the pre-jump state, so the interval's start row is not replaced
    from pulsim._result_views import _reinit_step_starts
    _, r = _caploop_closing(1e-6, store_every=4)
    assert r.reinit_records
    t = np.asarray(r.times)
    # the case under test: a stored pulse sample whose interval spans earlier steps
    assert any(0 < rec.sample < t.size and t[rec.sample - 1] < rec.t - 0.5e-6
               for rec in r.reinit_records)
    x = np.asarray(r.states)
    st, _ = _reinit_step_starts(r, x, None)
    assert st is x
