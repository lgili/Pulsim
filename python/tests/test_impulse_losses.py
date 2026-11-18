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


def test_no_records_no_new_keys():
    # a run without re-initialisation keeps its entries as they were
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_ideal_switch("S", "a", "k", 0.0, 0.0)
    b.add_resistor("R", "k", "gnd", 1.0)
    m = p.SwitchStateMask(1)
    m.set(0, True)
    r = p.simulate(b, t_end=1e-5, dt=1e-6, engine="pwl", switch_fn=lambda t: m)
    assert len(r.reinit_records) == 0
    assert all("E_impulse_total" not in e for e in device_loss_summary(b, r))
