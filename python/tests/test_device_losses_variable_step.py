"""add-in-kernel-event-losses 3.1-3.3 and the impulsive category: loss ports
on the variable-step engine (trbdf2) and impulsive re-initialisation energy
booked into the junction of the element that carried it.

The cross-engine truth is closed form: an RL chopper (no output capacitor,
one state i_L; each mode is di/dt = a - b i) with PWL conductance devices
and E = k·v·i tables (bilinear interpolation reproduces v·i exactly), so
the event energy at the exact commutation points and ∫v·i of each device
are sums of exponentials. Loss data here is synthetic.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import pulsim as p

VIN, GON, GOFF, LL, RL = 24.0, 100.0, 1e-6, 100e-6, 2.0
FSW, DUTY, NPER = 20e3, 0.5, 20
T_END = NPER / FSW + 10e-6
KON, KOFF = 2e-7, 3e-7   # J / (V·A), synthetic


def _kvi(k):
    return p.LossTable(v_axis=[0.0, 50.0], i_axis=[0.0, 20.0], tj_axis=[25.0, 125.0],
                       energy=np.array([[[0.0, 0.0], [0.0, 0.0]],
                                        [[0.0, 0.0], [k * 1000, k * 1000]]]))


def _rl_chopper(losses=True, thermal=None):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", VIN)
    b.add_switch("S1", "in", "sw", GON, GOFF)
    b.add_diode("D1", "gnd", "sw", GON, GOFF)
    b.add_inductor("L", "sw", "out", LL)
    b.add_resistor("R", "out", "gnd", RL)
    if thermal is not None:
        r_th, c_th = thermal
        b.add_voltage_source("Vamb", "amb", "gnd", 25.0)
        b.add_resistor("Rth", "tj", "amb", r_th)
        b.add_capacitor("Cth", "tj", "amb", c_th)
    if losses:
        for n in ("S1", "D1"):
            kw = dict(junction="tj") if thermal is not None and n == "S1" else dict(Tj=25.0)
            b.set_device_losses(n, E_on=_kvi(KON), E_off=_kvi(KOFF), **kw)
    fn = p.make_pwm_switch_fn(FSW, DUTY, b.switch_index_of("S1"), b.graph.num_switches)
    return b, fn


def _mode(s_on):
    gs, gd = (GON, GOFF) if s_on else (GOFF, GON)
    c0, c1 = gs * VIN / (gs + gd), -1.0 / (gs + gd)   # v_sw = c0 + c1 i
    return gs, gd, c0, c1, c0 / LL, (RL - c1) / LL      # di/dt = a - b i


def _devices(s_on, i):
    gs, gd, c0, c1, _, _ = _mode(s_on)
    vsw = c0 + c1 * i
    return (VIN - vsw, gs * (VIN - vsw)), (-vsw, -gd * vsw)


def _truth():
    """Event energy and ∫v·i per device over [0, T_END], closed form."""
    i, t, s_on = 0.0, 0.0, True
    ev = {"S1": 0.0, "D1": 0.0}
    cond = {"S1": 0.0, "D1": 0.0}
    while True:
        T = min(DUTY / FSW if s_on else (1 - DUTY) / FSW, T_END - t)
        gs, gd, c0, c1, a, bb = _mode(s_on)
        A, B = a / bb, i - a / bb
        e1 = -math.expm1(-bb * T) / bb
        e2 = -math.expm1(-2 * bb * T) / (2 * bb)
        m1, m2 = A * T + B * e1, A * A * T + 2 * A * B * e1 + B * B * e2
        u0 = VIN - c0
        cond["S1"] += gs * (u0 * u0 * T - 2 * u0 * c1 * m1 + c1 * c1 * m2)
        cond["D1"] += gd * (c0 * c0 * T + 2 * c0 * c1 * m1 + c1 * c1 * m2)
        i = A + B * math.exp(-bb * T)
        t += T
        if t >= T_END - 1e-15:
            return ev, cond
        (vsl, isl), (vdl, idl) = _devices(s_on, i)
        (vsr, isr), (vdr, idr) = _devices(not s_on, i)
        if s_on:   # S1 off: v after, i before; D1 on: v before, i after
            ev["S1"] += KOFF * abs(vsr) * abs(isl)
            ev["D1"] += KON * abs(vdl) * abs(idr)
        else:
            ev["S1"] += KON * abs(vsl) * abs(isr)
            ev["D1"] += KOFF * abs(vdr) * abs(idl)
        s_on = not s_on


def _kernel(res, n):
    ev = res.loss_events(n)
    keep = ev["t"] > 1e-12            # the run's start is not a commutation
    return float(np.sum(ev["E"][keep])), res.device_energy(n)["E_cond_electrical"], int(keep.sum())


def _run(engine, **kw):
    b, fn = _rl_chopper()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, T_END, switch_fn=fn, engine=engine, **kw)


@pytest.mark.parametrize("engine,kw,tol_ev,tol_cond", [
    # pwl reads an edge's two sides one dt apart: first order in dt
    ("pwl", dict(dt=1e-7), 1.5e-3, 1.5e-3),
    ("trbdf2", dict(rtol=1e-6, atol=1e-9, dt=2.5e-7), 2e-5, 2.5e-5),
])
def test_event_and_conduction_energy_converge_to_the_closed_form_truth(engine, kw, tol_ev, tol_cond):
    ev, cond = _truth()
    res = _run(engine, **kw)
    for n in ("S1", "D1"):
        e_ev, e_cond, k = _kernel(res, n)
        assert k == 2 * NPER, (n, k)
        assert e_ev == pytest.approx(ev[n], rel=tol_ev), n
        assert e_cond == pytest.approx(cond[n], rel=tol_cond), n


def test_pwl_event_energy_is_first_order_in_dt():
    ev, _ = _truth()
    err = [abs(_kernel(_run("pwl", dt=dt), "S1")[0] - ev["S1"]) for dt in (1e-7, 2.5e-8)]
    assert 2.5 < err[0] / err[1] < 6.0


def test_trbdf2_ledger_and_junction_receive_the_recorded_heat():
    """Closed loop on trbdf2: heat_injected + heat_pending = every event and
    conduction energy of S1 (the ledger), and the thermal network received
    it (C·ΔT + ∫(T - T_amb)/R dt) to the run's accuracy."""
    r_th, c_th = 5.0, 1e-4
    b, fn = _rl_chopper(thermal=(r_th, c_th))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, T_END, switch_fn=fn, engine="trbdf2", rtol=1e-6, atol=1e-9)
    e = res.device_energy("S1")
    recorded = e["E_on"] + e["E_off"] + e["E_cond_electrical"]
    assert res.heat_injected + res.heat_pending == pytest.approx(recorded, rel=1e-12)
    t = np.asarray(res.times)[1:]          # row 0 is the unsolved seed
    tj = np.asarray(res.v("tj"))[1:]
    network = c_th * (tj[-1] - tj[0]) + np.trapezoid(tj - 25.0, t) / r_th
    assert network == pytest.approx(res.heat_injected, rel=2e-3)
    assert res.heat_pending < 0.05 * recorded


def test_trbdf2_record_only_port_leaves_the_run_bit_identical():
    """A port with a fixed Tj and no heat node only observes: the trbdf2
    run is byte for byte the run without it (same binary, same path)."""
    b0, fn0 = _rl_chopper(losses=False)
    b1, fn1 = _rl_chopper(losses=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r0 = p.simulate(b0, T_END, switch_fn=fn0, engine="trbdf2")
        r1 = p.simulate(b1, T_END, switch_fn=fn1, engine="trbdf2")
    assert np.asarray(r0.times).tobytes() == np.asarray(r1.times).tobytes()
    assert np.asarray(r0.states).tobytes() == np.asarray(r1.states).tobytes()
    assert r1.heat_injected == 0.0


def test_trbdf2_resume_continues_the_loss_record():
    """Item 4 on trbdf2: the resumed run's record continues the first run's
    (events, ledgers, heat in flight); trbdf2 resumes to its tolerance, so
    the totals agree with the uninterrupted run to that tolerance."""
    kw = dict(engine="trbdf2", rtol=1e-6, atol=1e-9)
    b, fn = _rl_chopper(thermal=(5.0, 1e-4))
    t1 = 10.3 / FSW
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = p.simulate(b, T_END, switch_fn=fn, **kw)
        first = p.simulate(b, t1, switch_fn=fn, **kw)
        snap = first.final_snapshot
        second = p.simulate(b, T_END, switch_fn=fn, t_start=snap.t, resume_from=snap, **kw)
    ef, es = full.device_energy("S1"), second.device_energy("S1")
    assert (es["n_on"], es["n_off"]) == (ef["n_on"], ef["n_off"])
    # measured 4e-6 (the resumed trajectory differs at rtol); 5e-5 for platforms
    assert es["E_on"] + es["E_off"] == pytest.approx(ef["E_on"] + ef["E_off"], rel=5e-5)
    assert es["E_cond_electrical"] == pytest.approx(ef["E_cond_electrical"], rel=5e-5)
    assert second.heat_injected + second.heat_pending == pytest.approx(
        full.heat_injected + full.heat_pending, rel=5e-5)
    assert np.asarray(second.v("tj"))[-1] == pytest.approx(np.asarray(full.v("tj"))[-1], abs=1e-4)


def test_loss_ports_are_in_the_circuit_hash():
    """Item 5: two circuits that differ only in a loss model never hash
    equal; a circuit without one keeps its hashes (hashed when present)."""
    def h(**kw):
        b, _ = _rl_chopper(losses=False)
        if kw:
            b.set_device_losses("S1", **kw)
        c = b.build()
        return c.structure_hash(), c.value_hash()

    base = h()
    assert base == h()
    a = h(Tj=25.0, E_on=_kvi(KON))
    assert a != base
    assert h(Tj=25.0, E_on=_kvi(2 * KON))[1] != a[1]       # a table value
    assert h(Tj=125.0, E_on=_kvi(KON))[1] != a[1]          # the temperature
    assert h(Tj=25.0, E_off=_kvi(KON))[0] != a[0]          # which table
