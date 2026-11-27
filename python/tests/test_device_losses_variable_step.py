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
    # integrator='esdirk43': conduction on its stage weights b, the same
    # event sites (add-high-order-stiff-integrator)
    pytest.param("trbdf2", dict(rtol=1e-6, atol=1e-9, dt=2.5e-7, integrator="esdirk43"),
                 2e-5, 2.5e-5, id="esdirk43"),
    # the exact engine: the event limits and ∫v·i in closed form (measured
    # 2e-15; the bound stays above 1e-12 for other platforms' libm)
    ("dsed", {}, 1e-10, 1e-10),
])
def test_event_and_conduction_energy_converge_to_the_closed_form_truth(engine, kw, tol_ev, tol_cond):
    ev, cond = _truth()
    res = _run(engine, **kw)
    for n in ("S1", "D1"):
        e_ev, e_cond, k = _kernel(res, n)
        assert k == 2 * NPER, (n, k)
        assert e_ev == pytest.approx(ev[n], rel=tol_ev), n
        assert e_cond == pytest.approx(cond[n], rel=tol_cond), n


@pytest.mark.parametrize("kw,lo,hi", [
    # the restart-less step reads an edge's two sides one dt apart: first order
    # (1.89e-7 J -> 4.70e-8 J of 1.4639e-3 J for dt / 4: ratio 4.03)
    (dict(discontinuity_restart=False), 2.5, 6.0),
    # release: the restart rule reads the edge at its limits: second order
    # (2.96e-10 J -> 1.85e-11 J: ratio 15.98)
    ({}, 12.0, 20.0),
], ids=["restart-less", "default"])
def test_pwl_event_energy_order_in_dt(kw, lo, hi):
    ev, _ = _truth()
    err = [abs(_kernel(_run("pwl", dt=dt, **kw), "S1")[0] - ev["S1"]) for dt in (1e-7, 2.5e-8)]
    assert lo < err[0] / err[1] < hi


@pytest.mark.parametrize("integ", [None, "esdirk43"])
def test_trbdf2_ledger_and_junction_receive_the_recorded_heat(integ):
    """Closed loop on trbdf2: heat_injected + heat_pending = every event and
    conduction energy of S1 (the ledger), and the thermal network received
    it (C·ΔT + ∫(T - T_amb)/R dt) to the run's accuracy."""
    r_th, c_th = 5.0, 1e-4
    b, fn = _rl_chopper(thermal=(r_th, c_th))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, T_END, switch_fn=fn, engine="trbdf2", rtol=1e-6, atol=1e-9,
                         integrator=integ)
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
    the totals agree with the uninterrupted run to that tolerance.
    rtol 1e-8 (was 1e-6, where this run aborted before DECISIONS 17): at
    1e-6 the junction is accurate to ~6e-3 K against the closed form (each
    commutation's energy reaches it over the first step after the edge), so
    two step sequences cannot agree to 1e-4 K there; at 1e-8 they agree to
    1.2e-5 K."""
    kw = dict(engine="trbdf2", rtol=1e-8, atol=1e-11)
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


# ---- an unflipped device's right limit at a diode landing (gate-edge line) --
# 325 V / 50 Hz half-wave rectifier, RC load; Q3 is never gated (g_off 1e-2 S)
# across the diode, so every diode landing is an event Q3 does not flip. Truth:
# engine='pwl' + loss port at dt 1e-6 / 5e-7 / 2.5e-7 reads 96.470079186 /
# 96.470079394 / 96.470079366 J. Writing Q3's right-limit reading at once
# made the landing step (booked after the event) end on it: -1.57e-4 J on
# trbdf2, -4.6e-5 J on esdirk43; pending until the next step: -1.75e-5 /
# -2.0e-6 J (= the 14f numbers).
@pytest.mark.parametrize(("kw", "tol"), [
    (dict(engine="trbdf2"), 5e-5),
    (dict(engine="trbdf2", integrator="esdirk43"), 1e-5),
], ids=["trbdf2", "esdirk43"])
def test_unflipped_device_conduction_across_diode_landings(kw, tol):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "a", "gnd", 0.0, 325.0, 50.0)
    b.add_resistor("Rs", "a", "a2", 0.5)
    b.add_diode("D1", "a2", "k", 1e3, 1e-9)
    b.add_capacitor("C", "k", "gnd", 470e-6)
    b.add_resistor("RL", "k", "gnd", 100.0)
    b.add_switch("Q3", "a2", "k", 1e3, 1e-2)
    b.set_device_losses("Q3", Tj=25.0)
    fn = p.make_combined_switch_fn(b.graph.num_switches, [])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=0.1, switch_fn=fn, **kw)
    assert len(r.commutation_events) == 9
    assert abs(r.device_energy("Q3")["E_cond_electrical"] - 96.4700794) < tol
