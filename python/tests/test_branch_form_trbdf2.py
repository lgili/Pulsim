"""add-branch-form-ideal-switch on `trbdf2` (tasks 5.5, 6.3, 7.2 and the trbdf2 half of
group 4): the branch-form switch and diode, the structural refusals, and the impulsive
re-initialisation by three SDIRK(2,2) steps with the fixed engine's record.

Oracles: the RL closed form 9.99954600 A; the conductance form with the same values; the
prototype's cut-set (|Lambda| = L*i_L(t-)) and capacitor loop (4.000000 V, 6.0 uC, 24.0 uJ);
the ideal DCM buck orbit 8.9963970815 V (`oracles/ideal_switch_reference.py`).
"""
import math
import warnings

import numpy as np
import pytest

import pulsim as p
from tests.oracles import ideal_switch_reference as ref


def _mask(b, on_names):
    m = p.SwitchStateMask(b.graph.num_switches)
    for nm in on_names:
        m.set(b.switch_index_of(nm), True)
    return m


def _sim(b, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, **kw)


# ---------------------------------------------------------------------------------------------
# a. modes without dependent states
# ---------------------------------------------------------------------------------------------
def test_ideal_rl_with_g_off_zero_runs_on_trbdf2():
    """Task 4.4 with g_off = 0 too (refused statically before): a switch that never opens
    makes no dependent mode."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S", "vin", "a", 0.0, 0.0)
    b.add_resistor("R", "a", "b", 1.0)
    b.add_inductor("L", "b", "gnd", 1e-3)
    r = _sim(b, t_end=10e-3, switch_fn=lambda t: _mask(b, ["S"]), engine="trbdf2",
             rtol=1e-9, atol=1e-12)
    assert r.i("L")[-1] == pytest.approx(10 * (1 - math.exp(-10)), abs=1e-6)   # 9.99954600
    assert len(r.reinit_records) == 0


def _buck(branch, R_on=1e-3, g_off=1e-9):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    if branch:
        b.add_ideal_switch("S", "vin", "sw", R_on, g_off)
        b.add_diode("D", "gnd", "sw", form="branch", R_on=R_on, g_off=g_off)
    else:
        b.add_switch("S", "vin", "sw", 1.0 / R_on, g_off)
        b.add_diode("D", "gnd", "sw", 1.0 / R_on, g_off)
    b.add_inductor("L", "sw", "out", 100e-6)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", 5.0)
    return b, p.make_pwm_switch_fn(1e5, 0.4, b.switch_index_of("S"), b.graph.num_switches)


def _bridge(branch):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "ac", "gnd", 0.0, 100.0, 50.0)
    for nm, a, c in (("D1", "ac", "p"), ("D2", "gnd", "p"), ("D3", "n", "ac"),
                     ("D4", "n", "gnd")):
        if branch:
            b.add_diode(nm, a, c, form="branch", R_on=1e-3, g_off=1e-9)
        else:
            b.add_diode(nm, a, c, 1e3, 1e-9)
    b.add_capacitor("C", "p", "n", 100e-6)
    b.add_resistor("R", "p", "n", 50.0)
    b.add_resistor("Rg", "n", "gnd", 1e3)
    return b, lambda t: p.SwitchStateMask(4)


def _vsi(branch):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc", "gnd", 400.0)
    names = []
    for ph in "abc":
        for hi, a, c in ((True, "dc", ph), (False, ph, "gnd")):
            nm = f"S{ph}{'h' if hi else 'l'}"
            names.append(nm)
            if branch:
                b.add_ideal_switch(nm, a, c, 1e-3, 1e-9)
            else:
                b.add_switch(nm, a, c, 1e3, 1e-9)
        b.add_resistor(f"R{ph}", ph, f"m{ph}", 10.0)
        b.add_inductor(f"L{ph}", f"m{ph}", "star", 5e-3)
    idx = [b.switch_index_of(nm) for nm in names]
    n = b.graph.num_switches
    f0, fs = 50.0, 5e3

    def fn(t):
        m = p.SwitchStateMask(n)
        for k, ph in enumerate((0.0, -2 * math.pi / 3, 2 * math.pi / 3)):
            ref_ = 0.8 * math.sin(2 * math.pi * f0 * t + ph)
            car = 2.0 * abs(2.0 * ((t * fs) % 1.0) - 1.0) - 1.0
            up = ref_ > car
            m.set(idx[2 * k], up)
            m.set(idx[2 * k + 1], not up)
        return m
    return b, fn


@pytest.mark.parametrize("name,factory,node,t_end", [
    ("buck", _buck, "out", 1e-3),
    ("bridge", _bridge, None, 20e-3),
    ("vsi", _vsi, "star", 4e-3),
])
def test_branch_form_reproduces_the_conductance_form_on_trbdf2(name, factory, node, t_end):
    """Task 4.4 / 3.4 on trbdf2: R_on = 1e-3, g_off = 1e-9, the branch form against the
    conductance form within the engine tolerance (rtol 1e-7). The bridge is compared on its
    bus (its common mode floats on 1 kOhm, task 3.4 / DECISIONS #8); the inverter on its
    phase currents (the star floats in both forms)."""
    out = {}
    for branch in (False, True):
        b, fn = factory(branch)
        r = _sim(b, t_end=t_end, switch_fn=fn, engine="trbdf2", rtol=1e-7, atol=1e-9)
        t = np.asarray(r.times)
        if name == "bridge":
            y = np.asarray(r.v("p")) - np.asarray(r.v("n"))
        elif name == "vsi":
            y = np.asarray(r.i("La"))
        else:
            y = np.asarray(r.v(node))
        tg = np.linspace(0.0, t_end, 2001)
        out[branch] = (np.interp(tg, t, y), r)
    span = float(np.ptp(out[False][0]))
    assert span > 1.0
    gap = float(np.max(np.abs(out[True][0] - out[False][0])))
    assert gap <= 1e-4 * span, gap
    # the branch currents are algebraic: they do not drive the step controller
    n_c = out[False][1]._trbdf2_stats["n_accept"]
    n_b = out[True][1]._trbdf2_stats["n_accept"]
    assert n_b <= 1.3 * n_c, (n_b, n_c)


def test_branch_diode_reads_its_solved_current():
    """D5: an ON branch-form diode's crossing signal is i_s (amperes): the half-wave
    rectifier's turn-off lands where the solved diode current is zero."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, 10.0, 1e3)
    b.add_diode("D", "a", "o", form="branch", R_on=1e-3, g_off=1e-9)
    b.add_resistor("R", "o", "gnd", 10.0)
    r = _sim(b, t_end=3e-3, switch_fn=lambda t: p.SwitchStateMask(1), engine="trbdf2",
             rtol=1e-8, atol=1e-10)
    st = r._trbdf2_stats
    assert st["n_diode_events"] >= 5           # on/off every half period
    iD = np.asarray(r.i("D"))
    vo = np.asarray(r.v("o"))
    np.testing.assert_allclose(iD, vo / 10.0, atol=1e-7)
    assert iD.min() >= -1e-7                   # reverse current is g_off-scale only


# ---------------------------------------------------------------------------------------------
# d. structural refusals on trbdf2, same four fields as on pwl
# ---------------------------------------------------------------------------------------------
def test_floating_node_refused_on_trbdf2():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S1", "vin", "mid", 0.0, 0.0)
    b.add_ideal_switch("S2", "mid", "gnd", 0.0, 0.0)
    b.add_resistor("R", "vin", "gnd", 10.0)
    with pytest.raises(p.UnsupportedError) as ei:
        _sim(b, t_end=1e-5, switch_fn=lambda t: _mask(b, []), engine="trbdf2")
    e = ei.value
    assert (e.engine, e.feature, e.device, e.kind) == ("trbdf2", "floating_node", "S1, S2",
                                                       "Switch")


def test_ideal_voltage_loop_refused_on_trbdf2():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S1", "vin", "gnd")
    b.add_ideal_switch("S2", "vin", "gnd")
    with pytest.raises(p.UnsupportedError) as ei:
        _sim(b, t_end=1e-5, switch_fn=lambda t: _mask(b, ["S1", "S2"]), engine="trbdf2")
    e = ei.value
    assert (e.engine, e.feature, e.device, e.kind) == ("trbdf2", "ideal_voltage_loop", "S1",
                                                       "Switch")


# ---------------------------------------------------------------------------------------------
# b. re-initialisation at impulsive events (task 5.5 = 5.3 / 5.4 on trbdf2)
# ---------------------------------------------------------------------------------------------
L = 1e-3
T_OPEN = 5e-3


def _cutset(**kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S", "vin", "a", 0.0, 0.0)
    b.add_resistor("R", "a", "b", 1.0)
    b.add_inductor("L", "b", "gnd", L)
    k = b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(1)
        m.set(k, t < T_OPEN)
        return m
    return _sim(b, t_end=6e-3, switch_fn=fn, engine="trbdf2", rtol=1e-9, atol=1e-12, **kw)


@pytest.fixture(scope="module")
def cutset():
    return _cutset()


def test_cutset_on_trbdf2_is_reinitialised_and_recorded(cutset):
    r = cutset
    t = np.asarray(r.times)
    iL, va = np.asarray(r.i("L")), np.asarray(r.v("a"))
    (rec,) = r.reinit_records
    k0 = rec.sample - 1                                     # the sample at the opening
    assert t[k0] == pytest.approx(T_OPEN, abs=1e-12)
    assert rec.t == t[k0]
    assert iL[k0] == pytest.approx(10 * (1 - math.exp(-5)), rel=1e-7)
    assert np.all(np.abs(iL[k0 + 1:]) <= 1e-15 * abs(iL[k0]))     # from the first step
    assert np.max(np.abs(va[k0 + 2:])) <= 1e-9                     # from the second
    assert rec.kind == "cut_set" and list(rec.devices) == ["S"] and list(rec.states) == ["L"]
    assert abs(rec.lambda_) == pytest.approx(L * iL[k0], rel=1e-12)
    assert rec.dW == pytest.approx(0.5 * L * iL[k0] ** 2, rel=1e-9)
    assert rec.dW >= 0.0
    # the explicit projection is the restart's end state
    assert rec.projection_gap <= 1e-12 * abs(iL[k0])
    # three SDIRK steps re-initialise it (not the TR stage): no ringing
    assert r._trbdf2_stats["n_reinit_steps"] == 3


def test_cutset_flux_is_the_sdirk_quadrature_of_the_pulse(cutset):
    """The flux the record moves is the first step's SDIRK quadrature of the node voltage:
    v_b = L di/dt over the step (b - gnd is the inductor), i(t+) = 0."""
    r = cutset
    (rec,) = r.reinit_records
    t = np.asarray(r.times)
    k0 = rec.sample - 1
    h = t[k0 + 1] - t[k0]
    vb = np.asarray(r.v("b"))
    # stage-2 BE: v(t+h) = L (0 - base2)/(gs h); base2 carries (1 - r) of i(t-)
    gs = 1.0 - 1.0 / math.sqrt(2.0)
    base2 = (1.0 - (1.0 - gs) / gs) * rec.lambda_ / L
    assert vb[k0 + 1] == pytest.approx(-L * base2 / (gs * h), rel=1e-9)


def test_cutset_on_impulse_error_on_trbdf2():
    with pytest.raises(p.UnsupportedError) as ei:
        _cutset(on_impulse="error")
    e = ei.value
    assert (e.engine, e.feature, e.device, e.kind) == ("trbdf2", "impulse", "S", "Switch")
    assert "'S' makes L dependent" in str(e)


def test_cutset_on_impulse_warn_on_trbdf2():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S", "vin", "a", 0.0, 0.0)
    b.add_resistor("R", "a", "b", 1.0)
    b.add_inductor("L", "b", "gnd", L)

    def fn(t):
        m = p.SwitchStateMask(1)
        m.set(0, t < 1e-4)
        return m
    with pytest.warns(UserWarning, match="impulsive re-initialisation"):
        r = p.simulate(b, t_end=2e-4, switch_fn=fn, engine="trbdf2", on_impulse="warn")
    assert len(r.reinit_records) == 1


def _caploop(**kw):
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "x", "gnd", 1e-6, 10.0)
    b.add_capacitor("C2", "y", "gnd", 3e-6, 2.0)
    b.add_ideal_switch("S", "x", "y", 0.0, 1e-9)
    m = p.SwitchStateMask(1)
    m.set(0, True)
    return _sim(b, t_end=5e-7, switch_fn=lambda t: m, engine="trbdf2", rtol=1e-9,
                atol=1e-12, auto_regularize=False, **kw)


def test_capacitor_loop_on_trbdf2_is_reinitialised_and_recorded():
    r = _caploop()
    v1, v2 = np.asarray(r.v("x")), np.asarray(r.v("y"))
    assert v1[0] == 10.0 and v2[0] == 2.0
    assert np.max(np.abs(v1[1:] - 4.0)) <= 1e-12
    assert np.max(np.abs(v2[1:] - 4.0)) <= 1e-12
    iS = np.asarray(r.i("S"))
    assert np.max(np.abs(iS[2:])) < 1e-9
    (rec,) = r.reinit_records
    assert rec.kind == "capacitor_loop" and list(rec.devices) == ["S"]
    assert sorted(rec.states) == ["C1", "C2"]
    assert abs(rec.lambda_) == pytest.approx(6.0e-6, rel=1e-9)
    assert rec.dW == pytest.approx(24.0e-6, rel=1e-9)
    assert rec.t == 0.0 and rec.sample == 1
    assert rec.projection_gap <= 1e-12


def test_the_record_means_the_same_on_both_engines():
    """Same circuit, same record: the capacitor loop at t = 0 on pwl (restart rule) and on
    trbdf2 (charge 6.0 uC, 24.0 uJ, the explicit projection)."""
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "x", "gnd", 1e-6, 10.0)
    b.add_capacitor("C2", "y", "gnd", 3e-6, 2.0)
    b.add_ideal_switch("S", "x", "y", 0.0, 1e-9)
    m = p.SwitchStateMask(1)
    m.set(0, True)
    rp = _sim(b, t_end=5e-7, dt=1e-7, switch_fn=lambda t: m, engine="pwl",
              auto_regularize=False, discontinuity_restart=True)
    rt = _caploop()
    (a,), (c,) = rp.reinit_records, rt.reinit_records
    assert (a.kind, list(a.devices), sorted(a.states), a.t) == (
        c.kind, list(c.devices), sorted(c.states), c.t)
    assert c.lambda_ == pytest.approx(a.lambda_, rel=1e-12)
    assert c.dW == pytest.approx(a.dW, rel=1e-12)


def test_a_consistent_commutation_records_nothing_on_trbdf2():
    """5.6: the ideal diode of a DCM buck turns off at i = 0: no record."""
    r = _ideal_dcm_buck(periods=20)
    assert len(r.reinit_records) == 0
    iL = np.asarray(r.i("L"))
    # The landing sample is on the post-crossing side of i_s = 0: -1.9e-10 A measured,
    # 2e-15 s of di/dt = 9e4 A/s (the landing bracket); the idle samples are exact zeros.
    assert np.min(iL) >= -1e-9


# ---------------------------------------------------------------------------------------------
# c. the ideal DCM buck (6.3) and the smallest committed step (7.2)
# ---------------------------------------------------------------------------------------------
T_BUCK = ref.T_BUCK


def _ideal_dcm_buck(periods, branch=True, **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    if branch:
        b.add_ideal_switch("S", "vin", "sw", 0.0, 0.0)
        b.add_diode("D1", "gnd", "sw", form="branch", R_on=0.0, g_off=0.0)
    else:
        b.add_switch("S", "vin", "sw", 1e3, 1e-9)
        b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6, 0.0)
    b.add_capacitor("C", "vout", "gnd", 47e-6, ref.VC_REF)
    b.add_resistor("R", "vout", "gnd", 50.0)
    fn = p.make_pwm_switch_fn(1 / T_BUCK, 0.3, b.switch_index_of("S"), b.graph.num_switches)
    kw.setdefault("rtol", 1e-9)
    kw.setdefault("atol", 1e-12)
    return _sim(b, t_end=periods * T_BUCK, switch_fn=fn, engine="trbdf2", **kw)


@pytest.fixture(scope="module")
def dcm_runs():
    return {branch: _ideal_dcm_buck(periods=100, branch=branch) for branch in (True, False)}


def test_ideal_dcm_buck_orbit_on_trbdf2(dcm_runs):
    """Task 6.3: started on the orbit, the ideal DCM buck stays on 8.9963970815 V within
    1e-6 at rtol 1e-9 (sampled at the rising edges)."""
    r = dcm_runs[True]
    t, v = np.asarray(r.times), np.asarray(r.v("vout"))
    edges = np.arange(1, 101) * T_BUCK
    idx = [int(np.argmin(np.abs(t - te))) for te in edges]
    assert np.all(np.abs(t[idx] - edges) <= 1e-12)
    err = np.abs(v[idx] - ref.VC_REF)
    assert err.max() <= 1e-6, err.max()
    iL = np.asarray(r.i("L"))
    sw = np.asarray(r.switch_states)
    idle = ~sw.any(axis=1)
    assert idle.sum() > 50
    assert np.max(np.abs(iL[idle])) <= 1e-12


def test_branch_form_smallest_step_is_not_the_smaller(dcm_runs):
    """Task 7.2: the smallest committed step of the DCM buck, branch form (ideal) against
    the conductance form (g_on = 1e3, g_off = 1e-9; 8.26e-13 s in the audit): the branch
    form's must not be the smaller. The committed gate-edge probes (h_probe_floor, the same
    1e-14 s in both forms) are left out. Measured: 1.56e-10 s (branch) against 1.11e-14 s
    (conductance, at the diode turn-off, 3099 steps below 1e-12 s in 100 periods)."""
    h = {}
    for k, r in dcm_runs.items():
        t = np.asarray(r.times)
        ph = (t[:-1] / T_BUCK) % 1.0
        edge = (np.abs(ph - 0.3) < 1e-6) | (ph < 1e-6) | (ph > 1 - 1e-6)
        h[k] = float(np.min(np.diff(t)[~edge]))
    assert h[True] >= h[False], h
    assert h[True] > 1e-11, h


# ---------------------------------------------------------------------------------------------
# rounding fuzz on the two decisions this change adds (dependent-state detection, the
# branch-form predicate): identical records and event counts under last-bit perturbations
# ---------------------------------------------------------------------------------------------
def test_reinit_and_branch_predicate_are_stable_under_rounding_fuzz():
    # Two decisions under 1e-15 perturbations: without a freewheel path every
    # opening of S enters the cut-set {S} (3 records, 3 SDIRK steps each); with
    # the ideal freewheel diode the cascade turns D on at each opening (it
    # carries the inductor current, no record).
    for freewheel in (False, True):
        rng = np.random.default_rng(7)
        seen = set()
        for _ in range(8):
            f = 1.0 + float(rng.uniform(-1.0, 1.0)) * 1e-15
            b = p.CircuitBuilder()
            b.add_voltage_source("V", "vin", "gnd", 10.0 * f)
            b.add_ideal_switch("S", "vin", "a", 0.0, 0.0)
            b.add_resistor("R", "a", "b", 1.0 * f)
            b.add_inductor("L", "b", "gnd", L)
            if freewheel:
                b.add_diode("D", "c", "a", form="branch", R_on=0.0, g_off=0.0)
                b.add_resistor("Rd", "c", "gnd", 5.0)
            k = b.switch_index_of("S")

            def fn(t, k=k, n=b.graph.num_switches):
                m = p.SwitchStateMask(n)
                m.set(k, (t % 2e-3) < 1e-3)
                return m
            r = _sim(b, t_end=6e-3, switch_fn=fn, engine="trbdf2", rtol=1e-7,
                     atol=1e-10)
            d_on = bool(np.max(np.asarray(r.i("D"))) > 1.0) if freewheel else False
            seen.add((len(r.reinit_records), r._trbdf2_stats["n_reinit_steps"], d_on))
        assert seen == {(0, 0, True) if freewheel else (3, 9, False)}, seen
