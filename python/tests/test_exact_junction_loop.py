"""add-in-kernel-event-losses 3.3 / add-exact-propagator-strategy session 25:
the junction temperature loop on the exact engine.

The heat is a current into the deposit node and the thermal network is states
of the same reduced vector; each accepted step adds the network's exact
response to the step's conduction heat (the devices' v·i along the mode's
closed form) and each event's energy is an impulse at its instant
(dsed/loss_port.hpp). Truths: the RL chopper's single RC node in closed form
(its loss is known in closed form, test_device_losses_variable_step), the
buck and the VSI against trbdf2 at tight tolerance, the energy ledger, a
resume. Loss data here is synthetic.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import pulsim as p

from .test_device_losses_kernel import _switch_names, _vsi
from .test_device_losses_variable_step import (DUTY, FSW, KOFF, KON, T_END, VIN, _kvi, _mode, _rl_chopper)

R_TH, C_TH, T_AMB = 5.0, 1e-4, 25.0


def _sim(b, t_end, fn, engine, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end, switch_fn=fn, engine=engine, **kw)


def _rc_truth(times):
    """T_j of S1's single RC node (R_TH, C_TH to T_AMB) at `times`: its heat is
    gs·v_S1² in each mode (i_L = A + B e^{-bt}, so a sum of three exponentials)
    plus the switching energy k·v·i at each commutation as an impulse E/C."""
    tau = R_TH * C_TH
    out = np.full(len(times), np.nan)
    i, t, s_on, th = 0.0, 0.0, True, 0.0   # th = T_j - T_amb
    k = 0
    times = np.asarray(times)

    def seg(th0, T, terms):
        # th(T) = th0 e^{-T/tau} + (1/C) ∫0^T e^{-(T-s)/tau} Σ c e^{-q s} ds
        v = th0 * math.exp(-T / tau)
        for c, q in terms:
            d = 1.0 / tau - q
            v += c / C_TH * (math.exp(-q * T) - math.exp(-T / tau)) / d
        return v

    while t < T_END:
        T = min(DUTY / FSW if s_on else (1 - DUTY) / FSW, T_END - t)
        gs, gd, c0, c1, a, bb = _mode(s_on)
        A, B = a / bb, i - a / bb
        al, be = (VIN - c0) - c1 * A, c1 * B           # v_S1 = al - be e^{-bb s}
        terms = [(gs * al * al, 0.0), (-2 * gs * al * be, bb), (gs * be * be, 2 * bb)]
        while k < len(times) and times[k] <= t + T:
            out[k] = seg(th, times[k] - t, terms)
            k += 1
        th = seg(th, T, terms)
        i = A + B * math.exp(-bb * T)
        t += T
        if t >= T_END - 1e-15:
            break
        # the commutation at t: S1's energy (the reading rule of the truth)
        gsl, gdl, c0l, c1l, _, _ = _mode(s_on)
        gsr, gdr, c0r, c1r, _, _ = _mode(not s_on)
        vsl, isl = VIN - (c0l + c1l * i), gsl * (VIN - (c0l + c1l * i))
        vsr, isr = VIN - (c0r + c1r * i), gsr * (VIN - (c0r + c1r * i))
        e = KOFF * abs(vsr) * abs(isl) if s_on else KON * abs(vsl) * abs(isr)
        th += e / C_TH
        s_on = not s_on
    return out + T_AMB


def test_a_single_rc_junction_follows_its_closed_form():
    """The RL chopper's S1 into one RC node: T_j(t) against the closed form at
    every sample off a gate edge (an edge's row is its left or right limit)."""
    b, fn = _rl_chopper(thermal=(R_TH, C_TH))
    r = _sim(b, T_END, fn, "dsed")
    t = np.asarray(r.times)
    tj = np.asarray(r.v("tj"))
    edge = 0.5 / FSW
    off = np.abs(t / edge - np.round(t / edge)) > 1e-6
    truth = _rc_truth(t[off])
    rise = truth.max() - T_AMB
    assert rise > 1.0                                 # the loop moved the junction
    np.testing.assert_allclose(tj[off], truth, rtol=0, atol=1e-9 * rise)


def test_the_heat_booked_is_the_heat_the_network_received():
    """heat_injected = every energy booked to S1 (nothing pending: the exact
    engine delivers each step's heat when the step is accepted)."""
    b, fn = _rl_chopper(thermal=(R_TH, C_TH))
    r = _sim(b, T_END, fn, "dsed")
    e = r.device_energy("S1")
    booked = e["E_on"] + e["E_off"] + e["E_cond_electrical"] + e["E_impulse"]
    assert r.heat_pending == 0.0
    assert r.heat_injected == pytest.approx(booked, rel=1e-12)


_RNG = np.random.default_rng(7)
_AX = dict(v_axis=[0.0, 400.0, 800.0], i_axis=[0.0, 30.0, 60.0], tj_axis=[25.0, 150.0])
E_ON = p.LossTable(**_AX, energy=_RNG.uniform(1e-3, 5e-3, (3, 3, 2)))
E_OFF = p.LossTable(**_AX, energy=_RNG.uniform(1e-3, 5e-3, (3, 3, 2)))


def _thermal(b, names, r_th=2.0, c_th=2e-4):
    """One RC junction per device, temperature-dependent tables (the loop closes)."""
    b.add_voltage_source("Vamb", "amb", "gnd", T_AMB)
    for k, n in enumerate(names):
        b.add_resistor(f"Rth{k}", f"tj{k}", "amb", r_th)
        b.add_capacitor(f"Cth{k}", f"tj{k}", "amb", c_th)
        b.set_device_losses(n, junction=f"tj{k}", E_on=E_ON, E_off=E_OFF)


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("Q1", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 10.0)
    fn = p.make_pwm_switch_fn(100e3, 0.5, b.switch_index_of("Q1"), b.graph.num_switches)
    _thermal(b, ["Q1", "D1"])
    return b, fn, ["Q1", "D1"]


def _vsi_case():
    b, fn = _vsi()
    names = _switch_names(b)
    _thermal(b, names)
    return b, fn, names


# The VSI runs on trbdf2 since its thermal fix (integration batch 14n). Its
# reference is taken at rtol 1e-11: at 1e-9 trbdf2's own error is the gap
# (2.9e-4 of the rise; on the single RC junction above it is 1.1e-4 against the
# closed form, the exact engine 1.9e-15), at 1e-10 6.8e-5, at 1e-11 2.9e-6
# (integration batch 14o, macOS arm64).
@pytest.mark.parametrize("make,t_end,rtol", [
    (_buck, 202.5e-6, 1e-10),
    (_vsi_case, 2.03e-3, 1e-11),
], ids=["buck", "vsi"])
def test_junction_temperatures_agree_with_trbdf2_at_tight_tolerance(make, t_end, rtol):
    """T_j(t) of every junction against trbdf2 at tight rtol, within 1e-4 of
    the temperature rise (trbdf2 delivers a step's heat over the next step: a
    lag of one tiny step)."""
    b, fn, names = make()
    ex = _sim(b, t_end, fn, "dsed")
    b2, fn2, _ = make()
    tr = _sim(b2, t_end, fn2, "trbdf2", rtol=rtol, atol=1e-12)
    te, tt = np.asarray(ex.times), np.asarray(tr.times)
    # Under the default output rule the exact engine writes each edge's right
    # limit 2 ps after it (the event's heat already in); trbdf2 books that heat
    # on its tiny post-edge step and delivers it over the NEXT step (~70 ns):
    # the reference's own lag, so the exact rows inside it are not compared
    # (one per edge; every other row is held at the original bound).
    edges = te[:-1][np.diff(te) < 1e-10]
    lag = np.zeros(len(te), bool)
    for e in edges:
        lag |= (te > e) & (te <= tt[np.searchsorted(tt, e + 1e-12, side="right")])
    assert len(edges) > 0 and lag.sum() == len(edges)
    # the heat those rows carry is the same on both engines (the ledger)
    assert ex.heat_injected == pytest.approx(tr.heat_injected, rel=1e-4)
    for k, _n in enumerate(names):
        a = np.asarray(ex.v(f"tj{k}"))
        c = np.interp(te, tt[1:], np.asarray(tr.v(f"tj{k}"))[1:])
        rise = a.max() - T_AMB
        assert rise > 0.01, k
        d = np.abs(a - c)[~lag]
        assert np.max(d) <= 1e-4 * rise, (k, np.max(d) / rise)


def test_a_resumed_run_carries_the_junction_temperatures():
    b, fn, _ = _buck()
    u = _sim(b, 200e-6, fn, "dsed")
    b1, fn1, _ = _buck()
    a = _sim(b1, 100e-6, fn1, "dsed")
    b2, fn2, _ = _buck()
    r = _sim(b2, 200e-6, fn2, "dsed", resume_from=a.final_snapshot)
    np.testing.assert_allclose(r.final_snapshot.x, u.final_snapshot.x, rtol=1e-12, atol=1e-12)
    t_u = np.asarray(u.times)
    i0 = int(np.searchsorted(t_u, r.times[0]))
    assert t_u[i0] == pytest.approx(r.times[0], rel=1e-15)
    for k in (0, 1):
        assert r.v(f"tj{k}")[-1] == pytest.approx(u.v(f"tj{k}")[-1], rel=1e-12)
        # row 0 (the landing at the split edge) carries that edge's heat as the
        # uninterrupted run's row at the same instant does (session 26 review)
        assert r.v(f"tj{k}")[0] == pytest.approx(u.v(f"tj{k}")[i0], rel=1e-12)
    # the resumed tracker continues the first run's ledger (field 17): cumulative
    assert r.heat_injected == pytest.approx(u.heat_injected, rel=1e-12)


def test_closed_loop_mean_junction_temperature_agrees_across_the_three_engines():
    """Task 3.3: the closed-loop mean T_j within 0.2 K on pwl, trbdf2 and dsed
    (the RL chopper, S1 into its RC node), and the fixed-Tj energies within
    1e-6 on the variable-step engines (trbdf2 at tight tolerance, dsed)."""
    means = {}
    for eng, kw in (("pwl", dict(dt=1e-7)), ("trbdf2", dict(rtol=1e-8, atol=1e-10)),
                    ("dsed", {})):
        b, fn = _rl_chopper(thermal=(R_TH, C_TH))
        r = _sim(b, T_END, fn, eng, **kw)
        t = np.asarray(r.times)[1:]
        tj = np.asarray(r.v("tj"))[1:]
        means[eng] = np.trapezoid(tj, t) / (t[-1] - t[0])
    assert max(means.values()) - min(means.values()) < 0.2, means
    b, fn = _rl_chopper()
    ex = _sim(b, T_END, fn, "dsed")
    b, fn = _rl_chopper()
    tr = _sim(b, T_END, fn, "trbdf2", rtol=1e-10, atol=1e-12)
    for n in ("S1", "D1"):
        a, c = ex.device_energy(n), tr.device_energy(n)
        for key in ("E_on", "E_off", "E_cond_electrical"):
            assert a[key] == pytest.approx(c[key], rel=1e-6), (n, key)


def test_a_deposit_node_read_instantly_by_an_algebraic_unknown_is_refused():
    """A Foster-like ladder whose middle node has no capacitor: the heat's
    response there is instantaneous, not a state's; refused by name."""
    b, fn = _rl_chopper(losses=False)
    b.add_voltage_source("Vamb", "amb", "gnd", T_AMB)
    b.add_capacitor("Cj", "tj", "amb", C_TH)
    b.add_resistor("R1", "tj", "mid", 1.0)
    b.add_resistor("R2", "mid", "amb", 1.0)
    b.set_device_losses("S1", junction="mid", E_on=_kvi(KON), E_off=_kvi(KOFF))
    with pytest.raises(Exception):
        _sim(b, T_END, fn, "dsed")


def test_a_trbdf2_snapshot_resumed_on_the_exact_engine_delivers_its_heat_once():
    """Integration batch 14o: a trbdf2 snapshot's tracker state carries the
    predicted conduction power of trbdf2's next step (its thermal fix); the
    exact engine delivers conduction heat in closed form and must drop it.
    It booked it every step and took it back out of the queue: the RL
    chopper's junction lost heat, 2.5e-4 of the rise after 500 us. The
    resumed run follows the single RC node's closed form; pwl's snapshot as
    the control."""
    edge = 0.5 / FSW
    ts = round(0.5 * T_END / edge) * edge
    for eng, kw in (("trbdf2", dict(rtol=1e-11, atol=1e-13)), ("pwl", dict(dt=1e-9))):
        b, fn = _rl_chopper(thermal=(R_TH, C_TH))
        a = _sim(b, ts, fn, eng, **kw)
        b, fn = _rl_chopper(thermal=(R_TH, C_TH))
        r = _sim(b, T_END, fn, "dsed", resume_from=a.final_snapshot)
        t = np.asarray(r.times)
        off = np.abs(t / edge - np.round(t / edge)) > 1e-6
        truth = _rc_truth(t[off])
        rise = truth.max() - T_AMB
        np.testing.assert_allclose(np.asarray(r.v("tj"))[off], truth, rtol=0, atol=1e-5 * rise,
                                   err_msg=eng)
        assert r.heat_pending == 0.0
