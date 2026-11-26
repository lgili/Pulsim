"""Release findings, group D3: TR-BDF2's restart at a gate edge and at the
run start (openspec/changes/update-number-changing-release/fixes/group-d3.md)."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import pulsim as p

sys.path.insert(0, str(Path(__file__).parent))


def _sim(b, fn, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, switch_fn=fn, engine="trbdf2", **kw)


def test_coss_gate_edge_restarts_by_the_sdirk_step():
    """The landed gate edge committed a trapezoidal probe that started from
    the pre-edge capacitor current (38.94 V where the ODE gives 30.67 V at
    2.8e-13 s): every edge of a circuit with a stateful kind now restarts by
    the SDIRK(2,2) step from the committed state, as a diode event does.
    Truth: the channel's conduction energy from the run's own trajectory
    (2.1.0 +3.80 %, this line +0.057 %)."""
    from test_release_findings_de import _conduction_truth, _run_coss
    r = _run_coss(engine="trbdf2")
    st = r._trbdf2_stats
    assert st["n_gate_events"] == 40
    assert st["n_sdirk_steps"] >= st["n_gate_events"]
    got = r._device_losses[0].E_cond_electrical
    assert abs(got / _conduction_truth(r) - 1) < 1e-3


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("Q1", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 10.0)
    fn = p.make_pwm_switch_fn(100e3, 0.5, b.switch_index_of("Q1"), b.graph.num_switches)
    return b, fn


def test_linear_gate_edge_restarts_from_the_probe_slots_not_sdirk():
    """A linear circuit keeps TR-BDF2 after a gate edge (slots from the
    probe's own companion currents, the probe not committed): an SDIRK step
    after every edge cost +70 % accepted steps on this buck (its estimate is
    first order) and moved two esdirk43 scoreboard cells out of their class."""
    b, fn = _buck()
    r = _sim(b, fn, t_end=1e-3)
    st = r._trbdf2_stats
    assert st["n_gate_events"] >= 199
    assert st["n_sdirk_steps"] < 0.2 * st["n_gate_events"]
    # 1088 accepted steps measured (1097 before); SDIRK after every edge: 1575
    assert st["n_accept"] < 1400
    # the edge is a jump in the record: a sample right after each edge
    t = np.asarray(r.times)
    for k in (20, 50, 99):
        edge = k * 1e-5
        j = np.searchsorted(t, edge + 1e-15)
        assert t[j] - edge < 1e-9


def _satind(i_sat=2.0):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "a", "gnd", 0.0, 20.0, 1e3)
    b.add_resistor("R", "a", "b", 1.0)
    b.add_saturable_inductor("Ls", "b", "gnd", 1e-3, i_sat, 1e-5)
    return b, p.make_combined_switch_fn(b.graph.num_switches, [])


def _rl_sine_truth(t, L=1e-3, R=1.0, A=20.0, f=1e3):
    """i(t) of a sine from zero into R + L, i(0) = 0 (closed form)."""
    w = 2 * np.pi * f
    z, ph = np.hypot(R, w * L), np.arctan2(w * L, R)
    return A / z * (np.sin(w * t - ph) + np.sin(ph) * np.exp(-R * t / L))


def test_consistent_start_takes_no_restart_step():
    """A sine from zero into R + a saturable L starts consistent (v_L(0) = 0
    is the slot): the first step is TR-BDF2's own, not the first-order
    restart step the stateful kind forced on every start. Truth: the RL
    closed form (the law unsaturated, i_sat 1e6 A)."""
    b, fn = _satind(i_sat=1e6)
    for kw in ({}, {"integrator": "esdirk43"}):
        r = _sim(b, fn, t_end=1e-4, rtol=1e-8, atol=1e-11, **kw)
        assert r._trbdf2_stats["n_sdirk_steps"] == 0
        t = np.asarray(r.times)
        err = np.max(np.abs(np.asarray(r.i("Ls")) - _rl_sine_truth(t)))
        assert err < 1e-7, (kw, err)


def test_inconsistent_start_keeps_the_restart_step():
    """A current-driven nonlinear capacitor (F-S-1) and a source switched onto
    a zero-state gated buck with a Lauritzen diode start inconsistent: the
    restart step stays (without it the buck's first TR-BDF2 step was rejected
    down to 4e-13 s). Truth: v = -I R (1 - exp(-t / RC)) (10 A out of n) of the (linear,
    m = 0) capacitor."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 0.0)
    b.add_resistor("R", "a", "n", 1e-3)
    b.add_nonlinear_capacitor("C", "n", "gnd", 1e-9, 25.0, 0.0)
    b.add_current_source("I", "gnd", "n", 10.0)
    r = _sim(b, p.make_combined_switch_fn(b.graph.num_switches, []), t_end=1e-11)
    assert r._trbdf2_stats["n_sdirk_steps"] >= 1
    t = np.asarray(r.times)
    v = np.asarray(r.v("n"))
    assert np.max(np.abs(v + 1e-2 * (1 - np.exp(-t / 1e-12)))) < 1e-6
    from test_trbdf2_stateful_with_diodes import _gated_buck
    r = _gated_buck("lauritzen", "trbdf2")
    t = np.asarray(r.times)
    assert r._trbdf2_stats["n_sdirk_steps"] >= 1
    assert t[1] - t[0] > 1e-10


def _syncbuck_dead_time():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("Q1", "vin", "sw", 1e3, 1e-9)
    b.add_switch("Q2", "sw", "gnd", 1e3, 1e-9)
    b.add_diode("D2", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 10.0)
    fn = p.make_dead_time_pwm_pair_fn(100e3, 0.5, b.switch_index_of("Q1"),
                                      b.switch_index_of("Q2"), b.graph.num_switches, 2e-7)
    return b, fn


def test_post_edge_samples_reach_the_recorder_and_the_rows():
    """A post-edge sample (x(t_e+) stored h_p after the edge, not a step) is
    a stored row for every consumer: the in-loop recorder has each one, also
    when the step after an edge lands on the next edge (dead time)."""
    b0, fn0 = _syncbuck_dead_time()
    r0 = _sim(b0, fn0, t_end=2e-4)
    b1, fn1 = _syncbuck_dead_time()
    r1 = _sim(b1, fn1, t_end=2e-4, recorders=[p.Recorder("r", ["v(sw)"], keep="every")])
    assert r1.recorder_feed["mode"] == "in-loop", r1.recorder_feed
    t = np.asarray(r1.times)
    np.testing.assert_array_equal(t, np.asarray(r0.times))
    np.testing.assert_array_equal(r1.recorded["r"].times, t)
    np.testing.assert_array_equal(r1.recorded["r"]["v(sw)"], np.asarray(r1.v("sw")))


# ---- a run resumed at / just after a gate edge, with a stateful kind ---------
# V -> S (ON at TE) -> R -> C0/(1 + v/V0)^m. Truth by quadrature (mpmath):
# t(v) = int R(t) C(u)/(V - u) du, with 1/g_off + R before TE (1e-5 V of
# leakage at TE) and R after. Measured (rtol 1e-8): esdirk43 -9.8e-10 V on
# every split (uninterrupted -1.0e-9); trbdf2 +2.4e-7 V split, +1.4e-6 V
# uninterrupted (its h is the span's h_max there, 1.5 ns, not a defect).
_RV, _RR, _RC0, _RV0, _RM, _RTE, _RT = 10.0, 100.0, 1e-9, 25.0, 0.5, 1e-6, 1.5e-6


def _resume_truth():
    import mpmath as mp  # noqa: PLC0415
    mp.mp.dps = 30

    def tq(r, a, v):
        return mp.quad(lambda u: r * _RC0 / (1 + u / _RV0) ** _RM / (_RV - u), [a, v])
    v1 = mp.findroot(lambda v: tq(_RR + mp.mpf(1e9), 0, v) - _RTE, mp.mpf(1e-5))
    guess = _RV * (1 - mp.e ** (-(_RT - _RTE) / (_RR * _RC0)))
    return float(mp.findroot(lambda v: tq(_RR, v1, v) - (_RT - _RTE), guess))


def _resumed(splits, **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", _RV)
    b.add_switch("S", "in", "a", 1e3, 1e-9)
    b.add_resistor("R", "a", "n", _RR - 1e-3)   # + 1/g_on = R
    b.add_nonlinear_capacitor("C", "n", "gnd", _RC0, _RV0, _RM)
    n, k = b.graph.num_switches, b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, t >= _RTE)
        return m
    snap = r = None
    for te in splits:
        r = _sim(b, fn, t_end=te, resume_from=snap, **kw)
        snap = r.final_snapshot
    return float(np.asarray(r.v("n"))[-1])


def test_run_resumed_at_and_after_a_gate_edge_meets_the_quadrature():
    pytest = __import__("pytest")
    pytest.importorskip("mpmath")
    tv = _resume_truth()
    for d in (0.0, 1e-12, 1e-10, 3e-9):
        sp = [_RTE + d, _RT]
        assert abs(_resumed(sp, integrator="esdirk43", rtol=1e-8, atol=1e-11) - tv) < 5e-9, d
        assert abs(_resumed(sp, rtol=1e-8, atol=1e-11) - tv) < 1e-6, d


# ---- the gate-edge jump rule (fs5 cut-set) is bounded to the round-off floor --
# L1 (atan law, unsaturated) + L2 forced into series when S opens at 200 us:
# S's off conductance dissipates 1/2 Leq (i1- - i2-)^2 over the equalisation,
# tau = Leq g_off. The rule (retake the restart once as one L-stable step
# across the impulse) needs a roundoff-accepted Newton solve at h_lo: it fires
# at g_off <= 1e-8 (node >= 1.8e8 V) and not at g_off >= 1e-7 (tau >= 5e-11 s),
# where the mode is resolved and its energy booked (trbdf2 +1.5e-3 / +1.8e-3
# rel at rtol 1e-8 / defaults; 2.1.0 -1.7e-2). Where it fires the sampled
# loss reads E_S = 4.0e-2 J against 8.2e-4 J (the post-edge probe sample at
# 1.9e9 V starts the jump step's trapezoid): the strict xfail below.
_TE = 2e-4


def _cut_set_goff(goff, **kw):
    from pulsim.losses import device_loss_summary
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "v", "gnd", 10.0)
    b.add_resistor("R", "v", "a", 1.0)
    b.add_saturable_inductor("L1", "a", "b", 1e-3, 1e6, 1e-4)
    b.add_inductor("L2", "b", "gnd", 1e-3)
    b.add_switch("S", "b", "gnd", 1e3, goff)
    n, k = b.graph.num_switches, b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, t < _TE)
        return m
    r = _sim(b, fn, t_end=_TE + 2e-6, **kw)
    t = np.asarray(r.times)
    j = int(np.searchsorted(t, _TE))
    j = j if t[j] == _TE else j - 1
    di = np.asarray(r.i("L1"))[j] - np.asarray(r.i("L2"))[j]
    tau = 0.5e-3 * goff
    e = [x for x in device_loss_summary(b, r, switch_fn=fn, t_from=_TE,
                                        t_to=_TE + min(40 * tau, 1e-6))
         if x["name"] == "S"][0]["E_total"]
    return e, 0.5 * 0.5e-3 * di ** 2, r._trbdf2_stats["n_forced_accepts"]


@pytest.mark.parametrize("goff", [1e-7, 1e-6, 1e-5])
@pytest.mark.parametrize("kw", [{}, {"rtol": 1e-8}, {"integrator": "esdirk43"}],
                         ids=["trbdf2", "trbdf2-rtol1e-8", "esdirk43"])
def test_jump_rule_does_not_fire_on_a_resolvable_lossy_mode(goff, kw):
    e, truth, n_forced = _cut_set_goff(goff, **kw)
    assert n_forced == 0
    # esdirk43's longer steps: the sampled line of g_off v^2 reads +1.8e-2
    # (2.1.0 +2.4e-2); trbdf2 +1.5e-3 .. +1.8e-3 (2.1.0 -1.7e-2)
    assert abs(e / truth - 1) < (2.5e-2 if kw.get("integrator") else 2.5e-3)


@pytest.mark.xfail(strict=True, reason="open: where the jump rule fires (g_off 1e-9, "
                   "node 1.9e9 V) the sampled loss of S reads 4.0e-2 J against "
                   "1/2 Leq di^2 = 8.2e-4 J (2.1.0: 9.4e-5 J, or NewtonRoundoffBudget"
                   "Exceeded at rtol 1e-8); the jump's energy is not routed as a "
                   "re-initialisation record yet")
def test_jump_rule_books_the_impulse_energy():
    e, truth, n_forced = _cut_set_goff(1e-9)
    assert n_forced == 1
    assert abs(e / truth - 1) < 1e-2
