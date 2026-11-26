"""Release findings, group D3: TR-BDF2's restart at a gate edge and at the
run start (openspec/changes/update-number-changing-release/fixes/group-d3.md)."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

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
    the SDIRK(2,2) step from the committed state, as a diode event does."""
    from test_release_findings_de import _run_coss
    r = _run_coss(engine="trbdf2")
    st = r._trbdf2_stats
    assert st["n_gate_events"] == 40
    assert st["n_sdirk_steps"] >= st["n_gate_events"]


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


def _satind():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "a", "gnd", 0.0, 20.0, 1e3)
    b.add_resistor("R", "a", "b", 1.0)
    b.add_saturable_inductor("Ls", "b", "gnd", 1e-3, 2.0, 1e-5)
    return b, p.make_combined_switch_fn(b.graph.num_switches, [])


def test_consistent_start_takes_no_restart_step():
    """A sine from zero into R + a saturable L starts consistent (v_L(0) = 0
    is the slot): the first step is TR-BDF2's own, not the first-order
    restart step the stateful kind forced on every start."""
    b, fn = _satind()
    for kw in ({}, {"integrator": "esdirk43"}):
        st = _sim(b, fn, t_end=1e-4, **kw)._trbdf2_stats
        assert st["n_sdirk_steps"] == 0


def test_inconsistent_start_keeps_the_restart_step():
    """A current-driven nonlinear capacitor (F-S-1) and a source switched onto
    a zero-state gated buck with a Lauritzen diode start inconsistent: the
    restart step stays (without it the buck's first TR-BDF2 step was rejected
    down to 4e-13 s)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 0.0)
    b.add_resistor("R", "a", "n", 1e-3)
    b.add_nonlinear_capacitor("C", "n", "gnd", 1e-9, 25.0, 0.0)
    b.add_current_source("I", "gnd", "n", 10.0)
    st = _sim(b, p.make_combined_switch_fn(b.graph.num_switches, []), t_end=1e-9)._trbdf2_stats
    assert st["n_sdirk_steps"] >= 1
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
