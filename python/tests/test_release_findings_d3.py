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
