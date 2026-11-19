"""Snapshot and resume on the exact engine's default 'auto' integrator
(PEDSimulatorAuto: per mode the exact step, else RK45 or BDF2 by stiffness;
add-exact-propagator-strategy task 20.2, session 26).

Its committed end instant travels in the snapshot's controller block (kind 3,
pickle field 14) as `AutoResumeState` (dsed/scheduler_auto.hpp): t, the
reduced state, the integrated mask, the numeric path's next step and PI
memory, the accepted-step phase mod store_every, the fast samples ahead, the
BDF2 history and which path took the last step (its landing rule). A resumed
run lands first what the uninterrupted one did at that instant (a gate edge
the first run left at its t_end). The payload's first value (101) tells it
from the event-driven integrator's: each integrator continues only its own
(refused by name otherwise). Split = uninterrupted at 1e-12 here (CI); bit
for bit on the machine that measured it where the split is a common instant.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import pulsim as p


def _chopper():
    """A diode-free RL chopper: engine='dsed' runs it on integrator 'auto'."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 48.0)
    b.add_switch("Q1", "in", "sw", 1e3, 1e-9)
    b.add_resistor("Rfw", "sw", "gnd", 2.0)
    b.add_inductor("L", "sw", "out", 200e-6)
    b.add_resistor("R", "out", "gnd", 5.0)
    b.add_capacitor("C", "out", "gnd", 10e-6)
    fn = p.make_pwm_switch_fn(50e3, 0.4, b.switch_index_of("Q1"), b.graph.num_switches)
    return b, fn


def _run(t_end, **kw):
    b, fn = _chopper()
    return p.simulate(b, t_end, switch_fn=fn, engine="dsed", **kw)


def _common(u, r, t_split):
    tu, xu = np.asarray(u.times), np.asarray(u.states)
    tr, xr = np.asarray(r.times), np.asarray(r.states)
    common = np.intersect1d(tu[tu > t_split], tr)
    return common, xu[np.searchsorted(tu, common)], xr[np.searchsorted(tr, common)]


@pytest.mark.parametrize("t_split", [100e-6, 108e-6, 113.7e-6],
                         ids=["edge_on", "edge_off", "mid_interval"])
def test_a_resumed_auto_run_continues_the_uninterrupted_one(t_split):
    t_end = 300e-6
    u = _run(t_end)
    assert u.final_snapshot.valid
    a = _run(t_split)
    snap = pickle.loads(pickle.dumps(a.final_snapshot))
    assert snap.controller_state[0] == 3 and snap.controller_state[1][0] == 101.0
    r = _run(t_end, resume_from=snap)
    assert r.times[0] == pytest.approx(t_split, rel=1e-15)
    common, xu, xr = _common(u, r, t_split)
    assert common.size > 10
    scale = max(1.0, float(np.abs(xu).max()))
    np.testing.assert_allclose(xr, xu, rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(r.final_snapshot.x, u.final_snapshot.x, rtol=0,
                               atol=1e-12 * scale)
    assert r.final_snapshot.controller_state[1][0] == 101.0


def test_row_zero_of_an_edge_resume_is_the_uninterrupted_runs_landing_row():
    u = _run(200e-6)
    a = _run(100e-6)
    r = _run(200e-6, resume_from=a.final_snapshot)
    i = int(np.searchsorted(np.asarray(u.times), 100e-6))
    np.testing.assert_allclose(np.asarray(r.states)[0], np.asarray(u.states)[i],
                               rtol=0, atol=1e-12 * 48.0)


def test_each_integrator_continues_only_its_own_snapshot():
    auto = _run(100e-6)
    with pytest.raises(p.UnsupportedError) as ei:
        _run(200e-6, integrator="rk45", resume_from=auto.final_snapshot)
    assert ei.value.feature == "resume_from" and "integrator='auto'" in str(ei.value)
    ped = _run(100e-6, integrator="rk45")
    with pytest.raises(p.UnsupportedError) as ei:
        _run(200e-6, integrator="auto", resume_from=ped.final_snapshot)
    assert ei.value.feature == "resume_from"
    # unset: each resumes on the integrator that wrote it
    r1 = _run(200e-6, resume_from=auto.final_snapshot)
    r2 = _run(200e-6, resume_from=ped.final_snapshot)
    assert r1.final_snapshot.controller_state[1][0] == 101.0
    assert r2.final_snapshot.controller_state[1][0] == 1.0


def test_the_auto_snapshot_refusal_names_output_rtol_as_the_cause():
    auto = _run(100e-6)
    with pytest.raises(p.UnsupportedError) as ei:
        _run(200e-6, output_rtol=1e-5, resume_from=auto.final_snapshot)
    assert ei.value.feature == "resume_from" and "from output_rtol" in str(ei.value)
