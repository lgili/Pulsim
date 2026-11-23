"""Snapshot and resume on the exact engine (add-exact-propagator-strategy,
session 24).

The exact engine's state at a committed instant is (t, the reduced state, the
integrated mask, the diode latch) plus its schedule position: gate edges and
source corners are asked of their oracles at every step (no cursor), control
ticks carry their integer index (snapshot fields 15-16), and the rest — the
last time event, the last predicate fired, the fast-sample plan, the chatter
guard, the numeric path's step memory — travels in the snapshot's controller
block with kind 3 (pickle field 14; no new field, 19 fields as before). The
loss tracker's state is field 17.

A split at an instant the uninterrupted run also stops at (a gate edge, a
located diode event) continues it bit for bit on this machine; CI checks it at
1e-12 relative (different platforms, same code path). A split mid-interval
takes a different step grid after it (the exact propagator is a semigroup only
to rounding): the runs agree to rounding, checked at 1e-12 of the state scale
at the common sample instants. Another engine's snapshot resumes where its
full state is one state of the mode, decided at t as at a start; the exact
engine's snapshot is refused by name on pwl and trbdf2 (no companion history).
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest

import pulsim as p

from .test_device_losses_kernel import _switch_names, _vsi
from .test_trbdf2_rounding_fuzz import _bridge


def _buck(r=10.0):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("Q1", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6 if r < 50 else 10e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", r)
    fn = p.make_pwm_switch_fn(100e3, 0.5, b.switch_index_of("Q1"), b.graph.num_switches)
    return b, fn


def _ccm():
    return _buck(10.0)


def _dcm():
    return _buck(200.0)


def _bridge_case():
    return _bridge([0.0] * 6), None


def _run(make, t_end, losses=False, **kw):
    b, fn = make()
    if losses:
        rng = np.random.default_rng(7)
        ax = dict(v_axis=[0.0, 400.0, 800.0], i_axis=[0.0, 30.0, 60.0], tj_axis=[25.0, 150.0])
        e_on = p.LossTable(**ax, energy=rng.uniform(1e-3, 5e-3, (3, 3, 2)))
        e_off = p.LossTable(**ax, energy=rng.uniform(1e-3, 5e-3, (3, 3, 2)))
        names = _switch_names(b) if make is _vsi else ["Q1", "D1"]
        for n in names:
            b.set_device_losses(n, Tj=125.0, E_on=e_on, E_off=e_off)
    # legacy subject: continuity at common instants needs the old grid on both runs (a run resumed
    # under the release default output_rtol=1e-5 samples its own grid: test_exact_output_rule_combinations)
    kw.setdefault("output_rtol", 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end, switch_fn=fn, engine="exact", **kw)


def _split(make, t_end, t_split, **kw):
    u = _run(make, t_end, **kw)
    a = _run(make, t_split, **kw)
    snap = pickle.loads(pickle.dumps(a.final_snapshot))   # through the pickle
    r = _run(make, t_end, resume_from=snap, **kw)
    return u, a, r


def _common(u, r, t_split):
    # the split instant's own row is the uninterrupted run's left limit when a
    # located event ended the first run (its snapshot is the right limit)
    tu, xu = np.asarray(u.times), np.asarray(u.states)
    tr, xr = np.asarray(r.times), np.asarray(r.states)
    common = np.intersect1d(tu[tu > t_split], tr)
    iu = np.searchsorted(tu, common)
    ir = np.searchsorted(tr, common)
    return common, xu[iu], xr[ir]


def _dcm_diode_event():
    """The DCM buck's first diode turn-off after 105 us (a located event)."""
    u = _run(_dcm, 110e-6)
    et = np.asarray(u.event_times)
    return float(et[(et > 105e-6) & (et < 110e-6)][0])


CASES = [
    # (name, circuit, t_end, split instant); 'vsi_at_edge' and the DCM edge
    # are gate edges (an SPWM edge at 1 ms, the PWM's turn-off at 105 us)
    ("buck_ccm_at_edge", _ccm, 200e-6, 100e-6),
    ("buck_ccm_mid", _ccm, 200e-6, 102.3e-6),
    ("buck_dcm_at_edge", _dcm, 200e-6, 105e-6),
    ("buck_dcm_mid", _dcm, 200e-6, 103.7e-6),
    ("buck_dcm_at_diode_event", _dcm, 200e-6, None),
    ("vsi_at_edge", _vsi, 2.0e-3, 1.0e-3),
    ("vsi_mid", _vsi, 2.0e-3, 1.0137e-3),
]


@pytest.mark.parametrize("name,make,t_end,ts", CASES, ids=[c[0] for c in CASES])
def test_a_resumed_run_continues_the_uninterrupted_one(name, make, t_end, ts):
    if ts is None:
        ts = _dcm_diode_event()
    u, a, r = _split(make, t_end, ts)
    assert a.final_snapshot.valid and a.final_snapshot.t == pytest.approx(ts, rel=1e-15)
    assert a.final_snapshot.controller_state[0] == 3
    assert len(a.final_snapshot.__getstate__()) == 19   # no new pickle field
    assert np.asarray(r.times)[0] == a.final_snapshot.t
    common, xu, xr = _common(u, r, ts)
    # the step grids agree after the split's next gate edge (at once at an event)
    assert len(common) >= 0.5 * len(r.times), name
    scale = np.max(np.abs(np.asarray(u.states)))
    np.testing.assert_allclose(xr, xu, rtol=1e-12, atol=1e-12 * scale)
    # t_end: the resumed run's end snapshot is the uninterrupted run's
    assert r.final_snapshot.t == u.final_snapshot.t
    np.testing.assert_allclose(np.asarray(r.final_snapshot.x),
                               np.asarray(u.final_snapshot.x), rtol=1e-12,
                               atol=1e-12 * scale)
    eu = np.asarray(u.event_times)
    er = np.asarray(r.event_times)
    np.testing.assert_allclose(er[er > ts], eu[eu > ts], rtol=0, atol=1e-15)


def test_a_bridge_split_mid_interval_continues_to_its_commutation_accuracy():
    """No gate edge resynchronises the step grid after a mid-interval split
    (the bridge is driven by a sine): the two runs differ by rounding to the
    end, and the third commutation (D3's turn-off, a flat crossing that float64
    resolves only to ~1e-12 s, benchmarks/BRIDGE_GROUND_TRUTH.md) moves by
    5e-13 s on this machine — within the engine's 1e-10 s commutation accuracy
    against the mpmath truth (test_exact_bridge_ground_truth)."""
    ts = 0.0123
    u, a, r = _split(_bridge_case, 0.02, ts)
    scale = np.max(np.abs(np.asarray(u.states)))
    np.testing.assert_allclose(np.asarray(r.final_snapshot.x),
                               np.asarray(u.final_snapshot.x), rtol=1e-12,
                               atol=1e-12 * scale)
    eu = np.asarray(u.event_times)
    er = np.asarray(r.event_times)
    assert len(er) == len(eu[eu > ts]) == 3
    np.testing.assert_allclose(er, eu[eu > ts], rtol=0, atol=1e-10)


@pytest.mark.parametrize("make,t_end,ts", [(_ccm, 202.5e-6, 100e-6), (_ccm, 202.5e-6, 102.3e-6),
                                           (_vsi, 2.03e-3, 1.0137e-3)],
                         ids=["buck_at_edge", "buck_mid", "vsi_mid"])
def test_the_loss_ledgers_continue_through_a_resume(make, t_end, ts):
    """Pickle field 17 (the loss tracker's state) carried: the per-device
    counts equal the uninterrupted run's, the energies to rounding."""
    u, a, r = _split(make, t_end, ts, losses=True)
    assert len(a.final_snapshot.__getstate__()[17]) > 0
    names = _switch_names(None) if make is _vsi else ["Q1", "D1"]
    for n in names:
        eu, er = u.device_energy(n), r.device_energy(n)
        assert (er["n_on"], er["n_off"]) == (eu["n_on"], eu["n_off"]), n
        for k in ("E_on", "E_off", "E_cond_electrical"):
            assert er[k] == pytest.approx(eu[k], rel=1e-12, abs=1e-18), (n, k)


def test_a_closed_loop_resumed_on_its_tick_continues_the_uninterrupted_run():
    from .test_control_graph_queue_resume import SPLITS, T2, _duty
    from .test_control_graph_queue_resume import _run as crun
    from .test_control_graph_queue_resume import _split as csplit
    # legacy subject: the old output grid on both runs (see _run)
    (_b, loop_f), rf = crun("exact", T2, output_rtol=0)
    loop_s, r2 = csplit("exact", SPLITS["on_tick"], pickled=True, fresh=True, output_rtol=0)
    np.testing.assert_allclose(_duty(loop_s), _duty(loop_f), rtol=1e-12, atol=0)
    tf, vf = np.asarray(rf.times), np.asarray(rf.v("out"))
    t2, v2 = np.asarray(r2.times), np.asarray(r2.v("out"))
    j = int(np.searchsorted(tf, t2[0]))
    # bit for bit on this machine; CI (another platform's libm) at 1e-12
    np.testing.assert_allclose(t2, tf[j:], rtol=1e-12, atol=0)
    np.testing.assert_allclose(v2, vf[j:], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("eng,kw,bound", [("pwl", dict(dt=1e-8), 1e-3),
                                          ("trbdf2", dict(rtol=1e-10, atol=1e-12), 1e-6)])
def test_another_engines_snapshot_resumes_on_the_exact_engine(eng, kw, bound):
    """A pwl / trbdf2 snapshot whose full state is one state of its mode:
    reduced under its stamped mask, the modes decided at t as at a start. The
    continuation differs from the exact run by the first engine's own error at
    the split (pwl dt=1e-8: 6e-5 of scale, trbdf2 1e-10: 1e-8, measured)."""
    b, fn = _ccm()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = p.simulate(b, 100e-6, switch_fn=fn, engine=eng, **kw)
        b2, fn2 = _ccm()
        r = p.simulate(b2, 200e-6, switch_fn=fn2, engine="exact", resume_from=a.final_snapshot)
    u = _run(_ccm, 200e-6)
    xr, xu = np.asarray(r.final_snapshot.x), np.asarray(u.final_snapshot.x)
    assert np.max(np.abs(xr - xu)) <= bound * np.max(np.abs(xu))
    assert np.asarray(r.times)[0] == a.final_snapshot.t


def test_a_resume_that_would_relabel_time_is_refused_by_name():
    a = _run(_ccm, 100e-6)
    with pytest.raises(p.UnsupportedError) as ei:
        _run(_ccm, 200e-6, resume_from=a.final_snapshot, t_start=50e-6)
    assert ei.value.feature == "resume_from"


def test_controllers_that_cannot_continue_the_snapshots_tick_grid_are_refused():
    """A snapshot written without control ticks, resumed with a closed loop:
    its ticks would restart at the snapshot's t, off the first run's grid."""
    from .test_control_graph_queue_resume import _loop
    b, loop = _loop()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = p.simulate(b, 37.3e-5, engine="exact", switch_fn=loop.switch_fn,
                       integrator="rk45")
        b2, loop2 = _loop()
        with pytest.raises(p.UnsupportedError) as ei:
            p.simulate(b2, 60e-5, engine="exact", switch_fn=loop2.switch_fn,
                       closed_loops=[loop2], resume_from=a.final_snapshot)
    assert ei.value.feature == "resume_from"


def _rc():
    """A diode-free RC charge: engine='dsed' runs it on integrator 'auto'."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 5.0)
    b.add_resistor("R", "in", "out", 1e3)
    b.add_capacitor("C", "out", "gnd", 1e-6)
    return b


def test_a_run_on_a_route_that_keeps_no_snapshot_names_why_at_resume():
    """'bdf2' and a fresh t_start run keep no snapshot: the resume is refused
    with the cause and the workaround, not as a bare "not valid". ('auto'
    writes one since session 26: test_exact_auto_resume.)"""
    for kw, word in ((dict(integrator="bdf2"), "integrator='bdf2'"),
                     (dict(integrator="rk45", t_start=1e-4), "t_start")):
        a = p.simulate(_rc(), 2e-3, engine="dsed", **kw)
        assert not a.final_snapshot.valid and word in a.final_snapshot.invalid_reason
        with pytest.raises(p.UnsupportedError) as ei:
            p.simulate(_rc(), 4e-3, engine="dsed", resume_from=a.final_snapshot)
        assert ei.value.feature == "resume_from" and word in str(ei.value)
    # the workaround named: integrator='rk45' writes one, an unset integrator resumes on it
    a = p.simulate(_rc(), 2e-3, engine="dsed", integrator="rk45")
    r = p.simulate(_rc(), 4e-3, engine="dsed", resume_from=a.final_snapshot)
    u = p.simulate(_rc(), 4e-3, engine="dsed", integrator="rk45")
    np.testing.assert_allclose(r.final_snapshot.x, u.final_snapshot.x, rtol=1e-12)


def test_an_explicit_auto_that_would_not_continue_the_snapshot_is_refused():
    # legacy subject: integrator 'auto' on the old output grid (output_rtol=0)
    a = p.simulate(_rc(), 2e-3, engine="dsed", integrator="rk45", output_rtol=0)
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(_rc(), 4e-3, engine="dsed", integrator="auto", resume_from=a.final_snapshot,
                   output_rtol=0)
    assert ei.value.feature == "resume_from" and "integrator='auto'" in str(ei.value)
    # where 'auto' itself routes to the event-driven integrator (diodes), it runs
    s = _run(_ccm, 100e-6)
    r = _run(_ccm, 200e-6, resume_from=s.final_snapshot, integrator="auto")
    u = _run(_ccm, 200e-6)
    np.testing.assert_allclose(r.final_snapshot.x, u.final_snapshot.x, rtol=1e-12, atol=1e-12)


def test_the_exact_snapshot_carries_the_circuits_layout_and_refuses_another_circuit():
    """As the pwl / trbdf2 snapshots: the device-state layout (the linear
    companions' count: L and C) and its signature; a resume on a circuit with
    another layout is refused naming the block."""
    a = _run(_ccm, 100e-6)
    s = a.final_snapshot
    assert list(s.dev_layout)[0] == 2 and s.layout_signature != 0
    b = p.CircuitBuilder()   # the same state count, one capacitor more and one inductor less
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("Q1", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_resistor("Rs", "sw", "vout", 1.0)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_capacitor("C2", "sw", "gnd", 1e-9)
    b.add_capacitor("C3", "vin", "sw", 1e-9)
    b.add_resistor("RL", "vout", "gnd", 10.0)
    fn = p.make_pwm_switch_fn(100e3, 0.5, b.switch_index_of("Q1"), b.graph.num_switches)
    with pytest.raises(ValueError, match="device-state block"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p.simulate(b, 200e-6, switch_fn=fn, engine="dsed", resume_from=s)
    assert pickle.loads(pickle.dumps(s)).layout_signature == s.layout_signature


@pytest.mark.parametrize("make", [_ccm, _dcm], ids=["ccm", "dcm"])
def test_what_depends_on_the_runs_end_time_today(make):
    """Pins what the exact kernel reads of t_end (DECISIONS session 25): each
    mode's eigenpairs are refined and its tier chosen over the run horizon
    H = max(dt_max, t_end), and the last schedule interval's horizon is capped
    at t_end. Halving t_end changes the stated defect only: the tier, the
    refinement and the balancing are the same and the states at the common
    instants agree (bit-equal on the 81 scoreboard circuits, measured). A
    split run therefore builds its modes as the uninterrupted one."""
    u = _run(make, 200e-6)
    h = _run(make, 100e-6)

    def build(r):
        return {tuple(m["mask"]): (m["tier"], m["refined"], m["balanced"])
                for m in r.exact["masks"]}
    bu, bh = build(u), build(h)
    assert bh.keys() <= bu.keys() and all(bh[k] == bu[k] for k in bh)
    tu, th = np.asarray(u.times), np.asarray(h.times)
    _, iu, ih = np.intersect1d(tu, th, return_indices=True)
    assert len(ih) > 0.9 * len(th)
    xu, xh = np.asarray(u.states), np.asarray(h.states)
    np.testing.assert_allclose(xh[ih], xu[iu], rtol=0, atol=1e-12 * np.abs(xu).max())


def test_auto_resumes_an_exact_snapshot_on_the_route_that_wrote_it():
    """release: engine='auto' without dt runs the exact engine, so the resume
    leg of the same call takes the route that wrote the snapshot (it was
    refused by name while 'auto' never routed there). A call that leaves the
    exact route (a dt: the fixed step) is still refused by name."""
    def leg(t_end, **kw):
        b, fn = _ccm()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return p.simulate(b, t_end, switch_fn=fn, **kw)
    a = leg(100e-6)
    assert a.engine_used == "dsed"
    r = leg(200e-6, resume_from=pickle.loads(pickle.dumps(a.final_snapshot)))
    u = leg(200e-6)
    assert r.engine_used == "dsed" and np.asarray(r.times)[0] == a.final_snapshot.t
    xr, xu = np.asarray(r.final_snapshot.x), np.asarray(u.final_snapshot.x)
    assert np.max(np.abs(xr - xu)) <= 1e-9 * np.max(np.abs(xu))
    with pytest.raises(p.UnsupportedError, match="written by the exact engine") as ei:
        leg(200e-6, dt=1e-8, resume_from=a.final_snapshot)
    assert (ei.value.feature, ei.value.engine) == ("resume_from", "auto")
    # a refusal of the exact engine on this route stands as it is (no other
    # engine continues its snapshot): here, a resume that would relabel time
    with pytest.raises(p.UnsupportedError) as ei:
        leg(200e-6, resume_from=a.final_snapshot, t_start=50e-6)
    assert (ei.value.feature, ei.value.engine) == ("resume_from", "dsed")
    assert "written by the exact engine" not in str(ei.value)


def test_auto_falls_back_when_the_exact_routes_plan_refuses():
    """release: 'auto' without dt takes the exact engine; a refusal by name in
    its PLAN (controllers with t_start) lands the run on the variable step, as
    a refusal in the run does — the same samples as engine='trbdf2' asked by
    name."""
    from .test_tstart_time_base import _buck as tick_buck

    def run(engine):
        b, kw, _ = tick_buck()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return p.simulate(b, t_end=6e-4, t_start=4e-4, engine=engine, rtol=1e-8, **kw)
    a, t = run("auto"), run("trbdf2")
    assert a.engine_used == "trbdf2"
    np.testing.assert_array_equal(np.asarray(a.times), np.asarray(t.times))
    np.testing.assert_array_equal(np.asarray(a.states), np.asarray(t.states))
    with pytest.raises(p.UnsupportedError, match="t_start != 0"):
        run("exact")
