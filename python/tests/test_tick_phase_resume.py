"""Tick phase in snapshots (add-unified-event-queue): a run's final
snapshot carries, per tick source, its grid's origin and next index, and a
resume passes the origins back, so the resumed ticks fall where the
uninterrupted run's do — on trbdf2 too, which used to refuse a graph
resumed between two of its ticks."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import pulsim as p

T1, T2 = 0.537e-3, 1e-3          # T1 is on neither group's grid


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    b.add_switch("Q", "in", "sw", 1e2, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e2, 1e-9)
    b.add_inductor("L", "sw", "out", 1e-3)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", 10.0)
    return b


def _graph(b):
    ch = p.MixedDomainBlockChain()
    ch.add("i1", p.Integrator(gain=1.0), inputs=dict(x="node:out", dt="dt"),
           output="y1", rate=p.Rate(1e-4, offset=2e-5))
    ch.add("i2", p.Integrator(gain=1.0), inputs=dict(x="node:out", dt="dt"),
           output="y2", rate=p.Rate(3e-4, offset=5e-5))
    ch.record_channel("y1")
    ch.record_channel("y2")
    return ch, ch.make_step_observer(b)


def _sw(b):
    return p.make_pwm_switch_fn(10e3, 0.4, b.switch_index_of("Q"),
                                b.graph.num_switches)


ENGINES = [("pwl", dict(dt=1e-6)), ("trbdf2", dict(rtol=1e-9))]


@pytest.mark.parametrize("engine,kw", ENGINES)
def test_two_rate_groups_with_offsets_resume_on_their_grid(engine, kw):
    bf = _buck()
    chf, obsf = _graph(bf)
    p.simulate(bf, T2, engine=engine, step_observer=obsf, switch_fn=_sw(bf), **kw)
    b = _buck()
    ch, obs = _graph(b)
    first = p.simulate(b, T1, engine=engine, step_observer=obs,
                       switch_fn=_sw(b), **kw)
    p.simulate(b, T2, engine=engine, step_observer=obs, switch_fn=_sw(b),
               resume_from=first.final_snapshot, **kw)
    tf = chf._cxx_chain.get_recording_times()
    ts = ch._cxx_chain.get_recording_times()
    if engine == "trbdf2":   # every tick instant bit for bit (the queue's)
        assert ts.tobytes() == tf.tobytes()
    else:   # pwl graphs tick on the grid's labels: t1 + n*dt against n*dt
        np.testing.assert_allclose(ts, tf, rtol=0, atol=1e-15)
    assert len(ts) == 10 + 4              # 2e-5 + k*1e-4: 10; 5e-5 + k*3e-4: 4
    for name in ("y1", "y2"):
        yf = chf._cxx_chain.get_channel_history(name)
        ys = ch._cxx_chain.get_channel_history(name)
        if engine == "pwl":
            assert ys.tobytes() == yf.tobytes()
        else:   # a trbdf2 run's steps depend on its span: to tolerance
            np.testing.assert_allclose(ys, yf, rtol=1e-6)


@pytest.mark.parametrize("engine,kw", ENGINES)
def test_the_snapshot_carries_each_sources_origin_and_next_k(engine, kw):
    b = _buck()
    seen = []

    def tick(t, x):
        seen.append(t)

    first = p.simulate(b, T1, engine=engine, step_observer=tick,
                       controller_period=1e-4, switch_fn=_sw(b), **kw)
    snap = first.final_snapshot
    assert list(snap.tick_origin) == [0.0]
    assert list(snap.tick_next_k) == [6]      # 0, 1e-4, ..., 5e-4 < T1
    n1 = len(seen)
    p.simulate(b, T2, engine=engine, step_observer=tick,
               controller_period=1e-4, switch_fn=_sw(b),
               resume_from=snap, **kw)
    # the resumed ticks stay on k*1e-4 (before: T1 + k*1e-4)
    assert n1 == 6 and len(seen) == 10
    np.testing.assert_allclose(seen[6:], [k * 1e-4 for k in range(6, 10)],
                               rtol=0, atol=1e-12)


def test_pickle_version_5_keeps_the_tick_phase_and_reads_version_3():
    b = _buck()
    first = p.simulate(b, T1, engine="trbdf2", step_observer=lambda t, x: None,
                       controller_period=1e-4, switch_fn=_sw(b))
    snap = first.final_snapshot
    back = pickle.loads(pickle.dumps(snap))
    assert list(back.tick_origin) == list(snap.tick_origin) == [0.0]
    assert list(back.tick_next_k) == list(snap.tick_next_k) == [6]
    state = snap.__getstate__()
    assert state[0] == "SolverSnapshot" and state[1] == 5 and len(state) == 18
    cls = type(snap)
    v3 = cls.__new__(cls)
    v3.__setstate__(("SolverSnapshot", 3) + tuple(state[2:13]))
    assert list(v3.tick_origin) == [] and v3.t == snap.t
    bad = cls.__new__(cls)
    with pytest.raises(RuntimeError, match="reads version 3"):
        bad.__setstate__(("SolverSnapshot", 4) + tuple(state[2:16]))
    # 13 fields read as version 4 with the grid at 12: a bool there is a
    # malformed tuple, refused by name (not pybind's cast error)
    with pytest.raises(RuntimeError, match="SolverSnapshot: malformed"):
        bad.__setstate__(("SolverSnapshot", 4) + tuple(state[2:13]))


def test_a_snapshot_without_the_tick_grid_is_still_refused_off_grid():
    b = _buck()
    ch, obs = _graph(b)
    first = p.simulate(b, T1, engine="trbdf2", step_observer=obs,
                       switch_fn=_sw(b))
    snap = first.final_snapshot
    snap.tick_origin = []                    # an older or hand-built snapshot
    with pytest.raises(p.UnsupportedError, match="between two"):
        p.simulate(b, T2, engine="trbdf2", step_observer=obs,
                   switch_fn=_sw(b), resume_from=snap)


def _count_split(t1, t2, T, monitor, second):
    """Observer calls of one trbdf2 run to t2 and of a trbdf2 run split at
    t1 and resumed on `second`."""
    def run(t_end, seen, engine="trbdf2", **extra):
        b = _buck()
        def fn(t, x):
            seen.append(t)
        obs = dict(step_observer=p.control.Monitor(fn, T)) if monitor else \
            dict(step_observer=fn, controller_period=T)
        kw = dict(rtol=1e-6) if engine == "trbdf2" else dict(dt=1e-7)
        return p.simulate(b, t_end, engine=engine, switch_fn=_sw(b), **obs, **kw,
                          **extra)
    full, split = [], []
    run(t2, full)
    first = run(t1, split)
    run(t2, split, engine=second, resume_from=first.final_snapshot,
        t_start=first.final_snapshot.t)
    return full, split


@pytest.mark.parametrize("second", ["trbdf2", "pwl"])
@pytest.mark.parametrize("monitor", [True, False])
def test_an_instant_just_below_the_split_is_called_once(monitor, second):
    """Batch 14a review: an instant within h_min below the split fired at
    the end of the first run (monitors had no end cut) and again at the
    start of the resumed run; a control tick in the gap between the two
    runs' start tolerances could be fired by neither (the resume now takes
    the snapshot's tick_next_k; that gap was not reproduced on these
    circuits, the test pins the count)."""
    T = 1e-5
    t1 = 37 * T + 2e-15        # h_min of the first run ~3.7e-15 s, of the second ~1e-16
    full, split = _count_split(t1, t1 + T * 1.5, T, monitor, second)
    assert len(split) == len(full), (len(split), len(full))
