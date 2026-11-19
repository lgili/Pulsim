"""Control graphs on the kernel's event queue (pwl, control_tick='state')
and closed loops resumed from a pickled snapshot (add-compiled-control-graph
tasks 6.1 / 6.3, DECISIONS G32 / G33).

Identity gates compare bytes on this machine only: a resume is the same
build continuing its own run.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest

import pulsim as p

T = 1e-5
T2 = 60 * T
# on a tick; one pwl step after one (the state-time tick's queue instant is
# the first run's t_end); between two
SPLITS = {"on_tick": 37 * T, "step_after_tick": 37.05 * T,
          "between_ticks": 37.3 * T}
PWL = {"label": dict(dt=5e-7),
       "state": dict(dt=5e-7, control_tick="state", discontinuity_restart=True,
                     enable_substep_state_correction=True)}


def _loop():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 10.0)
    b.add_switch("SH", "vin", "sw", 100.0, 1e-6)
    b.add_diode("D", "gnd", "sw", 100.0, 1e-6)
    b.add_resistor("R_dcr", "sw", "n_l", 0.2)
    b.add_inductor("L1", "n_l", "n_esr", 10e-6)
    b.add_resistor("R_esr", "n_esr", "out", 0.05)
    b.add_capacitor("C1", "out", "gnd", 100e-6)
    b.add_resistor("R", "out", "gnd", 10.0)
    pi = p.PIController(Kp=0.05, Ki=200.0, output_min=0.0, output_max=1.0)
    loop = p.bind_pi_to_switch(b, pi=pi, measured="out", setpoint=5.0,
                               switch="SH", freq=1.0 / T)
    return b, loop


def _run(engine, t_end, made=None, **kw):
    b, loop = made or _loop()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end, engine=engine, switch_fn=loop.switch_fn,
                       closed_loops=[loop], **kw)
    return (b, loop), r


def _duty(loop):
    return np.asarray(list(loop.duty_history))


def _split(engine, t1, pickled, fresh, **kw):
    made, r1 = _run(engine, t1, **kw)
    snap = r1.final_snapshot
    if pickled:
        snap = pickle.loads(pickle.dumps(snap))
    made2 = _loop() if fresh else made
    _, r2 = _run(engine, T2, made=made2, resume_from=snap, **kw)
    return made2[1], r2


@pytest.mark.parametrize("split", sorted(SPLITS))
@pytest.mark.parametrize("mode", sorted(PWL))
@pytest.mark.parametrize("pickled", [False, True])
def test_pwl_closed_loop_resume_is_the_uninterrupted_run(mode, split, pickled):
    kw = PWL[mode]
    (_b, loop_f), rf = _run("pwl", T2, **kw)
    # pickled: new controller objects built the same way take the state
    loop_s, r2 = _split("pwl", SPLITS[split], pickled, fresh=pickled, **kw)
    df, ds = _duty(loop_f), _duty(loop_s)
    assert ds.tobytes() == df.tobytes()
    tf, vf = np.asarray(rf.times), np.asarray(rf.v("out"))
    t2, v2 = np.asarray(r2.times), np.asarray(r2.v("out"))
    j = int(np.searchsorted(tf, t2[0]))
    assert tf[j:].tobytes() == t2.tobytes()
    assert vf[j:].tobytes() == v2.tobytes()


@pytest.mark.parametrize("split", sorted(SPLITS))
def test_trbdf2_resume_from_a_pickle_is_the_in_memory_resume(split):
    kw = dict(rtol=1e-8, atol=1e-12)
    loop_m, rm = _split("trbdf2", SPLITS[split], pickled=False, fresh=False, **kw)
    loop_p, rp = _split("trbdf2", SPLITS[split], pickled=True, fresh=True, **kw)
    assert _duty(loop_p).tobytes() == _duty(loop_m).tobytes()
    assert np.asarray(rp.v("out")).tobytes() == np.asarray(rm.v("out")).tobytes()
    # against the run that did not stop: its steps depend on the span
    (_b, loop_f), rf = _run("trbdf2", T2, **kw)
    np.testing.assert_allclose(_duty(loop_p), _duty(loop_f), rtol=0, atol=1e-6)


def test_state_mode_ticks_the_graph_from_the_kernel_queue():
    (_b, loop), r = _run("pwl", 20 * T, **PWL["state"])
    cxx = loop.step_observer._cxx_chain
    assert cxx.queue_driven
    assert list(r.control_tick_counts) == [20]
    snap = r.final_snapshot
    # one kernel source for the helper's rate group, one step behind its grid
    assert len(snap.tick_origin) == 1
    assert snap.tick_origin[0] == pytest.approx(5e-7, abs=1e-18)
    assert list(snap.tick_next_k) == [int(cxx.group_next_k(1))] == [20]
    (_b, loop), r = _run("pwl", 20 * T, **PWL["label"])
    assert not loop.step_observer._cxx_chain.queue_driven
    assert list(r.final_snapshot.tick_origin) == []


def test_a_reused_loop_restarts_its_queue_grid_on_a_new_run():
    made, r1 = _run("pwl", 20 * T, **PWL["state"])
    _, r2 = _run("pwl", 30 * T, made=made, **PWL["state"])   # time goes back
    assert list(r2.control_tick_counts) == [30]
    assert list(r2.final_snapshot.tick_next_k) == [30]
    # the PI's state carries over (G22); the ticks land on k*T again
    t = np.asarray(list(made[1].duty_history))[20:, 0]
    np.testing.assert_allclose(t, np.arange(30) * T, rtol=0, atol=1e-15)


def test_a_pickled_snapshot_refuses_other_controllers_by_name():
    made, r1 = _run("pwl", 37 * T, dt=5e-7)
    snap = pickle.loads(pickle.dumps(r1.final_snapshot))
    b = made[0]
    pi = p.PIController(Kp=0.05, Ki=200.0, output_min=0.0, output_max=1.0)
    other = p.bind_pi_to_switch(b, pi=pi, measured="out", setpoint=5.0,
                                switch="SH", freq=1.0 / T)
    with pytest.raises(p.UnsupportedError, match="other controllers"):
        p.simulate(b, T2, dt=5e-7, switch_fn=made[1].switch_fn,
                   closed_loops=[made[1], other], resume_from=snap)


def test_graph_state_bytes_round_trip_and_refuse_another_graph():
    _made, r1 = _run("pwl", 10 * T, dt=5e-7)
    cxx = _made[1].step_observer._cxx_chain
    data = cxx.dump_state(cxx.save_state())
    before = cxx.dump_state(cxx.save_state())
    st = cxx.load_state(data)
    assert cxx.dump_state(cxx.save_state()) == before    # the live graph untouched
    assert cxx.dump_state(st) == data
    with pytest.raises(ValueError, match="truncated|another graph"):
        cxx.load_state(data[:-3])
    assert cxx.dump_state(cxx.save_state()) == before
    ch = p.MixedDomainBlockChain()
    ch.add("g", p.Gain(2.0), inputs=dict(x="node:out"), output="y")
    ch.make_step_observer(_made[0], dt=5e-7)
    other = ch._cxx_chain
    mine = other.dump_state(other.save_state())
    with pytest.raises(ValueError, match="another graph"):
        other.load_state(data)
    assert other.dump_state(other.save_state()) == mine    # left intact


class _Acc:
    """A Python block with state in its __dict__."""

    def __init__(self):
        self.total = 0.0

    def __call__(self, x):
        self.total += x
        return self.total


def test_a_python_block_resumes_from_a_pickle():
    def build():
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "a", "gnd", 1.0)
        b.add_resistor("R", "a", "out", 1.0)
        b.add_capacitor("C", "out", "gnd", 1e-4)
        ch = p.MixedDomainBlockChain()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ch.add("acc", p.PythonBlock(_Acc()), inputs=dict(x="node:out"),
                   output="y", rate=p.Rate(1e-5))
        ch.record_channel("y")
        return b, ch, ch.make_step_observer(b)

    def run(made, t_end, **kw):
        b, _ch, obs = made
        return p.simulate(b, t_end, dt=1e-6, step_observer=obs, **kw)

    full = build()
    run(full, 4e-4)
    first = build()
    snap = pickle.loads(pickle.dumps(run(first, 2.05e-4).final_snapshot))
    second = build()
    run(second, 4e-4, resume_from=snap)
    yf = full[1]._cxx_chain.get_channel_history("y")
    ys = second[1]._cxx_chain.get_channel_history("y")
    assert ys.tobytes() == yf.tobytes()
