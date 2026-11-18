"""G26: controller state per sweep point and per snapshot.

Every point of a sweep starts each control graph (helper loop, chain
passed as step_observer, C blocks) from the state it had when the sweep
began, so a point equals another point with the same parameters bit for
bit. The caller's Python
controllers are deep-copied per point; a closure over mutable state and
a graph on parallel points are refused by name. Before: point k
continued point k-1's integrator (R = 10, 8, 10, 8 ohm: v_out 5.149,
6.607, 7.716, 7.984 V at 1 ms).

resume_from restores the snapshot's controller state, so a resumed run
equals the uninterrupted one whatever ran in between (before: it
continued whatever the last run left).
"""
from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest

import pulsim as p

DT = 1e-6
GRID = [10.0, 8.0, 10.0, 8.0]          # point 2 repeats point 0, 3 repeats 1


def _buck(R=10.0):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    b.add_switch("Q", "in", "sw", 1e2, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e2, 1e-9)
    b.add_inductor("L", "sw", "out", 1e-3)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", R)
    return b


def _loop(b):
    return p.bind_pi_to_switch(
        b, pi=p.PIController(Kp=0.02, Ki=20.0, output_min=0.05,
                             output_max=0.95),
        measured="out", setpoint=12.0, switch="Q", freq=10e3)


def _sweep(form, kw):
    b = _buck()
    first = form == "circuit"
    target = b.build() if first else _buck
    params = {"R.R": GRID} if first else {"R": GRID}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.sweep(target, params=params, t_end=1e-3, dt=DT,
                       verbose=False, **kw(b))


@pytest.mark.parametrize("form", ["factory", "circuit"])
def test_a_serial_sweep_restarts_a_helper_loop_per_point(form):
    j = _buck().node_id_of("out")
    seen = {}

    def kw(b):
        loop = seen.setdefault("loop", _loop(b))
        return dict(closed_loops=[loop], kpi_fn=lambda r, d: {
            "v": float(np.asarray(r.states)[-1, j]),
            "duty": float(loop.duty_history[-1][1]),
            "n": float(len(loop.duty_history))})
    s = _sweep(form, kw=kw)
    for k in ("v", "duty"):
        v = s.kpis[k]
        assert v[2] == v[0] and v[3] == v[1], (k, v)
        assert v[0] != v[1]
    # every point records its own ticks only (the history restarts)
    assert set(s.kpis["n"]) == {10.0}


def test_a_serial_sweep_restarts_a_chain_passed_as_step_observer():
    # an integrator driving a source: its state carried over before
    def kw(b):
        ch = p.MixedDomainBlockChain(rate=p.Rate(1e-5))
        ch.add("i", p.Integrator(gain=1e3), inputs=dict(x="node:out",
                                                        dt="dt"),
               output="y")
        obs = ch.make_step_observer(b)
        return dict(step_observer=obs, switch_fn=p.make_pwm_switch_fn(
            10e3, 0.5, b.switch_index_of("Q"), b.graph.num_switches),
            kpi_fn=lambda r, d: {"y": float(ch._cxx_chain.get_channel("y"))})
    s = _sweep("factory", kw=kw)
    v = s.kpis["y"]
    assert v[2] == v[0] and v[3] == v[1] and v[0] != v[1]


class _Hysteresis:
    """A caller's controller object: its observer and gate share state."""

    def __init__(self, n, idx, j):
        self.on, self.n, self.idx, self.j, self.ticks = True, n, idx, j, 0

    def observe(self, t, x):
        self.ticks += 1
        v = x[self.j]
        if v > 12.2:
            self.on = False
        elif v < 11.8:
            self.on = True

    def gate(self, t):
        m = p.SwitchStateMask(self.n)
        m.set(self.idx, self.on)
        return m


@pytest.mark.parametrize("parallel", [False, True])
def test_a_callable_controller_object_is_copied_per_point(parallel):
    holder = {}

    def kw(b):
        c = holder.setdefault("c", _Hysteresis(
            b.graph.num_switches, b.switch_index_of("Q"), b.node_id_of("out")))
        j = b.node_id_of("out")
        return dict(step_observer=c.observe, switch_fn=c.gate,
                    parallel=parallel, backend="thread",
                    kpi_fn=lambda r, d: {"v": float(np.asarray(r.states)[-1, j])})
    s = _sweep("factory", kw=kw)
    v = s.kpis["v"]
    assert v[2] == v[0] and v[3] == v[1] and v[0] != v[1]
    assert holder["c"].ticks == 0          # the caller's object is untouched


def test_a_closure_over_mutable_state_is_refused_in_a_sweep():
    def kw(b):
        duty = [0.5]

        def obs(t, x):
            duty[0] = min(0.9, duty[0] + 1e-6)
        return dict(step_observer=obs, kpi_fn=lambda r, d: {})
    with pytest.raises(p.UnsupportedError) as e:
        _sweep("factory", kw=kw)
    assert e.value.feature == "controller_state" and "step_observer" in str(e.value)


def test_a_pure_observer_function_is_allowed():
    def kw(b):
        return dict(step_observer=lambda t, x: None, switch_fn=p.make_pwm_switch_fn(
            10e3, 0.5, b.switch_index_of("Q"), b.graph.num_switches),
            kpi_fn=lambda r, d: {"n": float(len(r.times))})
    assert _sweep("factory", kw=kw).n_simulations == 4


@pytest.mark.parametrize("form", ["factory", "circuit"])
def test_a_parallel_sweep_refuses_a_shared_graph(form):
    def kw(b):
        return dict(closed_loops=[_loop(b)], parallel=True,
                    kpi_fn=lambda r, d: {})
    with pytest.raises(p.UnsupportedError) as e:
        _sweep(form, kw=kw)
    assert e.value.feature == "controller_state"
    assert "share one controller" in str(e.value)


def test_a_plain_rerun_keeps_g22_carry_over():
    # outside a sweep the caller's loop continues (G22); the sweep's
    # restart does not leak into later plain runs
    b = _buck()
    loop = _loop(b)
    r1 = np.asarray(p.simulate(b, 1e-3, dt=DT, closed_loops=[loop]).states)
    r2 = np.asarray(p.simulate(b, 1e-3, dt=DT, closed_loops=[loop]).states)
    assert r1.tobytes() != r2.tobytes()


# ---------------------------------------------------------------- resume

# pwl resumes mid-period (label time is not exact on a tick, G22);
# trbdf2 ticks at t_start + k*T, so it resumes on a tick
ENGINES = [("pwl", dict(dt=DT), 1.53e-3), ("trbdf2", dict(rtol=1e-9), 1.5e-3)]


@pytest.mark.parametrize("engine,kw,t1", ENGINES)
def test_resume_restores_the_snapshots_loop_whatever_ran_between(engine, kw, t1):
    t2 = 3e-3
    bf = _buck()
    lf = _loop(bf)
    full = p.simulate(bf, t2, engine=engine, closed_loops=[lf], **kw)
    b = _buck()
    lp = _loop(b)
    first = p.simulate(b, t1, engine=engine, closed_loops=[lp], **kw)
    p.simulate(b, 2.2e-3, engine=engine, closed_loops=[lp], **kw)   # between
    second = p.simulate(b, t2, engine=engine, closed_loops=[lp],
                        resume_from=first.final_snapshot, **kw)
    dp = [d for _, d in lp.duty_history]
    df = [d for _, d in lf.duty_history]
    j = b.node_id_of("out")
    xf, xs = np.asarray(full.states), np.asarray(second.states)
    if engine == "pwl":
        assert dp == df
        assert xs[-1, j] == pytest.approx(xf[-1, j], rel=1e-9)
    else:   # a trbdf2 run's steps depend on its span: to tolerance
        assert dp == pytest.approx(df, rel=1e-6)
        assert xs[-1, j] == pytest.approx(xf[-1, j], rel=1e-6)


def test_a_pickled_snapshot_of_a_closed_loop_is_refused():
    b = _buck()
    lp = _loop(b)
    first = p.simulate(b, 1e-3, dt=DT, closed_loops=[lp])
    snap = pickle.loads(pickle.dumps(first.final_snapshot))
    with pytest.raises(p.UnsupportedError) as e:
        p.simulate(b, 2e-3, dt=DT, closed_loops=[lp], resume_from=snap)
    assert e.value.feature == "controller_state" and "pickled" in str(e.value)
    # without controllers the pickled snapshot still resumes
    sw = p.make_pwm_switch_fn(10e3, 0.5, b.switch_index_of("Q"),
                              b.graph.num_switches)
    p.simulate(b, 2e-3, dt=DT, switch_fn=sw, resume_from=snap)


def test_resume_with_other_controllers_is_refused():
    b = _buck()
    first = p.simulate(b, 1e-3, dt=DT, closed_loops=[_loop(b)])
    with pytest.raises(p.UnsupportedError, match="other controllers"):
        p.simulate(b, 2e-3, dt=DT, closed_loops=[_loop(b)],
                   resume_from=first.final_snapshot)


def test_closing_the_loop_on_an_open_loop_snapshot_starts_it_fresh():
    b = _buck()
    sw = p.make_pwm_switch_fn(10e3, 0.4, b.switch_index_of("Q"),
                              b.graph.num_switches)
    first = p.simulate(b, 1e-3, dt=DT, switch_fn=sw)
    lp = _loop(b)
    p.simulate(b, 2e-3, dt=DT, closed_loops=[lp],
               resume_from=first.final_snapshot)
    assert lp.duty_history[0][1] != 0.0 and len(lp.duty_history) == 10


def test_trbdf2_refuses_a_loop_resumed_between_two_ticks():
    # its ticks would fall at t_start + k*T, off the loop's grid (before:
    # ticks at 1.63, 1.73 ms ... instead of 1.6, 1.7 ms, duty 1 % off)
    b = _buck()
    lp = _loop(b)
    first = p.simulate(b, 1.53e-3, engine="trbdf2", closed_loops=[lp])
    with pytest.raises(p.UnsupportedError) as e:
        p.simulate(b, 3e-3, engine="trbdf2", closed_loops=[lp],
                   resume_from=first.final_snapshot)
    assert e.value.feature == "resume_from" and "between two" in str(e.value)
    # auto names the blocker (and asks for a dt, the pwl route)
    with pytest.raises(ValueError, match="between two of its ticks"):
        p.simulate(b, 3e-3, closed_loops=[lp], resume_from=first.final_snapshot)


def test_a_graph_block_over_a_closure_is_refused_in_a_sweep():
    # a PythonBlock running a function over a list: its state is not
    # copied, so the second point is refused instead of continuing
    def kw(b):
        acc = [0.0]

        def f(x):
            acc[0] += 1.0
            return acc[0]
        ch = p.MixedDomainBlockChain(rate=p.Rate(1e-5))
        ch.add("f", p.PythonBlock(f), inputs=dict(x="node:out"), output="y")
        return dict(step_observer=ch.make_step_observer(b),
                    switch_fn=p.make_pwm_switch_fn(
                        10e3, 0.5, b.switch_index_of("Q"), b.graph.num_switches),
                    kpi_fn=lambda r, d: {})
    with pytest.raises(p.UnsupportedError) as e:
        _sweep("factory", kw=kw)
    assert e.value.feature == "controller_state" and "'f'" in str(e.value)


def test_a_graph_and_a_controller_object_sweep_together():
    # before: deep-copying every key also tried the loop and was refused
    seen = {}

    def kw(b):
        loop = seen.setdefault("loop", _loop(b))
        j = b.node_id_of("out")
        return dict(closed_loops=[loop], step_observer=_Hysteresis(
            b.graph.num_switches, b.switch_index_of("Q"), j).observe,
            kpi_fn=lambda r, d: {"v": float(np.asarray(r.states)[-1, j])})
    v = _sweep("circuit", kw=kw).kpis["v"]
    assert v[2] == v[0] and v[3] == v[1]


def test_resume_from_a_snapshot_without_this_loops_state_is_refused():
    b = _buck()
    sw = p.make_pwm_switch_fn(10e3, 0.4, b.switch_index_of("Q"),
                              b.graph.num_switches)
    first = p.simulate(b, 1e-3, dt=DT, switch_fn=sw)
    lp = _loop(b)
    p.simulate(b, 1e-3, dt=DT, closed_loops=[lp])          # the loop has run
    with pytest.raises(p.UnsupportedError, match="carries no state"):
        p.simulate(b, 2e-3, dt=DT, closed_loops=[lp],
                   resume_from=first.final_snapshot)


def test_trbdf2_refuses_a_resume_off_any_groups_grid():
    # groups at 100 us and 300 us; resumed at 0.1 ms the fast group is on
    # its grid, the slow one (next due 0.3 ms) is not
    b = _buck()
    ch = p.MixedDomainBlockChain()
    ch.add("i1", p.Integrator(gain=1.0), inputs=dict(x="node:out", dt="dt"),
           output="y1", rate=p.Rate(1e-4))
    ch.add("i2", p.Integrator(gain=1.0), inputs=dict(x="node:out", dt="dt"),
           output="y2", rate=p.Rate(3e-4))
    obs = ch.make_step_observer(b)
    sw = p.make_pwm_switch_fn(10e3, 0.4, b.switch_index_of("Q"),
                              b.graph.num_switches)
    first = p.simulate(b, 1e-4, engine="trbdf2", step_observer=obs, switch_fn=sw)
    with pytest.raises(p.UnsupportedError, match="between two"):
        p.simulate(b, 1e-3, engine="trbdf2", step_observer=obs, switch_fn=sw,
                   resume_from=first.final_snapshot)
