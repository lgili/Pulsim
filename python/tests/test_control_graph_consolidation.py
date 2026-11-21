"""Task 9.1-9.2 of add-compiled-control-graph: the closed-loop helpers and
the C blocks build control graphs (label time), composed once above the
engine dispatch; the counter gate on those paths.

Label time (docs/control-graph.md): on pwl a tick due at tau runs at the
first observer call whose label t_n >= tau - 1e-9*dt, with that call's
state x_{n-1}, and drives step n (from t_{n-1}). Tick 0 runs at t_start
on x0; tick k >= 1 samples v(kT - h) and holds from kT - h: a one-step
ADVANCE of the nominal grid, not a delay.

P loop on an RC (tau = 1 ms), K_p = 2, set-point 1 V, T = 100 us, step h:
w_0 = 0, w_{k+1} = a_k w_k + (1 - a_k) * 2 (1 - w_k), a_0 = exp(-(T-h)/tau),
a_k = exp(-T/tau) (k >= 1), w_k = v(kT - h). h -> 0 gives the state-time
v[10] = 0.643545722992.
"""

from __future__ import annotations

import math
import sys
import warnings

import numpy as np
import pytest

import pulsim as p

TS, TAU = 1e-4, 1e-3


def w_label(k, h):
    w = 0.0
    for j in range(k):
        a = math.exp(-((TS - h) if j == 0 else TS) / TAU)
        w = a * w + (1.0 - a) * 2.0 * (1.0 - w)
    return w


def test_label_closed_form():
    assert w_label(10, 0.0) == pytest.approx(0.643545722992, abs=1e-12)
    assert w_label(10, 1e-6) == pytest.approx(0.643457839998, abs=1e-12)


def _rc():
    b = p.CircuitBuilder()
    b.add_resistor("R", "u", "v", 1e3)
    b.add_capacitor("C", "v", "gnd", 1e-6)
    return b


def _p_block(b, **kw):
    def step(t, dt, inp, out, state):
        out[0] = 2.0 * (1.0 - inp[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return p.add_c_block(b, inputs=[("v", "v")],
                             outputs=[("v", "u", "gnd")], dt=TS, fn=step, **kw)


@pytest.mark.parametrize("h,tol", [(1e-6, 1.2e-4), (1e-7, 1.2e-5)])
def test_c_block_p_loop_runs_the_label_law(h, tol):
    # The trapezoidal step averages each undeclared ZOH jump (first order
    # after a discontinuity: known-limitations), so the error is O(h):
    # measured 6.17e-5 at h = 1 us, 6.14e-6 at 0.1 us.
    b = _rc()
    blk = _p_block(b)
    r = p.simulate(b, 11 * TS, dt=h)
    t, v = np.asarray(r.times), np.asarray(r.v("v"))
    i = int(np.argmin(np.abs(t - (10 * TS - h))))
    assert abs(v[i] - w_label(10, h)) < tol
    assert blk.n_fires == 11            # ticks 0..10 (instants < t_end)


@pytest.mark.parametrize("engine,kw", [("pwl", {"dt": 1e-6}), ("trbdf2", {})])
def test_a_built_circuit_runs_its_c_blocks_bit_for_bit(engine, kw):
    # Integration batch 10 (control graph x circuit IR): the run's builder
    # is Circuit.to_builder(), which carries no Python attribute; the
    # block's graph is found through the circuit's ControlDecls. Before
    # the fix the circuit form ran without its C block (v stayed 0).
    b1, b2 = _rc(), _rc()
    h1, h2 = _p_block(b1), _p_block(b2)
    r1 = p.simulate(b1, 11 * TS, engine=engine, **kw)
    r2 = p.simulate(b2.build(), 11 * TS, engine=engine, **kw)
    assert h1.n_fires == h2.n_fires == 11
    assert np.array_equal(r1.times, r2.times)
    assert np.array_equal(r1.states, r2.states)
    assert np.asarray(r2.v("v"))[-1] > 0.6     # the block drove the RC


def test_label_horizon_drops_the_tick_at_t_end():
    # Before: a C block also fired at t_end (11 fires where 10 are due).
    b = _rc()
    blk = _p_block(b)
    p.simulate(b, 10 * TS, dt=1e-6)
    assert blk.n_fires == 10


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    b.add_switch("Q", "in", "sw", 1e2, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e2, 1e-9)
    b.add_inductor("L", "sw", "out", 1e-3)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", 10.0)
    return b


def _pi():
    return p.PIController(Kp=0.02, Ki=20.0, output_min=0.05, output_max=0.95)


def test_node_name_and_lambda_give_the_same_bits():
    b1, b2 = _buck(), _buck()
    l1 = p.bind_pi_to_switch(b1, pi=_pi(), measured="out", setpoint=12.0,
                             switch="Q", freq=10e3)
    with pytest.warns(DeprecationWarning, match="measured"):
        l2 = p.bind_pi_to_switch(
            b2, pi=_pi(), measured=lambda x, i=b2.node_id_of("out"): x[i],
            setpoint=12.0, switch="Q", freq=10e3)
    r1 = p.simulate(b1, 3e-3, dt=1e-6, closed_loops=[l1])
    r2 = p.simulate(b2, 3e-3, dt=1e-6, closed_loops=[l2])
    assert np.asarray(r1.states).tobytes() == np.asarray(r2.states).tobytes()
    assert list(l1.duty_history) == list(l2.duty_history)
    assert len(l1.duty_history) == 30


def test_closed_loop_matches_the_python_pi_tick_by_tick():
    # The kernel PI on the graph against PIController.update fed the same
    # measurements (the duty history and the error history).
    b = _buck()
    lp = p.bind_pi_to_switch(b, pi=_pi(), measured="out", setpoint=12.0,
                             switch="Q", freq=10e3)
    p.simulate(b, 3e-3, dt=1e-6, closed_loops=[lp])
    ref = _pi()
    assert len(lp.duty_history) == len(lp.error_history) == 30
    for (t, d), (_t, e) in zip(lp.duty_history, lp.error_history):
        assert ref.update(setpoint=12.0, measured=12.0 - e, dt=1e-4) == d


# ---------------------------------------------------------------------------
# 9.2 counter gate on the consolidated paths
# ---------------------------------------------------------------------------

_RUNNERS = {"run_transient", "run_transient_with_chain",
            "run_transient_trbdf2",
            "_run_simulation"}  # simulate()'s kernel object


class PyCallsDuringRun:
    def __init__(self):
        self.n = 0
        self._inside = 0

    def _hook(self, frame, event, arg):
        name = getattr(arg, "__name__", "")
        if event == "c_call" and name in _RUNNERS:
            self._inside += 1
        elif event in ("c_return", "c_exception") and name in _RUNNERS:
            self._inside -= 1
        elif event == "call" and self._inside:
            self.n += 1

    def __enter__(self):
        sys.setprofile(self._hook)
        return self

    def __exit__(self, *exc):
        sys.setprofile(None)


@pytest.mark.parametrize("engine,kw", [("pwl", {"dt": 1e-6}), ("trbdf2", {})])
def test_a_closed_loop_with_no_python_block_makes_no_python_call(engine, kw):
    b = _buck()
    lp = p.bind_pi_to_switch(b, pi=_pi(), measured="out", setpoint=12.0,
                             switch="Q", freq=10e3)
    with PyCallsDuringRun() as c:
        r = p.simulate(b, 2e-3, engine=engine, closed_loops=[lp], **kw)
    assert c.n == 0
    cxx = lp.switch_fn._graph._cxx_chain
    # every block of the graph evaluated once per tick, none per step
    assert cxx.n_block_evals == 20 * cxx.size()
    assert list(r.control_tick_counts) == [20]
    assert "closed_loops" not in r.step_loop_python_callbacks


def test_a_lambda_costs_one_python_call_per_tick():
    b = _buck()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        lp = p.bind_pi_to_switch(
            b, pi=_pi(), measured=lambda x, i=b.node_id_of("out"): x[i],
            setpoint=12.0, switch="Q", freq=10e3)
    with PyCallsDuringRun() as c:
        p.simulate(b, 2e-3, dt=1e-6, closed_loops=[lp])
    # per tick: the PythonBlock adapter, the helper's float() wrapper and
    # the user's lambda (before: 4044 calls on 1 ms, per step)
    assert c.n == 3 * 20


def test_a_compiled_c_block_makes_no_python_call():
    pytest.importorskip("pulsim.c_block")
    from pulsim.c_block import _find_compiler
    if _find_compiler("c") is None:
        if sys.platform == "win32":
            pytest.skip("no C compiler on PATH (Windows runner)")
        pytest.fail("no C compiler found")
    b = _rc()
    h = p.add_c_block(b, inputs=[("v", "v")], outputs=[("v", "u", "gnd")],
                      dt=TS, code="out[0] = 2.0 * (1.0 - in[0]);")
    with PyCallsDuringRun() as c:
        r = p.simulate(b, 11 * TS, dt=1e-6)
    assert c.n == 0
    assert h.n_fires == 11
    b2 = _rc()
    _p_block(b2)
    r2 = p.simulate(b2, 11 * TS, dt=1e-6)
    assert np.asarray(r.states).tobytes() == np.asarray(r2.states).tobytes()


def test_a_reused_loop_reports_this_runs_ticks():
    # The helper graph is built once; its counters keep running across
    # runs, control_tick_counts is per run (was [40] on the second).
    b = _buck()
    lp = p.bind_pi_to_switch(b, pi=_pi(), measured="out", setpoint=12.0,
                             switch="Q", freq=10e3)
    for _ in range(2):
        r = p.simulate(b, 2e-3, dt=1e-6, closed_loops=[lp])
        assert list(r.control_tick_counts) == [20]


def test_a_python_block_in_a_composed_tick_is_reported():
    b = _buck()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        lp = p.bind_pi_to_switch(
            b, pi=_pi(), measured=lambda x, i=b.node_id_of("out"): x[i],
            setpoint=12.0, switch="Q", freq=10e3)
    r = p.simulate(b, 1e-3, dt=1e-6, closed_loops=[lp])
    assert "closed_loops" in r.step_loop_python_callbacks


def test_a_built_circuit_keeps_its_c_blocks():
    # Integration batch 10 review: the builder and its Circuit share the
    # C-block graph, so a block added after build() would also have run in
    # the circuit built before it. Refused by name; simulate() does not seal.
    b = _rc()
    _p_block(b)
    p.simulate(b, 2 * TS, dt=1e-6)
    _p_block(b, name="second")             # after a run: still allowed
    c = b.build()
    with pytest.raises(ValueError, match="already carried by a pulsim.Circuit"):
        _p_block(b, name="third")
    with pytest.raises(ValueError, match="already carried by a pulsim.Circuit"):
        _p_block(c.to_builder(), name="fourth")
    assert [d[0] for d in c.controls()] == ["CBLK", "second"]
