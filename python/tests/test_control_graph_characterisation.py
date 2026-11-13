"""Characterisation of the control path before the compiled control graph
(add-compiled-control-graph, task group 1).

What these tests pin:

* evaluation order: the chain (``MixedDomainBlockChain``) runs its blocks
  in INSERTION order; a channel read before any block wrote it reads 0;
* tick counts per engine for one loop and for two rates;
* how many Python functions the kernel calls back during one run of a
  C++ chain, of a ``ClosedLoop`` and of a C block (the baseline of the
  counter gate, task 9.2).
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest

import pulsim as p

_RUNNERS = {"run_transient", "run_transient_with_chain",
            "run_transient_trbdf2"}


def test_pulsim_is_this_worktree():
    # A stale wheel would make every number below meaningless.
    assert "site-packages" not in p.__file__


class PyCallsDuringRun:
    """Counts Python function calls made while a kernel runner (a C
    function) is on the stack: the callbacks the run pays for."""

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


# ---------------------------------------------------------------------------
# 1.1 evaluation order
# ---------------------------------------------------------------------------

def _chain(order):
    specs = {
        "err": (p.Subtract(), dict(a=1.0, b="channel:meas"), "e"),
        "gain": (p.Gain(k=2.0), dict(x="channel:e"), "u"),
        "half": (p.Gain(k=0.5), dict(x="channel:u"), "y"),
    }
    ch = p.MixedDomainBlockChain()
    for name in order:
        blk, inputs, out = specs[name]
        ch.add(name, blk, inputs=inputs, output=out)
    return ch


def _run_chain(order, n=200):
    """Drive the compiled chain directly with a known `meas` sequence;
    return the `y` history."""
    b = p.CircuitBuilder()
    b.add_resistor("R", "a", "gnd", 1.0)
    ch = _chain(order)
    obs = ch.make_step_observer(b, dt=1e-6)
    cxx = ch._cxx_chain
    cxx.record_channel("y", n)
    x = np.zeros(b.graph.num_nodes() if callable(
        getattr(b.graph, "num_nodes", None)) else 1)
    meas = 0.5 * np.sin(np.arange(n) * 0.05)
    for k in range(n):
        cxx.set_channel("meas", float(meas[k]))
        obs(k * 1e-6, x)
    return np.asarray(cxx.get_channel_history("y")), meas


def test_insertion_order_changes_the_result_today():
    y_fwd, meas = _run_chain(["err", "gain", "half"])
    y_rev, _ = _run_chain(["half", "gain", "err"])
    np.testing.assert_array_equal(y_fwd, 0.5 * (2.0 * (1.0 - meas)))
    # Reverse order: each read-before-write is one step of delay, so the
    # last gain sees the measurement two steps late.
    np.testing.assert_array_equal(y_rev[2:], y_fwd[:-2])
    diff = float(np.max(np.abs(y_fwd - y_rev)))
    assert diff == pytest.approx(1.0)          # y_rev[0] = 0 vs y_fwd[0] = 1


def test_a_channel_read_before_it_is_written_reads_zero_today():
    y_rev, _ = _run_chain(["half", "gain", "err"], n=3)
    assert y_rev[0] == 0.0 and y_rev[1] == 0.0


@pytest.mark.parametrize("block,inputs", [
    (p.Gain(k=2.0), dict(x="out")),                         # node input
    (p.Limiter(output_min=0.0, output_max=1.0), dict(x="out")),
])
def test_some_chains_fall_back_to_the_python_evaluator_today(block, inputs):
    b = p.CircuitBuilder()
    b.add_resistor("R", "out", "gnd", 1.0)
    ch = p.MixedDomainBlockChain()
    ch.add("blk", block, inputs=inputs, output="u")
    with pytest.warns(UserWarning, match="falling back to Python"):
        ch.make_step_observer(b, dt=1e-6)
    assert ch._cxx_chain is None


# ---------------------------------------------------------------------------
# 1.2 tick counts per engine
# ---------------------------------------------------------------------------

def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    b.add_switch("Q", "in", "sw", 1e2, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e2, 1e-9)
    b.add_inductor("L", "sw", "out", 1e-3)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", 10.0)
    return b


def _loop(b, freq):
    return p.control.bind_pi_to_switch(
        b, pi=p.PIController(Kp=0.02, Ki=20.0, output_min=0.05,
                             output_max=0.95),
        measured=lambda x, i=b.node_id_of("out"): x[i], setpoint=12.0,
        switch="Q", freq=freq)


_ENGINES = [("pwl", {"dt": 1e-6}), ("trbdf2", {})]


@pytest.mark.parametrize("engine,kw", _ENGINES)
def test_one_loop_ticks_200_times_in_20_ms_at_10_khz(engine, kw):
    b = _buck()
    lp = _loop(b, 10e3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, 20e-3, engine=engine, closed_loops=[lp], **kw)
    assert len(lp.duty_history) == 200
    assert list(r.control_tick_counts) == [200]


@pytest.mark.parametrize("engine,kw", _ENGINES)
def test_two_rates_tick_40_and_400_times(engine, kw):
    b = _buck()
    lp = _loop(b, 2e3)
    calls = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, 20e-3, engine=engine, closed_loops=[lp],
                       step_observer=lambda t, x: calls.append(t),
                       controller_period=50e-6, **kw)
    assert (len(lp.duty_history), len(calls)) == (40, 400)
    assert list(r.control_tick_counts) == [40, 400]


def test_dsed_refuses_a_closed_loop():
    b = _buck()
    with pytest.raises(p.UnsupportedError):
        p.simulate(b, 1e-3, engine="dsed", closed_loops=[_loop(b, 10e3)])


# ---------------------------------------------------------------------------
# 1.3 Python calls per run (the baseline of the counter gate)
# ---------------------------------------------------------------------------

def _rc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "in", "gnd", 1.0)
    b.add_resistor("R", "in", "out", 1e3)
    b.add_capacitor("C", "out", "gnd", 1e-6)
    return b


N_STEPS = 1000            # t_end = 1 ms, dt = 1 us


def test_python_calls_per_run_of_a_cxx_chain():
    b = _rc()
    ch = p.MixedDomainBlockChain()
    ch.add("e", p.Subtract(), inputs=dict(a=1.0, b="out"), output="u")
    obs = ch.make_step_observer(b, dt=1e-6)
    assert ch._cxx_chain is not None
    with PyCallsDuringRun() as c:
        p.simulate(b, 1e-3, dt=1e-6, engine="pwl", step_observer=obs)
    assert c.n == 0                    # the fast path runs the chain in C++


def test_python_calls_per_run_of_a_closed_loop():
    b = _buck()
    lp = _loop(b, 10e3)
    with PyCallsDuringRun() as c, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p.simulate(b, 1e-3, dt=1e-6, engine="pwl", closed_loops=[lp])
    # Per step: the fixed-step tick scheduler, the composed observer, the
    # gate function and pwm_on; per tick: `.tick`, `measured`, PI.update.
    # (The same loop on trbdf2 for 1 ms: 13710 calls, gate re-evaluations.)
    assert c.n == 4044
    assert len(lp.duty_history) == 10


def test_python_calls_per_run_of_a_c_block():
    b = _rc()
    b.add_resistor("Rl", "o2", "gnd", 1e3)

    def step(t, dt, inp, out, state):
        out[0] = 0.5 * inp[0]

    h = p.add_c_block(b, inputs=[("v", "out")],
                      outputs=[("v", "o2", "gnd")], dt=1e-4, fn=step)
    with PyCallsDuringRun() as c:
        p.simulate(b, 1e-3, dt=1e-6, engine="pwl")
    # Per step: the block's observer, its b_extra injector and helpers.
    assert c.n == 5014
    # Its own throttle fires at t_end too: 11 fires where 10 ticks are due.
    assert h.n_fires == 11
