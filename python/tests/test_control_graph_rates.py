"""Rates of the control graph: tick counters, rate inheritance and
compute_delay (add-compiled-control-graph, task group 4).

Closed forms (design.md): discrete P loop on an RC (tau = 1 ms) through a
zero-order-hold controlled source, T_s = 100 us, K_p = 2, set-point 1 V,
zero state. a = exp(-0.1), v[k] = (2/3)(1 - p^k), p = a - 2(1 - a):
v[10] = 0.643545722992. With compute_delay = one_sample,
v[k+1] = a v[k] + (1 - a) K_p (1 - v[k-1]): v[10] = 0.659695654069.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

import pulsim as p

TS = 1e-4
A = math.exp(-0.1)
PP = A - 2.0 * (1.0 - A)


def v_zero_delay(k):
    return (2.0 / 3.0) * (1.0 - PP ** k)


def v_one_sample(n):
    # u[k] = Kp*(1 - v[k-1]) is applied on [t_k, t_k+1); on [t_0, t_1)
    # the source holds its initial 0 (nothing was computed before tick 0).
    v = [0.0]
    u_prev = 0.0
    for k in range(n):
        u_k = u_prev
        v.append(A * v[k] + (1.0 - A) * u_k)
        u_prev = 2.0 * (1.0 - v[k])
    return v


def test_closed_forms():
    assert v_zero_delay(10) == pytest.approx(0.643545722992, abs=1e-12)
    assert v_one_sample(10)[10] == pytest.approx(0.659695654069, abs=1e-12)


def _p_loop(compute_delay=0.0):
    b = p.CircuitBuilder()
    ch = p.MixedDomainBlockChain(rate=p.Rate(TS, compute_delay=compute_delay))
    ch.add("err", p.Subtract(), inputs=dict(a=1.0, b="v"), output="e")
    ch.add("kp", p.Gain(k=2.0), inputs=dict(x="channel:e"), output="u")
    ch.drive_source(b, "u", ("v", "u", "gnd"))
    b.add_resistor("R", "u", "v", 1e3)
    b.add_capacitor("C", "v", "gnd", 1e-6)
    return b, ch


def _samples(r, n=21):
    t = np.asarray(r.times)
    v = np.asarray(r.v("v"))
    out = []
    for k in range(n):
        i = int(np.argmin(np.abs(t - k * TS)))
        assert abs(t[i] - k * TS) < 1e-12 * TS + 1e-15
        out.append(v[i])
    return np.array(out)


def _run(engine, compute_delay=0.0, **kw):
    b, ch = _p_loop(compute_delay)
    obs = ch.make_step_observer(b, dt=kw.get("dt"))
    r = p.simulate(b, 21 * TS, engine=engine, step_observer=obs, **kw)
    return r, ch


# pwl with discontinuity_restart: the graph declares its tick instants
# (b_extra_fn.next_jump_after), the engine restarts on each, and the ZOH
# jump is not averaged into the step after it.
# Measured |v - closed form| over k <= 20 (v[10]): pwl 2.3e-8 (7.2e-9)
# at dt = 1 us, 2.3e-10 (7.3e-11) at 0.1 us (second order); trbdf2
# 8.9e-9 at rtol 1e-10, 4.0e-10 at rtol 1e-12.
_EXACT = [
    ("pwl", {"dt": 1e-7, "discontinuity_restart": True}, 1e-9),
    ("pwl", {"dt": 1e-6, "discontinuity_restart": True}, 1e-7),
    ("trbdf2", {"rtol": 1e-12, "atol": 1e-14}, 1e-9),
    ("trbdf2", {"rtol": 1e-10, "atol": 1e-12}, 2e-8),
]


@pytest.mark.parametrize("engine,kw,tol", _EXACT)
def test_discrete_p_loop_matches_the_closed_form(engine, kw, tol):
    r, ch = _run(engine, **kw)
    v = _samples(r)
    assert v[10] == pytest.approx(0.643545722992, abs=tol)
    np.testing.assert_allclose(v, [v_zero_delay(k) for k in range(21)],
                               atol=tol, rtol=0)
    assert list(ch.tick_counts) == [21]


@pytest.mark.parametrize("engine,kw,tol", _EXACT)
def test_one_sample_delay_matches_the_closed_form(engine, kw, tol):
    r, _ = _run(engine, compute_delay="one_sample", **kw)
    v = _samples(r)
    assert v[10] == pytest.approx(0.659695654069, abs=tol)
    np.testing.assert_allclose(v, v_one_sample(20), atol=tol, rtol=0)


@pytest.mark.parametrize("engine,kw", [(e, k) for e, k, _ in _EXACT])
def test_v_settles_on_two_thirds(engine, kw):
    b, ch = _p_loop()
    r = p.simulate(b, 120 * TS, engine=engine,
                   step_observer=ch.make_step_observer(b), **kw)
    t = np.asarray(r.times)
    i = int(np.argmin(np.abs(t - 120 * TS)))
    assert abs(np.asarray(r.v("v"))[i] - 2.0 / 3.0) < 1e-9


def test_without_the_restart_pwl_is_first_order_at_the_zoh_jumps():
    """The fixed engine's documented behaviour (known-limitations: first
    order after a discontinuity): the trapezoidal history averages each
    ZOH jump into the step after it. The error is O(dt), so it is the
    engine's, not the graph's; the restart rule removes it (above)."""
    errs = []
    for dt in (1e-6, 1e-7):
        r, _ = _run("pwl", dt=dt)
        errs.append(abs(_samples(r)[10] - 0.643545722992))
    assert 5e-5 < errs[0] < 7e-5
    assert errs[0] / errs[1] == pytest.approx(10.0, rel=0.02)


# ---------------------------------------------------------------------------
# tick counts (task 1.2 numbers, now exact on both engines for the graph)
# ---------------------------------------------------------------------------

def _rc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "in", "gnd", 1.0)
    b.add_resistor("R", "in", "out", 1e3)
    b.add_capacitor("C", "out", "gnd", 1e-6)
    return b


def _two_rate_graph():
    ch = p.MixedDomainBlockChain()
    slow, fast = p.Rate(1.0 / 2e3), p.Rate(1.0 / 20e3)
    ch.add("pi", p.PIController(Kp=0.1, Ki=10.0),
           inputs=dict(setpoint=1.0, measured="out", dt="dt"),
           output="u", rate=slow)
    ch.add("mon", p.FirstOrderLowPass(tau=1e-3),
           inputs=dict(input_value="out", dt="dt"), output="m", rate=fast)
    ch.add("sum", p.Sum(), inputs=dict(a="channel:u", b="channel:m"),
           output="s")                       # inherits the fast rate
    return ch


_ENG = [("pwl", {"dt": 1e-6}), ("trbdf2", {})]


@pytest.mark.parametrize("engine,kw", _ENG)
def test_one_rate_ticks_200_times(engine, kw):
    b = _rc()
    ch = p.MixedDomainBlockChain(rate=p.Rate(1e-4))
    ch.add("g", p.Gain(k=2.0), inputs=dict(x="out"), output="u")
    ch.add("i", p.Integrator(gain=1.0), inputs=dict(x="channel:u", dt="dt"),
           output="y")
    obs = ch.make_step_observer(b)
    p.simulate(b, 20e-3, engine=engine, step_observer=obs, **kw)
    assert ch.tick_counts == [200]
    cxx = ch._cxx_chain
    assert cxx.n_block_evals == 200 * 2     # n_ticks x blocks in the group


@pytest.mark.parametrize("engine,kw", _ENG)
def test_two_rates_tick_40_and_400_times(engine, kw):
    b = _rc()
    ch = _two_rate_graph()
    obs = ch.make_step_observer(b)
    p.simulate(b, 20e-3, engine=engine, step_observer=obs, **kw)
    assert ch.tick_counts == [40, 400]
    # pi on 40 ticks, mon and the inheriting sum on 400.
    assert ch._cxx_chain.n_block_evals == 40 + 2 * 400


def test_block_evaluations_fall_from_one_per_step_to_one_per_tick():
    # 10 kHz controller on a dt = 1e-7 fixed run of 10 ms.
    b = _rc()
    every = p.MixedDomainBlockChain()
    every.add("g", p.Gain(k=2.0), inputs=dict(x="out"), output="u")
    p.simulate(b, 10e-3, dt=1e-7, step_observer=every.make_step_observer(
        b, dt=1e-7))
    b2 = _rc()
    rated = p.MixedDomainBlockChain(rate=p.Rate(1e-4))
    rated.add("g", p.Gain(k=2.0), inputs=dict(x="out"), output="u")
    p.simulate(b2, 10e-3, dt=1e-7, step_observer=rated.make_step_observer(b2))
    assert every._cxx_chain.n_block_evals == 100001   # every observer call
    assert rated._cxx_chain.n_block_evals == 100      # one per tick


def test_offset_on_the_fixed_engine():
    b = _rc()
    ch = p.MixedDomainBlockChain(rate=p.Rate(1e-4, offset=5e-5))
    ch.add("g", p.Gain(k=1.0), inputs=dict(x="out"), output="u")
    obs = ch.make_step_observer(b)
    ch._cxx_chain.record_channel("u")
    p.simulate(b, 1e-3, dt=1e-6, step_observer=obs)
    t = np.asarray(ch._cxx_chain.get_recording_times())
    np.testing.assert_allclose(t, 5e-5 + np.arange(10) * 1e-4, atol=1e-12)


def test_offset_and_delay_time_are_refused_on_trbdf2():
    for rate, feature in ((p.Rate(1e-4, offset=5e-5), "control_rate_offset"),
                          (p.Rate(1e-4, compute_delay=2e-5), "compute_delay")):
        b = _rc()
        ch = p.MixedDomainBlockChain(rate=rate)
        ch.add("g", p.Gain(k=1.0), inputs=dict(x="out"), output="u")
        with pytest.raises(p.UnsupportedError) as e:
            p.simulate(b, 1e-3, engine="trbdf2",
                       step_observer=ch.make_step_observer(b))
        assert e.value.feature == feature


def test_compute_delay_time_on_the_fixed_engine():
    b, ch = _p_loop(compute_delay=2e-5)
    r = p.simulate(b, 3 * TS, dt=1e-6, step_observer=ch.make_step_observer(b))
    t = np.asarray(r.times)
    vu = np.asarray(r.v("u"))
    # u = 2 published at t = 20 us (tick 0 + Td), not before.
    assert vu[np.searchsorted(t, 19e-6)] == 0.0
    assert vu[np.searchsorted(t, 22e-6)] == pytest.approx(2.0)


def test_a_stateful_block_without_a_rate_is_a_build_error():
    b = _rc()
    ch = p.MixedDomainBlockChain()
    ch.add("g", p.Gain(k=1.0), inputs=dict(x="out"), output="u",
           rate=p.Rate(1e-4))
    ch.add("i", p.Integrator(), inputs=dict(x="channel:u", dt="dt"),
           output="y")
    with pytest.raises(p.ControlGraphBuildError,
                       match="'i' \\(Integrator\\) has no rate"):
        ch.make_step_observer(b)


def test_a_rate_graph_makes_no_python_call_during_a_run():
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_control_graph_characterisation import PyCallsDuringRun
    for engine, kw in _ENG:
        b = _rc()
        ch = _two_rate_graph()
        obs = ch.make_step_observer(b)
        with PyCallsDuringRun() as c:
            p.simulate(b, 2e-3, engine=engine, step_observer=obs, **kw)
        assert c.n == 0, engine


def test_the_chain_fast_path_passes_a_declared_b_extra_jump():
    """run_transient_with_chain (any chain observer) dropped the
    b_extra_fn's next_jump_after, so the restart rule never saw the jump:
    one 0 -> 1 V step into the RC stayed first order (5.0e-4 at 1 us)."""
    from pulsim.c_block import _V_SRC_SIGN
    tau = 5e-4
    b = p.CircuitBuilder()
    b.add_voltage_source("Vu", "u", "gnd", 0.0)
    b.add_resistor("R", "u", "v", 1e3)
    b.add_capacitor("C", "v", "gnd", 1e-6)
    n = int(b.pool.state_size(b.graph))
    row = int(b.pool.branch_var_id_for_source(int(b.branch_id_of("Vu")),
                                              b.graph))

    def bex(t):
        out = np.zeros(n)
        out[row] = _V_SRC_SIGN * (1.0 if t >= tau else 0.0)
        return out
    bex.next_jump_after = lambda t: tau if t < tau else math.inf
    ch = p.MixedDomainBlockChain()
    ch.add("g", p.Gain(k=1.0), inputs=dict(x="v"), output="y")
    obs = ch.make_step_observer(b, dt=1e-6)
    r = p.simulate(b, 1e-3, dt=1e-6, b_extra_fn=bex, step_observer=obs,
                   discontinuity_restart=True)
    t = np.asarray(r.times)
    exact = np.where(t >= tau, 1.0 - np.exp(-(t - tau) / 1e-3), 0.0)
    assert np.max(np.abs(np.asarray(r.v("v")) - exact)) < 1e-7
