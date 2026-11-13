"""update-fixed-step-loop-bookkeeping: the fixed engine's per-step overhead.

Pinned by counters, never by wall time: `res.counters` reports the
schedule evaluations of the step loop (`switch_fn_calls`), the segment
lookups the last-segment memo did not absorb (`segment_searches`) and
the rollback snapshots taken (`rollback_snapshots`). Results are
bit-identical to the loop before the change
(`openspec/changes/update-fixed-step-loop-bookkeeping/probes/
bookkeeping_baseline.py check`). Removing any item moves a count here.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p

DT = 1e-7


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("Q1", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 10.0)
    return b


def _run(n_steps, **kw):
    b = _buck()
    i, n = b.switch_index_of("Q1"), b.graph.num_switches
    calls = [0]

    def sw(t):  # 100 kHz, D = 0.4, a plain (opaque) Python callable
        calls[0] += 1
        m = p.SwitchStateMask(n)
        m.set(i, (t * 1e5) % 1.0 < 0.4)
        return m

    res = p.simulate(b, t_end=n_steps * DT, dt=DT, engine="pwl",
                     switch_fn=sw, **kw)
    return res, calls[0]


@pytest.mark.parametrize("kw", [{}, dict(discontinuity_restart=True)])
def test_switch_fn_is_read_once_per_step(kw):
    """calls(2N) − calls(N) == N: one evaluation per step (two before:
    one for the pre-event mask and one per event iteration)."""
    r1, c1 = _run(1000, **kw)
    r2, c2 = _run(2000, **kw)
    assert not r1.dt_retries and not r2.dt_retries
    assert c2 - c1 == 1000
    assert r2.counters["switch_fn_calls"] == 2000
    # The one call outside the loop is the start-up mask.
    assert c2 == r2.counters["switch_fn_calls"] + 1


def test_segment_memo_absorbs_repeated_lookups():
    """Default engine: 81 searches for 2000 steps (one per solve, >= 2000,
    before); with the restart rule 40. A search happens only when the
    mask differs from the previous solve's."""
    res, _ = _run(2000)
    assert res.counters["segment_searches"] == 81
    res, _ = _run(2000, discontinuity_restart=True)
    assert res.counters["segment_searches"] == 40


def test_rollback_snapshots_only_when_a_rollback_path_exists():
    res, _ = _run(500)
    assert res.counters["rollback_snapshots"] == 500
    res, _ = _run(500, max_dt_halvings=0)
    assert res.counters["rollback_snapshots"] == 0
    res, _ = _run(500, max_dt_halvings=0, discontinuity_restart=True)
    assert res.counters["rollback_snapshots"] == 500


def test_memo_survives_eviction_between_steps():
    """Masks A, B, A under a one-segment byte budget: each change
    evicts the other segment and re-inserts it (generation bump), so a
    memo that ignored the generation would hand back a freed segment.
    The bounded run equals the unbounded one bit for bit."""

    def run(bounded):
        b = p.CircuitBuilder()
        b.add_voltage_source("Vin", "vin", "gnd", 12.0)
        b.add_switch("S0", "vin", "a", 1e-3, 1e9)
        b.add_switch("S1", "vin", "a", 1e-1, 1e9)
        b.add_inductor("L", "a", "out", 1e-4)
        b.add_resistor("R", "out", "gnd", 10.0)
        cache = p.PwlStateSpaceCache(b.graph, b.pool)
        cache.build_lazy(1e-6)

        def sw(t):
            m = p.SwitchStateMask(2)
            k = int(round(t / 1e-6))
            m.set(0 if (k < 20 or k >= 40) else 1, True)
            return m

        if bounded:
            p.run_transient(cache, b.graph, b.pool, p.SimulationOptions(
                t_start=0.0, t_end=1e-6, dt=1e-6), sw)
            cache.set_segment_budget_bytes(cache.segment_cache_bytes())
        res = p.run_transient(cache, b.graph, b.pool, p.SimulationOptions(
            t_start=0.0, t_end=6e-5, dt=1e-6), sw)
        return cache, res

    cb, rb = run(True)
    cf, rf = run(False)
    assert cb.metrics().segment_evictions == 2
    assert cf.metrics().segment_evictions == 0
    assert rb.counters["segment_searches"] == 3
    assert np.array_equal(rb.states, rf.states)


def test_default_schedule_is_native(monkeypatch):
    """No switch_fn on the pwl route: simulate() installs the C++
    constant schedule (no Python callable in the step loop)."""
    made = []
    real = p.make_constant_switch_fn

    def spy(mask):
        made.append(mask)
        return real(mask)

    monkeypatch.setattr(p, "make_constant_switch_fn", spy)
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 1.0)
    b.add_resistor("R", "in", "out", 1.0)
    b.add_capacitor("C", "out", "gnd", 1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, t_end=1e-5, dt=1e-7, engine="pwl")
    assert len(made) == 1
    assert res.counters["switch_fn_calls"] == 100
    sched = real(made[0])
    assert isinstance(sched, p.SwitchSchedule)
    assert sched.next_edge_after(0.0) == float("inf")


def test_times_is_a_read_only_zero_copy_view():
    res, _ = _run(200)
    t = res.times
    assert isinstance(t, np.ndarray) and t.dtype == np.float64
    assert t.base is res
    assert not t.flags.writeable
    assert t.shape == (201,)
    np.testing.assert_allclose(t, np.arange(201) * DT, rtol=1e-12, atol=0)
    assert list(res.times) == t.tolist()
