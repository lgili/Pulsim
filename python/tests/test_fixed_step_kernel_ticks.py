"""The fixed engine's Python tick sources are scheduled by the kernel's
event queue (add-unified-event-queue group 5), not by a Python object in
the step observer."""

from __future__ import annotations

import math

import numpy as np
import pytest

import pulsim as p


def _rc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "in", "gnd", 12.0)
    b.add_resistor("R", "in", "out", 10.0)
    b.add_capacitor("C", "out", "gnd", 1e-6)
    return b


def test_ticks_are_counted_and_delivered_at_their_instants():
    b = _rc()
    seen = []
    T, dt, t_end = 2.5e-5, 1e-6, 1e-3
    r = p.simulate(b, t_end, dt=dt, engine="pwl",
                   step_observer=lambda t, x: seen.append(t),
                   controller_period=T)
    n = math.ceil(t_end / T)            # instants k*T < t_end
    assert r.control_tick_counts == [n] == [len(seen)]
    # Each tick at the first grid point at or after k*T (here on it).
    assert np.allclose(seen, np.arange(n) * T, rtol=0, atol=1e-9 * dt)
    assert r.control_tick_jitter_max < 1e-9 * dt
    assert r.kernel_tick_counts == [n]


def test_a_period_off_the_grid_is_late_by_less_than_dt():
    b = _rc()
    seen = []
    T, dt = 3.35e-5, 1e-6
    with pytest.warns(UserWarning, match="not a multiple of dt"):
        r = p.simulate(b, 1e-3, dt=dt, engine="pwl",
                       step_observer=lambda t, x: seen.append(t),
                       controller_period=T)
    assert len(seen) == r.control_tick_counts[0] == math.ceil(1e-3 / T)
    late = np.asarray(seen) - np.arange(len(seen)) * T
    assert np.all(late > -1e-9 * dt) and np.all(late < dt)
    assert r.control_tick_jitter_max == pytest.approx(late.max(), abs=1e-18)


def test_run_transient_refuses_mismatched_tick_lists():
    b = _rc()
    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build_lazy(1e-6)
    opts = p.SimulationOptions()
    opts.t_end, opts.dt = 1e-5, 1e-6
    with pytest.raises(ValueError, match="same length"):
        p.run_transient(cache, b.graph, b.pool, opts,
                        switch_fn=lambda t: p.SwitchStateMask(0),
                        tick_periods=[1e-6], tick_fns=[])


def test_python_ticks_keep_their_place_between_the_observers():
    """Per step: internal drivers (on_cache / MMC) and graphs, then the
    Python tick sources, then the every-step observers — the order of the
    observer the ticks used to be part of."""
    import types
    b = _rc()
    log = []
    loop = types.SimpleNamespace(
        step_observer=lambda t, x: log.append(("tick", t)), period=5e-6,
        switch_fn=None)
    r = p.simulate(b, 2e-5, dt=1e-6, engine="pwl", closed_loops=[loop],
                   step_observer=lambda t, x: log.append(("obs", t)),
                   on_cache=lambda c: (lambda t, x: log.append(("hook", t)),
                                       None))
    assert r.control_tick_counts == [4]
    assert [k for k, t in log if t == 0.0] == ["hook", "tick", "obs"]
    assert [k for k, t in log if abs(t - 5e-6) < 1e-12] == \
        ["hook", "tick", "obs"]
    assert [k for k, t in log if abs(t - 1e-6) < 1e-12] == ["hook", "obs"]
    assert "step_observer" in r.step_loop_python_callbacks
