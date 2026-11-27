"""F-Z-2: the event split's locators stop on the signal. A trial is taken
when the root of its bracket's raw chord is within split_snap*dt of it (the
time tolerance through the signal's slope), an end of the step's own bracket
the same way before any trial, and the cut at the last trial's instant
solves on the factor that trial left. Cell: a half-wave rectifier, a sine
through an ideal-ish diode into R: the crossings are the sine's zeros (the
circuit is algebraic, so the numerical crossing is the true one). Options
explicit: split_snap is reachable only through run_transient."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import pulsim as p

A, F, R, T_END, SNAP = 10.0, 50.0, 10.0, 0.1, 1e-6


def _rect(dt):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, A, F)
    b.add_diode("D", "in", "k", 1e6, 1e-12)
    b.add_resistor("R", "k", "gnd", R)
    n = b.graph.num_switches
    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build_lazy(dt)
    o = p.SimulationOptions(t_start=0.0, t_end=T_END, dt=dt)
    o.discontinuity_restart = True
    o.enable_substep_state_correction = True
    o.split_snap = SNAP
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.run_transient(cache, b.graph, b.pool, o, lambda t: p.SwitchStateMask(n))
    return res, cache.metrics()


def _err_over_dt(res, dt):
    te = np.array([e.t_estimated for e in res.commutation_events])
    zeros = np.arange(0.0, T_END + 1 / F, 0.5 / F)
    return np.array([np.min(np.abs(zeros - t)) for t in te]) / dt


@pytest.mark.parametrize("dt", [3.7e-5, 1e-4], ids=["off-grid", "on-grid"])
def test_located_time_within_the_stated_tolerance(dt):
    res, _ = _rect(dt)
    err = _err_over_dt(res, dt)
    assert len(err) >= 8
    # within snap of the final chord's root, which is within snap of the root
    # (first order): 2 snap
    assert err.max() <= 2 * SNAP, err


def test_trials_and_factorisations_per_located_event():
    # 3.7e-5 s: every zero falls inside a step. Before F-Z-2: 62 trials and 84
    # refactors for 9 split steps (6.9 / 9.3 per event); after: 40 / 58
    # (4.4 / 6.4). Bounds a trial above the measurement (platforms differ).
    res, m = _rect(3.7e-5)
    n = res.n_split_steps
    assert n >= 8
    assert res.counters["split_locator_trials"] <= 5 * n
    assert m.event_refactors <= 7 * n
    assert res.counters["split_locator_giveups"] == 0


def test_a_crossing_on_the_grid_takes_no_trial():
    # 1e-4 s: every zero is a grid point. The step's own bracket has the
    # root at its end (to round-off): no trial solve, no refactor (before
    # F-Z-2: 30 trials and 34 refactors for 10 events).
    res, m = _rect(1e-4)
    assert len(res.commutation_events) >= 8
    assert res.counters["split_locator_trials"] <= 2
    assert m.event_refactors <= 4
    assert math.isfinite(_err_over_dt(res, 1e-4).max())
