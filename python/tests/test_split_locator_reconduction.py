"""F-Z-2 follow-up: a diode that turns off and conducts again soon after.
Cell: a sine with an offset, 9 + 10 sin(2 pi 5 kHz t), through an ideal-ish
diode into R. The source dips below zero for 28.5 us per period, so each
period is a turn-off at the downward zero and a turn-on at the upward one
28.5 us later; the circuit is algebraic, so both are the source's zeros
(closed form). Rule and split explicit (split_snap is reachable only
through run_transient). A dip that starts and ends inside one step is not
seen by any step-end check (the step is ON at both ends) and is not counted.

Measured (macOS, 2026-11-27; identical on 87a4f6bb before F-Z-2, so not the
stop rule): every turn-off is located within 1e-6 dt at every dt; the
turn-on is located when the dip spans >= 3 steps (dt = 9.25 us: 10 / 10
within 1e-6 dt) and is NOT located when it follows within ~2 steps of the
located turn-off: dt = 37 us 8 / 8 and dt = 18.5 us 10 / 10 turn-ons sit on
a grid point, max error 0.936 dt / 0.873 dt (n_split_steps = the number of
turn-offs: the turn-on step is not split)."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import pulsim as p

C, A, F, R, T_END, SNAP = 9.0, 10.0, 5e3, 10.0, 2e-3, 1e-6
W = 2 * math.pi * F
K = np.arange(int(T_END * F) + 1)
OFF = (math.pi + math.asin(C / A) + 2 * math.pi * K) / W       # downward zeros
ON = (2 * math.pi - math.asin(C / A) + 2 * math.pi * K) / W    # upward zeros


def _run(dt, split=True):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", C, A, F)
    b.add_diode("D", "in", "k", 1e6, 1e-12)
    b.add_resistor("R", "k", "gnd", R)
    n = b.graph.num_switches
    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build_lazy(dt)
    o = p.SimulationOptions(t_start=0.0, t_end=T_END, dt=dt)
    o.discontinuity_restart = True
    o.enable_substep_state_correction = split
    o.split_snap = SNAP
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.run_transient(cache, b.graph, b.pool, o, lambda t: p.SwitchStateMask(n))


def _times(res, on):
    # the start-up turn-on at t = 0 is not a dip
    return np.array([e.t_estimated for e in res.commutation_events
                     if e.new_state == on and e.t_estimated > 0.0])


def _visible(dt):
    # dips whose two zeros fall in different steps (a step end sees them)
    return int(sum(1 for a, b in zip(OFF, ON)
                   if b < T_END and math.floor(a / dt) != math.floor(b / dt)))


def _err_over_dt(te, ref, dt):
    return np.array([np.min(np.abs(ref - t)) for t in te]) / dt


@pytest.mark.parametrize("dt", [3.7e-5, 1.85e-5, 9.25e-6])
def test_every_turn_off_is_located_and_counted(dt):
    res = _run(dt)
    te = _times(res, False)
    assert len(te) == _visible(dt) >= 8
    # within snap of the final chord's root, within snap of the root: 2 snap
    assert _err_over_dt(te, OFF, dt).max() <= 2 * SNAP
    assert res.counters["split_locator_giveups"] == 0


def test_a_turn_on_three_steps_later_is_located_against_a_fine_rule_alone_run():
    dt = 9.25e-6                       # the dip spans 3.1 steps
    res = _run(dt)
    te = _times(res, True)
    assert len(te) == _visible(dt) >= 8
    assert _err_over_dt(te, ON, dt).max() <= 2 * SNAP
    # the rule alone at dt/1000 takes each event on a fine grid point next to
    # the crossing: within dt_f of it, so within dt_f + 2 snap dt of the split's
    dt_f = dt / 1000
    fine = _run(dt_f, split=False)
    for on, mine in ((False, _times(res, False)), (True, te)):
        ref = _times(fine, on)
        assert len(ref) == len(mine)
        assert np.max(np.abs(ref - mine)) <= dt_f + 2 * SNAP * dt


@pytest.mark.xfail(strict=True, reason=(
    "a turn-on within ~2 steps of a located turn-off is taken at a grid point, "
    "not located: dt=37us 8/8 max 0.936 dt, dt=18.5us 10/10 max 0.873 dt "
    "(pre-existing on 87a4f6bb, not the F-Z-2 stop rule)"))
@pytest.mark.parametrize("dt", [3.7e-5, 1.85e-5])
def test_a_turn_on_soon_after_a_located_turn_off_is_located(dt):
    res = _run(dt)
    te = _times(res, True)
    assert len(te) == _visible(dt)
    assert _err_over_dt(te, ON, dt).max() <= 2 * SNAP
