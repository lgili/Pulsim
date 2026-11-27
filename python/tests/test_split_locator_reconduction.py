"""F-Z-2 follow-up: a diode that turns off and conducts again soon after.
Cell: a sine with an offset, 9 + 10 sin(2 pi 5 kHz t), through an ideal-ish
diode into R. The source dips below zero for 28.7 us per period, so each
period is a turn-off at the downward zero and a turn-on at the upward one
28.7 us later; the circuit is algebraic, so both are the source's zeros
(closed form). Rule and split explicit (split_snap is reachable only
through run_transient). A dip that starts and ends inside one step is not
seen by any step-end check (the step is ON at both ends) and is not counted.

Measured (macOS, 2026-11-27): every turn-off is located within 1e-6 dt at
every dt. Before F-Z-11 the turn-on was located only when the dip spans >= 3
steps (dt = 9.25 us); one that fell in the restart window the located
turn-off opened (dip = 0.78 / 1.55 steps) sat on a grid point: dt = 37 us
8 / 8 and dt = 18.5 us 10 / 10, max error 0.936 dt / 0.873 dt. F-Z-11 locates
every flip of a window step (a cut inside the restart step): max 8.1e-7 dt /
5.9e-7 dt."""

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


def _run(dt, split=True, c=C):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", c, A, F)
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
    # every turn-on is seen too (located or not: see the xfail below)
    assert len(_times(res, True)) == _visible(dt)
    # within snap of the final chord's root, within snap of the root: 2 snap
    assert _err_over_dt(te, OFF, dt).max() <= 2 * SNAP
    assert res.counters["split_locator_giveups"] == 0


def test_a_turn_on_three_steps_later_is_located_against_a_fine_rule_alone_run():
    dt = 9.25e-6                       # the dip spans 3.1 steps
    res = _run(dt)
    te = _times(res, True)
    assert len(te) == _visible(dt) >= 8
    assert _err_over_dt(te, ON, dt).max() <= 2 * SNAP
    # the rule-alone path agrees: at dt/1000 it takes each event on a fine
    # grid point next to the crossing, within dt_f of it, so within
    # dt_f + 2 snap dt of the split's (a check of the rule alone's count and times)
    dt_f = dt / 1000
    fine = _run(dt_f, split=False)
    for on, mine in ((False, _times(res, False)), (True, te)):
        ref = _times(fine, on)
        assert len(ref) == len(mine)
        assert np.max(np.abs(ref - mine)) <= dt_f + 2 * SNAP * dt


@pytest.mark.parametrize("dt", [3.7e-5, 1.85e-5, 9.25e-6])
def test_a_turn_on_soon_after_a_located_turn_off_is_located(dt):
    # F-Z-11: the turn-on falls in the restart window the turn-off's cut opened
    # (a window step, SDIRK): located inside it too (before: on a grid point)
    res = _run(dt)
    te = _times(res, True)
    assert len(te) == _visible(dt) >= 8
    assert _err_over_dt(te, ON, dt).max() <= 2 * SNAP
    assert res.counters["split_locator_giveups"] == 0


# --- the mirror: a bump, -9 + 10 sin: a turn-off soon after a located turn-on --
ON2 = (math.asin(C / A) + 2 * math.pi * K) / W
OFF2 = (math.pi - math.asin(C / A) + 2 * math.pi * K) / W


@pytest.mark.parametrize("dt", [3.7e-5, 1.85e-5, 9.25e-6])
def test_a_turn_off_soon_after_a_located_turn_on_is_located(dt):
    res = _run(dt, c=-C)
    on, off = _times(res, True), _times(res, False)
    vis = int(sum(1 for a, b in zip(ON2, OFF2)
                  if b < T_END and math.floor(a / dt) != math.floor(b / dt)))
    assert len(off) == vis >= 8
    # shortcut (F-Z-11): a flip inside the run's first step is not located (a
    # zero start's round-off flips a diode there); at 37 us the first turn-on,
    # 35.6 us, is in it: taken at the step end, so it is left out here
    on = on[on >= dt]
    assert len(on) == sum(1 for a, b in zip(ON2, OFF2)
                          if a >= dt and b < T_END and math.floor(a / dt) != math.floor(b / dt))
    assert _err_over_dt(on, ON2, dt).max() <= 2 * SNAP
    assert _err_over_dt(off, OFF2, dt).max() <= 2 * SNAP
    assert res.counters["split_locator_giveups"] == 0


# --- a gate edge, then the freewheel diode's turn-off within two steps ---------
# Switch from 10 V into L = 10 uH, R = 1 ohm and a 5 V back-EMF; the diode
# freewheels when the switch opens (3 us into each 10 us period) and the
# current reaches zero ~2.3 us later: the turn-off falls in the first
# (dt = 2 us) or second (dt = 1.3 us) window step after the edge's split.
TB, T_ON, GON, GOFF = 1e-5, 3e-6, 1e3, 1e-9


def _freewheel(dt, split=True, n_per=6):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 10.0)
    b.add_switch("S", "vin", "sw", GON, GOFF)
    b.add_diode("D", "gnd", "sw", GON, GOFF)
    b.add_inductor("L", "sw", "a", 1e-5)
    b.add_resistor("R", "a", "o", 1.0)
    b.add_voltage_source("Vo", "o", "gnd", 5.0)
    n, k = b.graph.num_switches, b.switch_index_of("S")

    def sw(t):
        m = p.SwitchStateMask(n)
        m.set(k, bool(t - TB * math.floor(t / TB) < T_ON))
        return m

    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build_lazy(dt)
    o = p.SimulationOptions(t_start=0.0, t_end=n_per * TB, dt=dt)
    o.discontinuity_restart = True
    o.enable_substep_state_correction = split
    o.split_snap = SNAP
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.run_transient(cache, b.graph, b.pool, o, sw)


@pytest.mark.parametrize("dt", [2e-6, 1.3e-6])
def test_a_diode_turn_off_after_a_located_gate_edge_is_located(dt):
    res = _freewheel(dt)
    off = _times(res, False)
    assert len(off) == 6
    # the rule alone at dt/1000 takes each edge and the zero at fine grid
    # points: within ~2 dt_f of the true instants
    dt_f = dt / 1000
    ref = _times(_freewheel(dt_f, split=False), False)
    assert len(ref) == len(off)
    # the located zero is the dt-solution's: its O(dt^2) integration error
    # measured 4.8e-4 dt (2 us) / 1.5e-3 dt (1.3 us); before F-Z-11 the records
    # sat at the step end (0.65 / 0.85 dt), and two zeros in a window step's
    # stage 1 were not located at 1.3 us
    assert np.max(np.abs(ref - off)) / dt <= 5e-3
    assert res.counters["split_locator_giveups"] == 0
