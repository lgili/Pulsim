"""Findings of the restart-rule scoreboard investigation
(openspec/changes/update-number-changing-release/fixes/restart-scoreboard.md).

The storage convention (release group B): a stored sample at an event is the
LEFT limit and ``result.right_limit_states`` holds the RIGHT limit, the
state where the next step starts; ``pulsim.with_event_limits`` inserts it one
ulp after the event as a value of the trajectory.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p

F_SB_1 = pytest.mark.xfail(strict=True, reason=(
    "finding F-SB-1: under the restart rule the fixed engine records as the right limit "
    "the start of the restart step's quadrature line, 2·w1·X1 + (1 − 2·w1)·X2, at every "
    "restart step (the event's and the window's next two): a capacitor voltage, continuous, "
    "reads 0 V left / 0.163 V right at the closing and jumps at two instants with no event"))


def _rc_closing():
    # 10 V -> ideal switch -> 1 ohm -> 1 uF from 0 V; the gate closes at 2.25 us, dt = tau/2
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S", "vin", "a", 0.0, 0.0)
    b.add_resistor("R", "a", "c", 1.0)
    b.add_capacitor("C", "c", "gnd", 1e-6, 0.0)
    k = b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(1)
        m.set(k, t >= 2.25e-6)
        return m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=6e-6, engine="pwl", dt=0.5e-6, switch_fn=fn,
                          discontinuity_restart=True)


@F_SB_1
def test_a_continuous_state_has_equal_left_and_right_limits():
    r = _rc_closing()
    assert r.right_limit_index is not None and len(r.right_limit_index) > 0
    v = p.with_event_limits(r)
    t, vc = np.asarray(v.times), np.asarray(v.v("c"))
    # each right-limit row sits one ulp after its left limit
    k = np.flatnonzero(t[1:] == np.nextafter(t[:-1], np.inf))
    assert k.size == len(r.right_limit_index)
    np.testing.assert_allclose(vc[k + 1], vc[k], rtol=0.0, atol=1e-9)


# ---- findings of the exact scoreboard's triage (three-part measure, 2026-11-20) ----

import dataclasses  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
RESTART = dict(discontinuity_restart=True)
SPLIT = dict(discontinuity_restart=True, enable_substep_state_correction=True)


def _netlist(rel, t_end):
    sys.path[:0] = [str(BENCH / "tools"), str(BENCH)]
    import v1_reader
    return dataclasses.replace(v1_reader.load(BENCH / rel), t_end=t_end)


def _flips(res, device, builder_index, col):
    """(instant, left, right, entered state) at each flip of ``device`` in
    ``with_event_limits(res)``: the instant is the row before the change, the
    right value the one-ulp row after it (None when the engine records none)."""
    v = p.with_event_limits(res)
    t = np.asarray(v.times)
    s = np.asarray(v.switch_states)[:, builder_index]
    y = np.asarray(col(v))
    out = []
    for j in np.flatnonzero(s[1:] != s[:-1]):
        right = y[j + 1] if t[j + 1] == np.nextafter(t[j], np.inf) else None
        out.append((t[j], y[j], right, bool(s[j + 1])))
    return out


F_SB_2 = pytest.mark.xfail(strict=True, reason=(
    "finding F-SB-2: a switch_fn edge inside a pwl step is not placed at the edge: the split rule "
    "applies it from the START of the step it falls in (vcswitch_resistor: the gate crosses its "
    "threshold 0.5 ns into a 100 ns step, delta = -0.005 h), the restart rule from the next step "
    "(+0.995 h); control_tick='state' with the split rule places it exactly"))


# release: placed exactly under the default control_tick='state' (F-SB-2 stays the PR line's)
@pytest.mark.parametrize("kw", [SPLIT, RESTART], ids=["split", "restart"])
def test_a_scheduled_switch_edge_is_placed_at_the_edge(kw):
    edge, h = 20.0005e-6, 1e-7                      # vcswitch_resistor: 1 ns ramp, threshold at its middle
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_switch("S", "vin", "out", 1000.0, 1e-9)
    b.add_resistor("R", "out", "gnd", 100.0)
    k = b.switch_index_of("S")
    fn = p.make_pwm_switch_fn(20e3, 0.6, k, b.graph.num_switches, edge)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=30e-6, engine="pwl", dt=h, switch_fn=fn, **kw)
    on = [f for f in _flips(r, "S", k, lambda v: v.v("out")) if f[3]]
    assert len(on) == 1
    assert abs(on[0][0] - edge) <= 1e-3 * h, (on[0][0] - edge) / h


F_SB_3 = pytest.mark.xfail(strict=True, reason=(
    "finding F-SB-3: at a diode turn-off behind a transient shorter than the step (a g_off "
    "time constant: ll06 L1*(g_off,S1 + g_off,D1) = 13.6 ns, dt 200 ns; diode_half_wave_rl_load "
    "L1*g_off = 1 ns), the engine records as the right limit a value the truth does not have there "
    "(ll06 V(sw): truth 0 V, continuous; pwl+restart 48.0 V = the settled value, pwl+split 27.9 V = "
    "neither; diode_half_wave_rl_load dsed V(out): truth -0.314 V, engine -3.1e-6 V, the projected "
    "value) and writes no projection record, so the scoreboard's rule (1) cannot price it as a "
    "collapsed transient"))


@F_SB_3
@pytest.mark.parametrize("rel,engine,kw,t_end,col,scale", [
    ("circuits/switching/buck/ll06_buck_vcswitch.yaml", "pwl", RESTART, 2e-3, "sw", 48.0),
    ("circuits/switching/buck/ll06_buck_vcswitch.yaml", "pwl", SPLIT, 3e-5, "sw", 48.0),
    ("circuits/diodes/diode_half_wave_rl_load.yaml", "dsed", {}, 0.012, "out", 10.0),
], ids=["ll06-restart", "ll06-split", "rl_load-dsed"])
def test_a_collapsed_transient_at_a_turn_off_is_recorded(rel, engine, kw, t_end, col, scale):
    case = _netlist(rel, t_end)
    b, _, _ = case.build()
    k = b.switch_index_of("D1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = case.run(engine, **kw)
    offs = [f for f in _flips(r, "D1", k, lambda v: v.v(col)) if not f[3] and f[2] is not None]
    assert offs
    recs = np.array([rec.t for rec in (r.reinit_records or [])])
    for te, left, right, _ in offs:
        # the column is continuous at the turn-off: right == left, or a projection is reported there
        reported = recs.size and np.any(np.abs(recs - te) <= 4 * np.spacing(te))
        assert abs(right - left) <= 1e-3 * scale or reported, (te, left, right)


F_SB_4 = pytest.mark.xfail(strict=True, reason=(
    "finding F-SB-4: the split rule locates diode_rectifier's D1 turn-on 0.227 h late (24.6028 us "
    "against the truth's 24.3754 us, dt = 1 us; the exact engine: within 1e-12 h); V(out) is "
    "continuous there, so the instants part prices it 0 and the limits part reads the drift over "
    "the displacement (0.016 of span)"))


@F_SB_4
def test_the_split_rule_locates_a_diode_turn_on():
    case = _netlist("circuits/diodes/diode_rectifier.yaml", 3e-5)
    b, _, _ = case.build()
    k = b.switch_index_of("D1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = case.run("pwl", **SPLIT)
    on = [f[0] for f in _flips(r, "D1", k, lambda v: v.v("out")) if f[3] and f[0] > 1e-5]
    truth = 2.4375423059672187e-05                  # exact_truth (mpmath), the D1+ after the first discharge
    assert abs(on[0] - truth) <= 1e-2 * case.dt, (on[0] - truth) / case.dt
