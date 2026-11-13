"""Checkpoint-review findings M3, L4, L5, L6, L8b on the restart/split path.

M3  the restart's stage-2 backward-Euler re-try that fails is counted AND
    warned, and the SDIRK stage 2 it falls back to is kept bit for bit.
L4  an opaque switch_fn edge exactly on a grid point is not "unlocated".
L5  split_event_overflows is warned.
L6  a step_observer that moves the declared b_extra_fn jumps is honoured.
L8b n_restart_factorizations counts the split's sub-interval stages.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import pulsim as p

RULE = dict(discontinuity_restart=True)
SPLIT = dict(discontinuity_restart=True, enable_substep_state_correction=True)
GAMMA = 1.0 - 1.0 / math.sqrt(2.0)


# --- M3 ----------------------------------------------------------------------
def _rect3(dt, trap):
    """The three-phase rectifier of test_fixed_step_discontinuity_order
    (tau = 5 us, dt/tau = 10). `trap` adds an isolated node q: C_q and
    R_q to ground and a VCCS feeding q back into itself with
    gm = C_q/h_b + 1/R_q, so the matrix is singular exactly at the
    re-try's dt = 2 h_b (h_b = (1 - gamma) dt) and nowhere else."""
    b = p.CircuitBuilder()
    for n, ph in (("a", 0.0), ("b", 2.0943951), ("c", 4.1887902)):
        b.add_sine_voltage_source("V" + n, n, "gnd", 0.0, 24.0, 100.0, ph)
        b.add_diode("D" + n + "h", n, "out", 200.0, 1e-6)
        b.add_diode("D" + n + "l", "gnd", n, 200.0, 1e-6)
    b.add_capacitor("Cout", "out", "gnd", 1e-3)
    b.add_resistor("Rload", "out", "gnd", 200.0)
    if trap:
        h_b = dt - GAMMA * dt
        b.add_capacitor("Cq", "q", "gnd", 1e-6)
        b.add_resistor("Rq", "q", "gnd", 1e6)
        b.add_vccs("Gq", "q", "gnd", "q", "gnd", 1e-6 / h_b + 1e-6)
    return p.simulate(b, t_end=40e-3, dt=dt, engine="pwl", **RULE)


def test_a_failed_stage2_retry_is_counted_warned_and_keeps_sdirk():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ref = _rect3(50e-6, False)
    c = ref.counters
    # engine: 19 re-tries, 5 kept, bus peak 23.999 V
    assert c["stage2_retries"] == 19 and c["stage2_be_kept"] == 5
    assert c["stage2_retry_failures"] == 0
    with pytest.warns(UserWarning, match="stage2_retry_failures"):
        bad = _rect3(50e-6, True)
    c = bad.counters
    # Every re-try throws (singular factor) and the SDIRK stage 2 stands:
    # the bus returns to the pre-fallback overshoot, 37.20 V (CHANGELOG).
    assert c["stage2_retry_failures"] == c["stage2_retries"] == 10, c
    assert c["stage2_be_kept"] == 0
    assert max(bad.v("out")) == pytest.approx(37.2032291, abs=1e-6)
    assert np.abs(np.asarray(bad.v("q"))).max() == 0.0


# --- L4 ----------------------------------------------------------------------
def _rl_opaque(te, dt=1e-6):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S", "in", "a", 1e3, 1e-9)
    b.add_resistor("R", "a", "m", 1.0)
    b.add_inductor("L", "m", "gnd", 1e-3)
    n, k = b.graph.num_switches, b.switch_index_of("S")

    def fn(t):        # opaque: no next_edge_after
        m = p.SwitchStateMask(n)
        m.set(k, t >= te)
        return m
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = p.simulate(b, t_end=20e-6, dt=dt, engine="pwl", switch_fn=fn,
                       **SPLIT)
    return r, [str(x.message) for x in w if "could not be located" in str(x.message)]


def test_an_opaque_edge_on_the_grid_is_not_unlocated():
    r, w = _rl_opaque(5e-6 * (1 - 1e-12))   # on the grid (within snap)
    assert r.counters["unlocated_gate_steps"] == 0 and not w


def test_an_opaque_edge_inside_a_step_is_unlocated():
    r, w = _rl_opaque(5.37e-6)
    assert r.counters["unlocated_gate_steps"] == 1 and len(w) == 1


# --- L5 ----------------------------------------------------------------------
class _Burst:
    """Ten gate toggles inside one 1 us step (more than the 8 the split takes)."""

    def __init__(self, n, k, t0=5.05e-6, gap=0.08e-6, count=10):
        self.n, self.k = n, k
        self.edges = [t0 + i * gap for i in range(count)]

    def __call__(self, t):
        m = p.SwitchStateMask(self.n)
        m.set(self.k, sum(t >= e for e in self.edges) % 2 == 1)
        return m

    def next_edge_after(self, t):
        return next((e for e in self.edges if e > t), math.inf)


def test_split_event_overflow_is_warned():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S", "in", "a", 1e3, 1e-9)
    b.add_resistor("R", "a", "m", 1.0)
    b.add_inductor("L", "m", "gnd", 1e-3)
    fn = _Burst(b.graph.num_switches, b.switch_index_of("S"))
    with pytest.warns(UserWarning, match="split_event_overflows"):
        r = p.simulate(b, t_end=20e-6, dt=1e-6, engine="pwl", switch_fn=fn,
                       **SPLIT)
    assert r.counters["split_event_overflows"] == 1


# --- L6 ----------------------------------------------------------------------
def test_an_observer_that_moves_a_declared_jump_is_honoured():
    te, t_end = 2.037e-6, 6.4e-6

    def run(late):
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "in", "gnd", 0.0)
        b.add_resistor("R", "in", "a", 1e3)
        b.add_capacitor("C", "a", "gnd", 1e-9)
        sched = {"te": math.inf if late else te}

        def bx(t):
            v = np.zeros(3)
            v[2] = 10.0 if t >= sched["te"] else 0.0
            return v
        bx.next_jump_after = lambda t: sched["te"] if t < sched["te"] else math.inf

        def obs(t, x):
            if t >= 1e-6:
                sched["te"] = te           # the schedule learns its jump
        return p.simulate(b, t_end=t_end, dt=1e-7, engine="pwl",
                          b_extra_fn=bx, step_observer=obs if late else None,
                          **SPLIT)
    early, late = run(False), run(True)
    assert early.n_split_steps == late.n_split_steps == 1
    np.testing.assert_array_equal(late.v("a"), early.v("a"))


# --- L8b ---------------------------------------------------------------------
def test_restart_factorizations_count_the_split_sub_intervals():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S", "in", "a", 1e3, 1e-9)
    b.add_resistor("R", "a", "m", 1.0)
    b.add_inductor("L", "m", "gnd", 1e-3)
    fn = p.make_pwm_switch_fn(1 / 2.0, 0.5, b.switch_index_of("S"),
                              b.graph.num_switches, 5.37e-6)
    rule = p.simulate(b, t_end=20e-6, dt=1e-6, engine="pwl", switch_fn=fn,
                      **RULE)
    split = p.simulate(b, t_end=20e-6, dt=1e-6, engine="pwl", switch_fn=fn,
                       **SPLIT)
    assert split.n_split_steps == 1
    # The split adds the sub-interval stage factor(s) at 2*gamma*h_sub.
    assert split.n_restart_factorizations > rule.n_restart_factorizations
