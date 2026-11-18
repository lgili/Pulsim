"""Controllers on the exact engine (add-unified-event-queue group 3).

The exact engine's scheduler takes its control ticks, gate edges and
source corners from one event queue: ticks land like gate edges, a tick
runs before a gate edge at the same instant (D2), and the gate is sampled
again after the ticks of an instant.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import pulsim as p


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    b.add_switch("Q", "in", "sw", 1e2, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e2, 1e-9)
    b.add_inductor("L", "sw", "out", 1e-3)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", 10.0)
    return b


class _Pwm:
    """A Python PWM with declared edges whose duty a tick writes."""

    def __init__(self, n, idx, T):
        self.n, self.idx, self.T, self.D = n, idx, T, 0.5

    def __call__(self, t):
        m = p.SwitchStateMask(self.n)
        m.set(self.idx, (t / self.T) % 1.0 < self.D)
        return m

    def next_edge_after(self, t):
        k = math.floor(t / self.T)
        for c in (k * self.T + self.D * self.T, (k + 1) * self.T,
                  (k + 1) * self.T + self.D * self.T):
            if c > t:
                return c
        return (k + 2) * self.T


def _run(b, sw, tick, T_tick, t_end):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end, engine="dsed", switch_fn=sw,
                          step_observer=tick, controller_period=T_tick)


def test_ticks_land_on_their_instants_and_count_ceil():
    b = _buck()
    T = 1e-4 / 3                        # not a binary fraction
    seen = []
    sw = _Pwm(b.graph.num_switches, b.switch_index_of("Q"), 1e-5)
    r = _run(b, sw, lambda t, x: seen.append(t), T, 1e-3)
    n = len(seen)
    assert n == math.ceil(1e-3 / T) == 30
    assert list(r.control_tick_counts) == [n]
    # the queue's instants are (0 + 0) + k*T; the step lands on them
    for k, t in enumerate(seen):
        assert abs(t - k * T) <= 1e-12
    # every tick instant is a recorded sample (it is an event)
    times = np.asarray(r.times)
    for t in seen[1:]:
        assert np.min(np.abs(times - t)) <= 1e-12


def test_a_tick_runs_before_the_gate_edge_of_its_instant():
    # Tick period = PWM period: every tick sits on a rising edge. The tick
    # at k = 5 writes duty 0, so the switch must stay OFF from that instant
    # on: the rising edge of the old duty is not taken first.
    b = _buck()
    T = 1e-5
    sw = _Pwm(b.graph.num_switches, b.switch_index_of("Q"), T)

    def tick(t, x):
        if round(t / T) >= 5:
            sw.D = 0.0

    r = _run(b, sw, tick, T, 1e-4)
    q = b.switch_index_of("Q")
    times = np.asarray(r.times)
    q_on = np.asarray(r.switch_states)[:, q].astype(bool)
    after = q_on[times > 5 * T + 1e-12]
    assert after.size and not after.any()
    assert q_on[times < 5 * T - 1e-9].any()


def test_closed_loop_regulates_on_dsed_like_trbdf2():
    def loop(b):
        return p.control.bind_pi_to_switch(
            b, pi=p.PIController(Kp=0.02, Ki=20.0, output_min=0.05,
                                 output_max=0.95),
            measured="out", setpoint=12.0, switch="Q", freq=10e3)

    out = {}
    for eng in ("dsed", "trbdf2"):
        b = _buck()
        lp = loop(b)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = p.simulate(b, 10e-3, engine=eng, closed_loops=[lp])
        assert list(r.control_tick_counts) == [100]
        out[eng] = np.asarray([d for _t, d in lp.duty_history])
    # two engines, one controller: the duty sequences agree closely
    assert np.max(np.abs(out["dsed"] - out["trbdf2"])) < 1e-3


def test_every_step_observer_still_refused_on_dsed():
    b = _buck()
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(b, 1e-4, engine="dsed",
                   step_observer=lambda t, x: None)
    assert (ei.value.engine, ei.value.feature) == ("dsed", "step_observer")


def test_controllers_with_t_start_refused_on_dsed():
    b = _buck()
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(b, 2e-4, engine="dsed", t_start=1e-4,
                   step_observer=lambda t, x: None, controller_period=1e-5)
    assert ei.value.feature == "step_observer"
