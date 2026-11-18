"""G30: simulate(control_tick='state') ticks the label-time graphs
(closed-loop helpers) at the state time and narrows the split's snap.
On a closed-loop buck the pwl+split duty history then sits on the
trbdf2 one (tick on x(kT)), where label time is a step early."""
from __future__ import annotations

import numpy as np
import pytest

import pulsim as p

T = 1e-4


def _run(engine, **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    b.add_switch("Q", "in", "sw", 1e2, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e2, 1e-9)
    b.add_inductor("L", "sw", "out", 1e-3)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", 10.0)
    loop = p.bind_pi_to_switch(
        b, pi=p.PIController(Kp=0.02, Ki=20.0, output_min=0.05,
                             output_max=0.95),
        measured="out", setpoint=12.0, switch="Q", freq=1 / T)
    p.simulate(b, 2e-3, engine=engine, closed_loops=[loop], **kw)
    return np.asarray([d for _, d in loop.duty_history])


def test_state_time_puts_pwl_split_on_the_trbdf2_duties():
    ref = _run("trbdf2", rtol=1e-10, atol=1e-14)
    split = dict(dt=T / 50, discontinuity_restart=True,
                 enable_substep_state_correction=True)
    label = _run("pwl", **split)
    state = _run("pwl", control_tick="state", **split)
    n = min(len(ref), len(label), len(state))
    e_label = np.abs(label[:n] - ref[:n]).max()
    e_state = np.abs(state[:n] - ref[:n]).max()
    assert e_state < e_label / 10, (e_label, e_state)
    # the default is the label-time run
    assert _run("pwl", **split).tobytes() == label.tobytes()


def test_control_tick_takes_label_or_state():
    with pytest.raises(ValueError, match="control_tick"):
        _run("pwl", dt=T / 50, control_tick="exact")


def test_a_reused_loop_in_state_time_takes_this_runs_step():
    # review: the helper's solver dt was set only while its chain was in
    # label time, so a loop reused in state time (composed with a user
    # chain) kept the previous run's dt
    def build():
        b = p.CircuitBuilder()
        b.add_voltage_source("Vin", "in", "gnd", 24.0)
        b.add_switch("Q", "in", "sw", 1e2, 1e-9)
        b.add_diode("D", "gnd", "sw", 1e2, 1e-9)
        b.add_inductor("L", "sw", "out", 1e-3)
        b.add_capacitor("C", "out", "gnd", 47e-6)
        b.add_resistor("R", "out", "gnd", 10.0)
        loop = p.bind_pi_to_switch(
            b, pi=p.PIController(Kp=0.02, Ki=20.0, output_min=0.05,
                                 output_max=0.95),
            measured="out", setpoint=12.0, switch="Q", freq=1 / T)
        ch = p.MixedDomainBlockChain(rate=p.Rate(T))
        ch.add("g", p.Gain(k=1.0), inputs=dict(x="node:out"), output="y")
        return b, loop, ch.make_step_observer(b)

    def state_run(b, loop, obs):
        loop.step_observer._graph.reset()     # same start for both
        p.simulate(b, 1e-3, dt=T / 200, closed_loops=[loop], step_observer=obs,
                   control_tick="state")
        return [d for _, d in loop.duty_history]

    fresh = state_run(*build())
    b, loop, obs = build()
    p.simulate(b, 1e-3, dt=T / 50, closed_loops=[loop], step_observer=obs)
    assert state_run(b, loop, obs) == fresh
