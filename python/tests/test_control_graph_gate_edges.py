"""The control graph's PWM gate states its edges (add-unified-event-queue
group 6: a modulator reading a duty register)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p


def _gate(duty, period=1e-5, phase=0.0):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 1.0)
    b.add_switch("S", "in", "o", 1e3, 1e-9)
    b.add_resistor("R", "o", "gnd", 1.0)
    ch = p.MixedDomainBlockChain(rate=p.Rate(period))
    ch.add("g", p.Gain(k=0.0), inputs=dict(x="o"), output="duty")
    ch.make_step_observer(b)
    ch._cxx_chain.set_output("duty", duty)
    return ch, ch._cxx_chain.make_pwm_gate(1, [("duty", 0, period, phase)])


@pytest.mark.parametrize("duty", (0.0, 0.25, 0.5, 0.731, 1.0))
def test_gate_edges_equal_the_pwm_helper(duty):
    ch, gate = _gate(duty)
    ref = p.make_pwm_switch_fn(1e5, duty, 0, 1)
    ts = np.concatenate([np.linspace(0.0, 1e-4, 997),
                         np.arange(20) * 1e-5, np.arange(20) * 1e-5 + duty * 1e-5])
    for t in ts:
        assert gate.next_edge_after(t) == ref.next_edge_after(t)


def test_mask_is_constant_between_gate_edges():
    ch, gate = _gate(0.37)
    t = 0.0
    for _ in range(60):
        e = gate.next_edge_after(t)
        assert e > t
        m = gate(t + 1e-3 * (e - t))
        for f in (0.25, 0.5, 0.75, 0.999):
            assert gate(t + f * (e - t)) == m
        t = e


def _loop_buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    b.add_switch("Q", "in", "sw", 1e2, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e2, 1e-9)
    b.add_inductor("L", "sw", "out", 1e-3)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", 10.0)
    loop = p.control.bind_pi_to_switch(
        b, pi=p.PIController(Kp=0.02, Ki=20.0, output_min=0.05,
                             output_max=0.95),
        measured=lambda x, i=b.node_id_of("out"): x[i], setpoint=12.0,
        switch="Q", freq=20e3)
    return b, loop


def test_closed_loop_edges_are_landed_not_sampled_on_trbdf2():
    b, loop = _loop_buck()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, 5e-3, closed_loops=[loop], engine="trbdf2")
    st = r._trbdf2_stats
    # The gate has an oracle: no pulse width is learned by sampling.
    assert st["gate_pulse_min"] == 0.0
    assert st["gate_sample_spacing_max"] == 0.0
    # Two edges per carrier period (duty strictly inside (0, 1) after
    # the first periods): about 2 * 100.
    assert 150 <= st["n_gate_events"] <= 200
