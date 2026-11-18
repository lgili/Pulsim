"""Modulator primitives (add-unified-event-queue task 6.1): a carrier PWM
reading a DutyRegister, and an EdgeList, as switch_fn on every engine."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import pulsim as p

T = 1e-5


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    b.add_switch("Q", "in", "sw", 1e2, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e2, 1e-9)
    b.add_inductor("L", "sw", "out", 1e-3)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", 10.0)
    return b


def _probe_times(d):
    ts = [k * 1e-7 for k in range(400)]
    for k in range(10):
        for f in (0.0, d, d - 0.02, 0.98):
            e = (k + f) * T
            ts += [e, math.nextafter(e, 0.0), math.nextafter(e, 1.0)]
    return ts


@pytest.mark.parametrize("d", [0.0, 0.25, 0.731, 1.0])
def test_carrier_pwm_equals_make_pwm_switch_fn(d):
    g = p.make_carrier_pwm_fn(T, p.DutyRegister(d), 1, 3)
    ref = p.make_pwm_switch_fn(1.0 / T, d, 1, 3)
    for t in _probe_times(d):
        assert g(t) == ref(t), t
        assert g.next_edge_after(t) == ref.next_edge_after(t), t
    assert g.follows_controller


def test_carrier_pwm_leg_equals_dead_time_pair():
    g = p.make_carrier_pwm_fn(T, p.DutyRegister(0.4), 0, 3, low_side=2,
                              dead_time=1e-7)
    ref = p.make_dead_time_pwm_pair_fn(1.0 / T, 0.4, 0, 2, 3, 1e-7)
    for t in _probe_times(0.4):
        assert g(t) == ref(t), t
        assert g.next_edge_after(t) == ref.next_edge_after(t), t


def test_duty_register_latches():
    now = p.DutyRegister(0.3)
    g = p.make_carrier_pwm_fn(T, now, 0, 1)
    now.write(0.6, 0.0)
    assert now.value == 0.6
    assert g(0.45 * T).get(0)                 # the new duty, at once
    held = p.DutyRegister(0.3, latch="carrier_start", period=T)
    h = p.make_carrier_pwm_fn(T, held, 0, 1)
    held.write(0.6, 0.1 * T)
    assert not h(0.45 * T).get(0)             # 0.3 until the next period
    assert h(1.45 * T).get(0)
    assert held.read(1.45 * T) == 0.6
    with pytest.raises(ValueError, match="carrier_start"):
        p.DutyRegister(0.5, latch="carrier_start")
    with pytest.raises(ValueError, match="latch must be"):
        p.DutyRegister(0.5, latch="later")
    other = p.DutyRegister(0.5, latch="carrier_start", period=2 * T)
    with pytest.raises(ValueError, match="different carrier"):
        p.make_carrier_pwm_fn(T, other, 0, 1)
    with pytest.raises(ValueError, match="outside the circuit"):
        p.make_carrier_pwm_fn(T, now, 4, 2)


def test_edge_list_mask_and_edges():
    g = p.make_edge_list_fn([(2e-6, 1, True), (1e-6, 0, True),
                             (3e-6, 0, False), (3e-6, 0, True)], 2)
    assert g(0.0) == p.SwitchStateMask(2)
    assert g(1e-6).get(0) and not g(1e-6).get(1)
    assert g(2.5e-6).get(0) and g(2.5e-6).get(1)
    assert g(3e-6).get(0)                     # the last edge of an instant wins
    assert g.next_edge_after(0.0) == 1e-6
    assert g.next_edge_after(1e-6) == 2e-6
    assert g.next_edge_after(3e-6) == math.inf
    assert not g.follows_controller
    with pytest.raises(ValueError, match="outside the circuit"):
        p.make_edge_list_fn([(1e-6, 3, True)], 2)
    with pytest.raises(ValueError, match="not finite"):
        p.make_edge_list_fn([(math.nan, 0, True)], 2)


def _sim(b, engine, sw, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opts = dict(dt=1e-7) if engine == "pwl" else {}
        return p.simulate(b, 2e-4, engine=engine, switch_fn=sw, **opts, **kw)


@pytest.mark.parametrize("engine", ["pwl", "trbdf2", "dsed"])
def test_constant_duty_run_is_bit_identical_to_the_helper(engine):
    b = _buck()
    q = b.switch_index_of("Q")
    n = b.graph.num_switches
    ref = _sim(b, engine, p.make_pwm_switch_fn(1.0 / T, 0.4, q, n))
    got = _sim(b, engine, p.make_carrier_pwm_fn(T, p.DutyRegister(0.4), q, n))
    assert np.array_equal(np.asarray(got.times), np.asarray(ref.times))
    assert np.array_equal(np.asarray(got.states), np.asarray(ref.states))


def _duty(k):
    return 0.3 if k % 2 == 0 else 0.6


@pytest.mark.parametrize("engine", ["trbdf2", "dsed"])
def test_tick_written_register_equals_the_edge_list(engine):
    """A tick at k*T writes duty d(k); the same gate as an EdgeList."""
    b = _buck()
    q = b.switch_index_of("Q")
    n = b.graph.num_switches
    reg = p.DutyRegister(_duty(0))

    def tick(t, x):
        reg.write(_duty(int(round(t / T))), t)

    got = _sim(b, engine, p.make_carrier_pwm_fn(T, reg, q, n),
               step_observer=tick, controller_period=T)
    edges = []
    for k in range(20):
        edges += [(k * T, q, True), ((k + _duty(k)) * T, q, False)]
    ref = _sim(b, engine, p.make_edge_list_fn(edges, n))
    assert list(got.control_tick_counts) == [20]
    x_got = np.asarray(got.states)[-1]
    x_ref = np.asarray(ref.states)[-1]
    assert np.allclose(x_got, x_ref, rtol=1e-6, atol=1e-9)


def test_the_graph_gate_reads_a_register_kept_current_by_the_graph():
    """ChainPwmGate legs read a DutyRegister the graph writes after every
    call that changes what the duty's sinks see."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 1.0)
    b.add_switch("S", "in", "o", 1e3, 1e-9)
    b.add_resistor("R", "o", "gnd", 1.0)
    ch = p.MixedDomainBlockChain(rate=p.Rate(T))
    ch.add("g", p.Gain(k=0.0), inputs=dict(x="o"), output="duty")
    ch.make_step_observer(b)
    cxx = ch._cxx_chain
    cxx.set_output("duty", 0.3)
    gate = cxx.make_pwm_gate(1, [("duty", 0, T, 0.0)])
    assert gate.next_edge_after(0.0) == p.make_pwm_switch_fn(
        1.0 / T, 0.3, 0, 1).next_edge_after(0.0)
    cxx.set_output("duty", 0.6)                     # a write after the gate
    assert gate(0.45 * T).get(0)
    saved = cxx.save_state()
    cxx.reset()                                     # duty 0: never ON
    assert not gate(0.0).get(0) and not gate(0.45 * T).get(0)
    cxx.restore_state(saved)
    assert gate(0.45 * T).get(0)
    with pytest.raises(ValueError, match="two legs"):
        cxx.make_pwm_gate(1, [("duty", 0, T, 0.0), ("duty", 0, T, 0.0)])
