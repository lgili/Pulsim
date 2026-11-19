"""A closed loop against an independent reference (G23).

The loop of bind_pi_to_switch drives its switch through a C++ gate that
declares its edges (`next_edge_after`): trbdf2 lands on them and pwl's
event split (`enable_substep_state_correction`) splits at them.

Reference: benchmarks cl_buck_pi (10 V, R_dcr + R_esr = 0.25 ohm, 10 uH,
100 uF, 10 ohm, switch / diode 100 S on, 1e-6 S off; PI kp 0.05, ki 200,
[0, 1]; 100 kHz) integrated outside pulsim: each linear mode's flow by
scipy.linalg.expm, the diode crossings by bisection on that flow, the PI
law re-typed, ticks at kT on x(kT) (state time), duty held from kT
(script: openspec/changes/add-compiled-control-graph/probes/
closed_loop_reference.py). Values of v(out) and the duty at tick k.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulsim as p

T = 1e-5
# k: (duty_k, v(out)(kT))
REF = {
    10: (0.19439924022185115, 2.5754087852607404),
    50: (0.2867647998058633, 3.7074360925242344),
    100: (0.3226620647223434, 4.577242840328066),
    250: (0.33056271829976036, 5.002290972754306),
    499: (0.3299775096028466, 5.000022780161521),
}


def _buck_loop():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 10.0)
    b.add_switch("SH", "vin", "sw", 100.0, 1e-6)
    b.add_diode("D", "gnd", "sw", 100.0, 1e-6)
    b.add_resistor("R_dcr", "sw", "n_l", 0.2)
    b.add_inductor("L1", "n_l", "n_esr", 10e-6)
    b.add_resistor("R_esr", "n_esr", "out", 0.05)
    b.add_capacitor("C1", "out", "gnd", 100e-6)
    b.add_resistor("R", "out", "gnd", 10.0)
    pi = p.PIController(Kp=0.05, Ki=200.0, output_min=0.0, output_max=1.0)
    loop = p.bind_pi_to_switch(b, pi=pi, measured="out", setpoint=5.0,
                               switch="SH", freq=1.0 / T)
    return b, loop


def _errors(engine, **kw):
    """max |v(kT) - ref| and max |d_k - ref| over the pinned ticks."""
    b, loop = _buck_loop()
    r = p.simulate(b, 5e-3, engine=engine, switch_fn=loop.switch_fn,
                   step_observer=loop.step_observer, **kw)
    t, v = np.asarray(r.times), np.asarray(r.v("out"))
    duty = np.asarray(list(loop.duty_history))[:, 1]
    ev = ed = 0.0
    for k, (d_ref, v_ref) in REF.items():
        j = np.searchsorted(t, k * T * (1 + 1e-12), side="right") - 1
        ev = max(ev, abs(v[j] - v_ref))
        ed = max(ed, abs(duty[k] - d_ref))
    return ev, ed


def test_the_gate_declares_its_edges():
    b, loop = _buck_loop()
    gate = loop.switch_fn
    loop.switch_fn._graph._cxx_chain.set_output("duty", 0.3)
    assert gate.next_edge_after(0.0) == pytest.approx(0.3 * T, rel=1e-12)
    assert gate.next_edge_after(0.3 * T) == pytest.approx(T, rel=1e-12)
    assert gate.next_edge_after(7.5 * T) == pytest.approx(8 * T, rel=1e-12)
    # every mask change on a fine grid sits on a declared edge
    ts = np.linspace(0.0, 3 * T, 3001)
    on = np.array([gate(t).get(0) for t in ts])
    for i in np.flatnonzero(on[1:] != on[:-1]):
        e = gate.next_edge_after(ts[i])
        assert ts[i] < e <= ts[i + 1]
    loop.switch_fn._graph._cxx_chain.set_output("duty", 0.0)
    assert gate.next_edge_after(0.5 * T) == pytest.approx(T, rel=1e-12)


def test_trbdf2_meets_the_independent_reference():
    ev, ed = _errors("trbdf2", rtol=1e-10, atol=1e-14)
    assert ev < 1e-6 and ed < 1e-6


def test_trbdf2_lands_every_edge_at_the_commanded_duty():
    b, loop = _buck_loop()
    r = p.simulate(b, 2e-3, engine="trbdf2", switch_fn=loop.switch_fn,
                   step_observer=loop.step_observer)
    duty = np.asarray(list(loop.duty_history))[:, 1]
    assert np.all((duty > 0.0) & (duty < 1.0))
    # two analytic edges per period, each landed (+-1: whether the edge at
    # t_end is counted depends on the last step)
    assert abs(r._trbdf2_stats["n_gate_events"] - 2 * len(duty)) <= 1
    t = np.asarray(r.times)
    on = np.asarray(r.switch_states)[:, 0]
    fall = t[1:][on[:-1] & ~on[1:]]        # first OFF sample after each pulse
    k = np.floor(fall / T + 1e-9)
    realised = fall / T - k
    assert len(fall) == len(duty)
    # within 1 ppm of the period (the post-edge sample sits ~2e-13 s late)
    assert np.max(np.abs(realised - duty[k.astype(int)])) < 1e-6


def test_pwl_split_uses_the_gate_edges():
    # split at the declared edges: within 1e-2 V; the restart rule alone
    # (gate sampled at the step start) is several times further off.
    ev_split, _ = _errors("pwl", dt=5e-7, discontinuity_restart=True,
                          enable_substep_state_correction=True)
    # (the split now follows the rule by default: "alone" says split off)
    ev_restart, _ = _errors("pwl", dt=5e-7, discontinuity_restart=True,
                            enable_substep_state_correction=False)
    assert ev_split < 1e-2
    assert ev_restart > 3 * ev_split


def test_a_loop_starting_at_duty_zero_does_not_grind_on_trbdf2():
    # cl_boost_pi: the first pulses are 0.28 ns wide. Without the gate's
    # edges trbdf2 learned that width and capped every later step at 2x
    # it (5.2e5 steps in 0.3 ms, max_steps at 5.59 ms of 30 ms).
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 5.0)
    b.add_resistor("R_dcr", "vin", "n_l", 0.1)
    b.add_inductor("L1", "n_l", "sw", 47e-6)
    b.add_switch("SL", "sw", "gnd", 100.0, 1e-6)
    b.add_diode("D", "sw", "out", 100.0, 1e-6)
    b.add_capacitor("C", "out", "gnd", 470e-6, 10.0)
    b.add_resistor("R", "out", "gnd", 40.0)
    pi = p.PIController(Kp=0.005, Ki=50.0, output_min=0.0, output_max=0.9)
    loop = p.bind_pi_to_switch(b, pi=pi, measured="out", setpoint=10.0,
                               switch="SL", freq=1.0 / T)
    r = p.simulate(b, 3e-4, engine="trbdf2", switch_fn=loop.switch_fn,
                   step_observer=loop.step_observer)
    t = np.asarray(r.times)
    assert np.diff(t[t > 1e-4]).max() > 1e-8
    assert len(t) < 5e4
