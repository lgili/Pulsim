"""Gate-edge counts against the analytic count (add-unified-event-queue
tasks 0.3, 5.3, 5.4, 5.5).

A PWM buck at T = 20 us over 5.15 periods: 2 edges per period x 5 periods
in (0, t_end) (the rise at t = 0 is the initial gate, not an edge). The
dt (t_end / 1500) puts no edge within split_snap*dt of a grid point.
"""

from __future__ import annotations

import warnings

import pytest

import pulsim as p

T = 2e-5
T_END = 5.15 * T
DT = T_END / 1500
EDGES = 2 * 5
# a pulse narrower than 0.15 T also falls in the 6th period: one more fall
EDGES_NARROW = 2 * 5 + 1


def _buck(duty):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 48.0)
    b.add_mosfet("M1", "vin", "sw")
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9, V_th=0.7)
    b.add_inductor("L1", "sw", "out", 100e-6)
    b.add_capacitor("C1", "out", "gnd", 100e-6)
    b.add_resistor("RL", "out", "gnd", 2.0)
    f = p.make_pwm_switch_fn(1 / T, duty, b.switch_index_of("M1"),
                             b.graph.num_switches)
    return b, f


def _pwl(duty, **kw):
    b, f = _buck(duty)
    return p.simulate(b, t_end=T_END, dt=DT, engine="pwl", switch_fn=f,
                      enable_substep_state_correction=True,
                      discontinuity_restart=True, **kw)


def test_fixed_engine_counts_every_declared_edge():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c = _pwl(0.3).counters
    assert c["n_gate_events"] == EDGES
    assert c["n_gate_events_unresolved"] == 0


def test_two_edges_of_one_bit_inside_one_fixed_step_are_both_counted():
    # a 40 ns pulse in a 68.7 ns step: both edges inside one step in
    # some periods, both landed
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c = _pwl(0.002).counters
    assert 0.002 * T < DT
    assert c["n_gate_events"] == EDGES_NARROW
    assert c["n_gate_events_unresolved"] == 0


def test_a_pulse_narrower_than_the_snap_is_counted_and_reported():
    # 20 fs < split_snap*dt = 69 fs (split_snap 1e-6, F-SB-2): each fall
    # merges into its rise, except the first (its rise is the initial gate;
    # the fall is taken at the step start)
    # legacy subject: the label tick snaps the fall onto its rise (control_tick='state' lands both)
    with pytest.warns(UserWarning, match="n_gate_events_unresolved"):
        c = _pwl(1e-9, control_tick="label").counters
    assert c["n_gate_events"] == EDGES_NARROW
    assert c["n_gate_events_unresolved"] == 5
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # the subject is the gate count; each 0.2 ns pulse cuts L1's current (two
        # cut-set records, 9.2e-13 J of 6.5e-12 J stored): the energy warning's
        warnings.filterwarnings("ignore", message=r".*instantaneous current cut")
        c = _pwl(1e-5).counters
    assert c["n_gate_events"] == EDGES_NARROW
    assert c["n_gate_events_unresolved"] == 0


def test_trbdf2_counts_every_declared_edge():
    b, f = _buck(0.3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=T_END, engine="trbdf2", switch_fn=f)
    assert r._trbdf2_stats["n_gate_events"] == EDGES
