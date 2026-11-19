"""A TR-BDF2 run resumed exactly AT a gate edge (add-unified-event-queue,
slice 3, defect (a)).

An opaque gate ``(t % T) < D*T`` evaluated at t = 4.3 T = 8.6e-5 s is still
ON (the remainder rounds one ulp below D*T): the first run ends on the
pre-edge side and commits no edge, the resumed run's first step reads the
gate OFF. Before the fix the zero-time settle decided the diodes under the
post-edge gate but no edge was landed: the history kept the pre-edge
derivatives (7 rejected steps, the final state 3.2e-4 V off the
uninterrupted run; every other split of this run lands within 1e-6 V) and
the edge was not counted. The first step now lands it with an h_min step,
as an edge one ulp after t_start is landed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p

T = 2e-5


def _buck(oracle):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 48.0)
    b.add_mosfet("M1", "vin", "sw")
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9, V_th=0.7)
    b.add_inductor("L1", "sw", "out", 100e-6)
    b.add_capacitor("C1", "out", "gnd", 100e-6)
    b.add_resistor("RL", "out", "gnd", 2.0)
    m1, n = b.switch_index_of("M1"), b.graph.num_switches
    if oracle:
        return b, p.make_pwm_switch_fn(1 / T, 0.3, m1, n)

    def pwm(t):
        m = p.SwitchStateMask(n)
        m.set(m1, (t % T) < 0.3 * T)
        return m
    return b, pwm


def _split(oracle, t1, t_end=8 * T):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b, f = _buck(oracle)
        r0 = p.simulate(b, t_end=t_end, engine="trbdf2", switch_fn=f)
        b, f = _buck(oracle)
        ra = p.simulate(b, t_end=t1, engine="trbdf2", switch_fn=f)
        b, f = _buck(oracle)
        rb = p.simulate(b, t_start=float(ra.final_snapshot.t), t_end=t_end,
                        engine="trbdf2", switch_fn=f,
                        resume_from=ra.final_snapshot)
    return r0, ra, rb


def test_an_opaque_gate_resumed_at_its_edge_lands_it():
    t1 = 4.3 * T
    _, pre = _buck(False)
    assert pre(t1).to_string() != pre(t1 + 1e-12).to_string()   # AT the edge
    r0, ra, rb = _split(False, t1)
    assert not ra.final_snapshot.post_edge_derivatives
    x0, xb = np.asarray(r0.states)[-1], np.asarray(rb.states)[-1]
    # 3.2e-4 V before; the other splits of this run: <= 1.1e-6 V
    assert np.max(np.abs(x0 - xb)) < 2e-5
    # the edge is landed at the start: an h_min step (2.4e-9 s before,
    # then 7 rejects). Its count is not pinned: an opaque gate's edge that
    # falls on a step boundary is settled there, uncounted, and where that
    # happens depends on the last bits of the step sequence.
    tb = np.asarray(rb.times)
    assert tb[1] - tb[0] < 1e-12


@pytest.mark.parametrize("t1", [4.15 * T, 4.3 * T, 4 * T])
def test_an_edge_the_first_run_committed_is_not_landed_again(t1):
    """With declared edges the run that ends on an edge lands it (its
    snapshot says post_edge_derivatives) and the resumed run must not land
    it a second time; a split between edges is unchanged."""
    r0, ra, rb = _split(True, t1)
    tb = np.asarray(rb.times)
    assert tb[1] - tb[0] > 1e-9
    assert (ra._trbdf2_stats["n_gate_events"] + rb._trbdf2_stats["n_gate_events"]
            == r0._trbdf2_stats["n_gate_events"])
    x0, xb = np.asarray(r0.states)[-1], np.asarray(rb.states)[-1]
    assert np.max(np.abs(x0 - xb)) < 2e-5
