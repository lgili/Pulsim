"""A TR-BDF2 snapshot carries the step controller's memory
(refactor-unified-transient-stepper task 12, C++ side).

Before it, a resumed variable-step run restarted its controller at
h_max/100 and re-learned its step; with it, the resumed run proposes the
step the stopped run would have taken next, so it continues on the same
step sequence.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "out", 100e-6)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", 5.0)
    n, k, T = b.graph.num_switches, b.switch_index_of("S"), 1e-5

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, bool((t / T) - np.floor(t / T + 1e-9) < 0.5 - 1e-9))
        return m
    return b, fn


def _run(t_end, **kw):
    b, fn = _buck()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # dt = h_max: the step bounds do not depend on the span
        return p.simulate(b, t_end=t_end, engine="trbdf2", dt=4e-7,
                          switch_fn=fn, **kw)


@pytest.mark.parametrize("k_after", [3, 40])
def test_resume_continues_the_step_sequence(k_after):
    full = _run(4e-4)
    tf = np.asarray(full.times)
    ts = float(tf[int(np.searchsorted(tf, 2e-4)) + k_after])  # a step time
    first = _run(ts)
    assert np.asarray(first.times)[-1] == ts
    resumed = _run(4e-4, resume_from=first.final_snapshot)
    tr = np.asarray(resumed.times)
    i0 = int(np.searchsorted(tf, ts))
    # the first resumed step is the continuous run's next step (it was
    # h_max/100 before the controller block), to rounding of the split
    h_full = tf[i0 + 1] - tf[i0]
    h_res = tr[1] - tr[0]
    assert h_res == pytest.approx(h_full, rel=1e-6)
    # and the run ends where the continuous one does
    xf = np.asarray(full.states)[-1]
    xr = np.asarray(resumed.states)[-1]
    assert np.allclose(xr, xf, rtol=1e-6, atol=1e-9)


def test_an_explicit_h_init_still_wins():
    # 10.2 T, between edges: a split AT an edge (10 T) lands that edge with
    # an h_min first step instead (test_trbdf2_resume_at_gate_edge.py; it
    # took 1e-9 here before, 1.0e-4 V off the uninterrupted run, 3.5e-7 now)
    first = _run(1.02e-4)
    resumed = _run(2e-4, resume_from=first.final_snapshot, dt_init=1e-9)
    tr = np.asarray(resumed.times)
    assert tr[1] - tr[0] == pytest.approx(1e-9, rel=1e-6)
