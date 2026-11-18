"""trbdf2: a Newton circuit from a zero start (DECISIONS #44 of
add-branch-form-ideal-switch).

48 V behind 1 uF, a closed switch and a Shockley diode, from zero: the
first solve converged in dx (7e-16 V) but its residual is the round-off of
the 2C/h·v terms (4.8e-6 A), and the round-off acceptance measured it
against the zero start state (a scale of 48), so the run aborted at t = 0
with h at h_min, in both switch forms. The failed solve is now retried
once from the step's linear solve.
"""
import warnings

import numpy as np
import pytest

import pulsim as p


def _circuit(form):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 48.0)
    b.add_capacitor("C", "in", "a", 1e-6)
    if form == "branch":
        b.add_ideal_switch("S", "a", "b", 0.02, 1e-7)
    else:
        b.add_switch("S", "a", "b", 50.0, 1e-7)
    b.add_resistor("R", "b", "gnd", 10.0)
    b.add_shockley_diode("D", "b", "gnd", I_S=1e-12, n=1.2)
    m = p.SwitchStateMask(b.graph.num_switches)
    m.set(b.switch_index_of("S"), True)
    return b, p.make_constant_switch_fn(m)


@pytest.mark.parametrize("form", ["conductance", "branch"])
def test_newton_circuit_runs_from_a_zero_start(form):
    b, fn = _circuit(form)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, 1e-5, engine="trbdf2", switch_fn=fn)
        b2, fn2 = _circuit(form)
        ref = p.simulate(b2, 1e-5, engine="pwl", dt=1e-9, switch_fn=fn2)
    assert r._trbdf2_stats["n_newton_start_retries"] >= 1
    # the capacitor charges through the switch and the diode: both engines
    # end on the same state (3.2e-5 V apart against pwl
    # at dt = 1e-9, both forms)
    np.testing.assert_allclose(r.v("a")[-1], ref.v("a")[-1], rtol=0, atol=1e-3)
    np.testing.assert_allclose(r.v("b")[-1], ref.v("b")[-1], rtol=0, atol=1e-3)
