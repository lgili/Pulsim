"""trbdf2: a gate edge a rounding error before t_end leaves no sliver step.

The edge is left to the next run (its first step reads the gates at
t_start + h_min / 2). Landing it left a final step of 9.2e-15 s whose
derivative slots were round-off / h, and steady_state(engine='trbdf2')
found no periodic orbit on this buck (DECISIONS #50 of
add-branch-form-ideal-switch).
"""
import warnings

import numpy as np
import pytest

import pulsim as p

T, DT = 1e-5, 1e-7


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "out", 100e-6)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", 5.0)
    n, k = b.graph.num_switches, b.switch_index_of("S")

    def fn(t):   # edges 1e-9*T before k*T and k*T + T/2
        m = p.SwitchStateMask(n)
        m.set(k, bool((t / T) - np.floor(t / T + 1e-9) < 0.5 - 1e-9))
        return m
    return b, fn


def test_no_final_sliver_step_before_t_end():
    b, fn = _buck()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=2 * T, dt=DT, switch_fn=fn, engine="trbdf2")
    t = np.asarray(r.times)
    assert t[-1] == 2 * T
    # was 9.2e-15 s (the edge 1e-14 s before t_end was landed)
    assert t[-1] - t[-2] > 1e-12


@pytest.mark.xfail(strict=False, reason=(
    "the sliver is gone (one period from the answer: 4.9e-3 -> 1.6e-5), but "
    "the monodromy probes take different step grids (a 2.9e-14 s step at "
    "the mid-period edge), so the one-period map is not affine in the "
    "probe; DECISIONS #50"))
def test_steady_state_finds_the_orbit_on_trbdf2():
    b, fn = _buck()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.steady_state(b, period=T, dt=DT, switch_fn=fn, engine="trbdf2")
    # pwl measures 1.7e-10 here; the bound is the run's own tolerance
    assert r.residual < 1e-6


def test_the_final_snapshot_resumes_on_pwl():
    """The snapshot of a run that ended next to an edge has the step's own
    slots (the edge is not landed, so no refresh-only probe flags it):
    post_edge_derivatives stays False and pwl resumes from it."""
    b, fn = _buck()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=2 * T, dt=DT, switch_fn=fn, engine="trbdf2")
        s = r.final_snapshot
        assert not s.post_edge_derivatives
        r2 = p.simulate(b, t_end=3 * T, dt=DT, switch_fn=fn, engine="pwl",
                        resume_from=s)
        b3, fn3 = _buck()
        full = p.simulate(b3, t_end=3 * T, dt=DT, switch_fn=fn3, engine="pwl")
    assert np.isfinite(np.asarray(r2.states)).all()
    # measured 6.8e-3 V from the continuous pwl run (the two engines are
    # 1.1e-2 V apart at 3T on this grid); a lost state is volts off
    np.testing.assert_allclose(r2.v("out")[-1], full.v("out")[-1],
                               rtol=0, atol=5e-2)
