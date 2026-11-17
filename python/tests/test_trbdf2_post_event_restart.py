"""TR-BDF2 restart after an event (update-trbdf2-post-event-restart).

After every gate edge the variable-step engine collapsed its step and
climbed back: the companion history still held the PRE-edge derivative
(inductor voltage, capacitor current), so the first post-edge step had
an O(h) error the controller rightly refused, and the post-edge memory
stored the accepted step, which can only shrink. On the 24 V buck of
the core review (numerics/p6_ratchet.py, p5_trbdf2_edges.py) the
rising-edge step was 8.26e-13 s for the whole 400-period run.

Counters, not wall time: accepted steps, solves and the first step
after each edge.
"""

import numpy as np
import pytest
from scipy.linalg import expm

import pulsim as p

VIN, L, C, GON, GOFF, T, T0 = 24.0, 100e-6, 47e-6, 1e3, 1e-9, 1e-5, 2e-6
NPER = 400
PERIODS = (1, 2, 3, 5, 10, 50, 100, 399)


def _buck(R, D, **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", VIN)
    b.add_switch("S", "vin", "sw", GON, GOFF)
    b.add_diode("D1", "gnd", "sw", GON, GOFF)
    b.add_inductor("L", "sw", "vout", L)
    b.add_capacitor("C", "vout", "gnd", C)
    b.add_resistor("RL", "vout", "gnd", R)
    n = b.graph.num_switches
    k = b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(n)
        ph = (t - T0) / T
        fr = ph - np.floor(ph + 1e-9)
        m.set(k, bool(t >= T0 - 1e-13 and fr < D - 1e-9))
        return m
    return p.simulate(b, t_end=T0 + NPER * T, engine="trbdf2",
                      switch_fn=fn, **kw)


def _ccm_exact():
    """x = [i_L, v_C] at T0 + NPER*T: CCM buck, piecewise LTI, expm."""
    def prop(gs, gd, x, dt):
        k = 1 / (gs + gd)
        M = np.zeros((3, 3))
        M[:2, :2] = [[-k / L, -1 / L], [1 / C, -1 / (1.0 * C)]]
        M[0, 2] = gs * VIN * k / L
        E = expm(M * dt)
        return E[:2, :2] @ x + E[:2, 2]
    x = np.zeros(2)
    for _ in range(NPER):
        x = prop(GON, GOFF, x, 0.5 * T)
        x = prop(GOFF, GON, x, 0.5 * T)
    return x


def _post_edge_step(t, te):
    k = int(np.argmin(np.abs(t - te)))
    assert abs(t[k] - te) < 1e-12, "edge not landed"
    if t[k + 1] - te < 1e-9:  # the committed edge probe (h_probe)
        k += 1
    return t[k + 1] - t[k]


@pytest.fixture(scope="module")
def ccm():
    return _buck(1.0, 0.5)


@pytest.fixture(scope="module")
def dcm():
    return _buck(50.0, 0.3)


def test_ccm_post_edge_step_and_steps_per_period(ccm):
    """The first step after every edge is sized by a true estimate: above
    1e-8 s at every listed period (8.26e-13 s on the rising edge before
    this change) and at most 8 accepted steps per period (18 before)."""
    t = np.asarray(ccm.times)
    for n in PERIODS:
        for te in (T0 + n * T, T0 + (n + 0.5) * T):
            assert _post_edge_step(t, te) > 1e-8
    st = ccm._trbdf2_stats
    assert st["n_accept"] <= 8 * NPER
    assert st["n_derivative_refreshes"] >= st["n_gate_events"]
    assert st["h_post_event_min"] > 1e-8
    assert st["n_refresh_probes"] == 0   # the edge cascade's probe serves


def test_ccm_error_not_above_twice_the_previous(ccm):
    """End-of-run error against the expm reference: at most twice the
    value before this change (9.84e-4, max of the i_L and v_C errors at
    the default tolerances, probes/trbdf2_baseline.json)."""
    xe = _ccm_exact()
    err = max(abs(np.asarray(ccm.i("L"))[-1] - xe[0]),
              abs(np.asarray(ccm.v("vout"))[-1] - xe[1]))
    assert err < 2 * 9.84e-4


@pytest.mark.xfail(strict=True, reason=(
    "not met: the DCM steps below 1e-11 s follow the diode turn-off at "
    "zero inductor current, and the derivative refresh is not applied at "
    "diode events (it excites the L*g_off mode; DECISIONS.md)"))
def test_dcm_no_step_below_1e_11(dcm):
    """Runs of accepted steps between 4e-14 and 1e-12 s before this
    change."""
    assert np.diff(np.asarray(dcm.times)).min() > 1e-11
