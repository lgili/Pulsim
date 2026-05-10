"""Phase 4 of `add-property-based-testing`: passivity per element.

A passive resistor always has `v · i ≥ 0` — it absorbs energy. A
capacitor's `v · i` swings sign over a cycle but cycle-averages to
zero (no dissipation). Same for an ideal inductor.

We pin the resistor passivity directly: across a randomized RC
charging from zero to a positive source, the resistor's instantaneous
power is non-negative at every step.
"""

from __future__ import annotations

import math

import pytest

pulsim = pytest.importorskip("pulsim")
np = pytest.importorskip("numpy")
hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, HealthCheck

from .strategies import gen_passive_rc, make_quick_options


@given(gen_passive_rc())
@settings(max_examples=15, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_resistor_dissipates_non_negative_power(generated):
    """RC charging: P_R(t) = (V_in - V_C(t))² / R ≥ 0 at every step.

    Catches sign-error regressions in resistor stamping (negative
    conductance would produce P < 0 violating thermodynamics).
    """
    R = generated.parameters["R"]
    C = generated.parameters["C"]
    V = generated.parameters["V"]
    tau = R * C

    opts = make_quick_options(tstop=3 * tau, dt=tau / 100)
    sim = pulsim.Simulator(generated.circuit, opts)
    dc = sim.dc_operating_point()
    assert dc.success
    run = sim.run_transient(dc.newton_result.solution)
    assert run.success

    in_idx = generated.circuit.get_node("in")
    out_idx = generated.circuit.get_node("out")
    states = np.array(run.states)
    v_in = states[:, in_idx]
    v_C = states[:, out_idx]
    P_R = (v_in - v_C) ** 2 / R
    # Strict ≥ 0; allow tiny round-off (V/R · 1e-12).
    eps = (abs(V) / R) ** 2 * 1e-12 + 1e-30
    assert np.all(P_R >= -eps), (
        f"Resistor power went negative: min = {P_R.min():.3e} "
        f"for R={R}, C={C}, V={V}")
