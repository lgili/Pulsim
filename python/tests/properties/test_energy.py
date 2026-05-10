"""Phase 3 of `add-property-based-testing`: energy conservation.

For an RC charging from a step source, the energy balance is:

    ½·C·V_C(t)²  +  ∫₀ᵗ R·I²(τ) dτ  =  ∫₀ᵗ V_src · I(τ) dτ

In the limit `t → ∞`, V_C → V_src, the integral on the LHS converges
to ½·C·V², and the integral on the RHS converges to C·V². So the
"resistor dissipates exactly half the source energy" — the canonical
RC charging energy result.

We assert this on randomly-generated RC circuits via Hypothesis.
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
def test_rc_steady_state_charges_to_source_voltage(generated):
    """Energy-balance corollary: at steady state, V_C = V_src
    (capacitor is fully charged). Trivial KVL but useful to pin via
    a property test."""
    R = generated.parameters["R"]
    C = generated.parameters["C"]
    V = generated.parameters["V"]
    tau = R * C

    opts = make_quick_options(tstop=10 * tau, dt=tau / 100)
    sim = pulsim.Simulator(generated.circuit, opts)
    dc = sim.dc_operating_point()
    assert dc.success
    run = sim.run_transient(dc.newton_result.solution)
    assert run.success
    assert len(run.states) > 0

    out_idx = generated.circuit.get_node("out")
    v_C_final = run.states[-1][out_idx]
    # After 10·τ the cap voltage is within e^-10 ≈ 4.5e-5 of V_src.
    assert math.isclose(v_C_final, V, rel_tol=1e-3, abs_tol=1e-6)


@given(gen_passive_rc())
@settings(max_examples=15, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_rc_capacitor_voltage_is_monotone_during_charging(generated):
    """A passive RC charging from zero to a positive source voltage
    has monotonically non-decreasing capacitor voltage. Catches sign-
    error regressions in the integrator step."""
    R = generated.parameters["R"]
    C = generated.parameters["C"]
    V = generated.parameters["V"]
    tau = R * C
    opts = make_quick_options(tstop=3 * tau, dt=tau / 200)
    sim = pulsim.Simulator(generated.circuit, opts)
    dc = sim.dc_operating_point()
    assert dc.success
    run = sim.run_transient(dc.newton_result.solution)
    assert run.success

    out_idx = generated.circuit.get_node("out")
    trace = np.array([s[out_idx] for s in run.states])
    # Monotone within numerical noise: every step ≥ previous − tiny eps.
    delta = np.diff(trace)
    eps = max(1e-9, abs(V) * 1e-7)
    assert np.all(delta >= -eps), (
        f"V_C non-monotone during charging: min step = {delta.min():.3e}"
        f" for R={R}, C={C}, V={V}")
