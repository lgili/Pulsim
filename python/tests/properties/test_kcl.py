"""Phase 2 of `add-property-based-testing`: KCL holds at the DC
operating point and after a settled transient.

KCL says the sum of currents entering any node equals zero. For a
voltage divider, that means I_through_R1 = I_through_R2 in steady
state. We assert this on randomly-generated networks via Hypothesis.
"""

from __future__ import annotations

import math

import pytest

pulsim = pytest.importorskip("pulsim")
hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, HealthCheck

from .strategies import gen_resistor_divider, make_quick_options


@given(gen_resistor_divider())
@settings(max_examples=25, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_kcl_holds_at_dc_op_for_resistor_divider(generated):
    """KCL at the midpoint: `(V_in - V_mid)/R1 == V_mid / R2`."""
    R1 = generated.parameters["R1"]
    R2 = generated.parameters["R2"]
    V = generated.parameters["V"]
    expected_v_mid = V * R2 / (R1 + R2)

    sim = pulsim.Simulator(generated.circuit, make_quick_options())
    dc = sim.dc_operating_point()
    assert dc.success, f"DC OP failed for {generated.parameters}"

    mid_idx = generated.circuit.get_node("mid")
    v_mid = dc.newton_result.solution[mid_idx]

    # Tolerate 1e-6 relative + 1e-9 absolute. Values can be tiny when
    # R1 ≫ R2.
    assert math.isclose(v_mid, expected_v_mid, rel_tol=1e-6,
                         abs_tol=1e-9), \
        (f"KCL violation: V_mid={v_mid} but analytical "
         f"V_mid={expected_v_mid} for R1={R1}, R2={R2}, V={V}")


@given(gen_resistor_divider())
@settings(max_examples=25, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_currents_sum_to_zero_at_node_after_transient(generated):
    """After a brief transient on a resistive divider, the current
    through R1 (top) equals the current through R2 (bottom): KCL at
    the midpoint."""
    sim = pulsim.Simulator(generated.circuit, make_quick_options())
    dc = sim.dc_operating_point()
    assert dc.success
    run = sim.run_transient(dc.newton_result.solution)
    assert run.success
    assert run.states, "transient produced no state samples"

    R1 = generated.parameters["R1"]
    R2 = generated.parameters["R2"]
    in_idx = generated.circuit.get_node("in")
    mid_idx = generated.circuit.get_node("mid")

    # Final-step state.
    final = run.states[-1]
    v_in = final[in_idx]
    v_mid = final[mid_idx]
    i_through_R1 = (v_in - v_mid) / R1
    i_through_R2 = v_mid / R2

    # KCL at midpoint: i_in = i_out within numerical noise.
    assert math.isclose(i_through_R1, i_through_R2,
                         rel_tol=1e-6, abs_tol=1e-9), \
        (f"KCL violation at midpoint: i_R1={i_through_R1} "
         f"i_R2={i_through_R2} for {generated.parameters}")
