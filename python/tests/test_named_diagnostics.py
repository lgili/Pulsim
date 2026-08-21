"""v2.0 Phase 1: kernel errors name the offending node / device.

Audit findings `kernel-has-no-name-context-for-errors` (the Graph now
carries branch names) and `singular-errors-dont-name-the-node` (the
error paths use them). Before this, an unsolvable circuit produced
only a mask bitstring — on a 200-switch MMC that is unactionable.
"""

import numpy as np
import pytest

import pulsim


def _floating_node_builder():
    # `vfloat` is tied to the circuit ONLY through a capacitor, which
    # is an open circuit at DC → its MNA column is empty.
    b = pulsim.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_resistor("R1", "vin", "gnd", 10.0)
    b.add_capacitor("Cfloat", "vin", "vfloat", 1e-6)
    return b


def test_singular_error_names_the_floating_node():
    b = _floating_node_builder()
    cache = pulsim.PwlStateSpaceCache(b.graph, b.pool)
    with pytest.raises(RuntimeError) as exc:
        cache.build(0.0)          # static build: capacitors are open
    msg = str(exc.value)
    # The name must survive the whole chain: builder → Graph →
    # row resolver → CacheError::what() → pybind → Python.
    assert "vfloat" in msg
    assert "no device ties it" in msg
    assert "DC path" in msg
    # ...and the old information is still there.
    assert "singular" in msg


def test_message_is_actionable_not_just_located():
    # A good diagnostic says what to DO. Regression guard against
    # someone trimming the remedy out of the message later.
    b = _floating_node_builder()
    cache = pulsim.PwlStateSpaceCache(b.graph, b.pool)
    with pytest.raises(RuntimeError) as exc:
        cache.build(0.0)
    msg = str(exc.value)
    assert "resistance" in msg or "tie it to ground" in msg


def test_healthy_circuit_still_builds():
    # The structural probe must not produce false positives on a
    # circuit that DOES have the device kinds whose stamping decides
    # whether a column ends up empty.
    b = pulsim.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_switch("S1", "vin", "sw", 1e-3, 1e9)
    b.add_resistor("R1", "sw", "gnd", 10.0)
    b.add_capacitor("C1", "sw", "gnd", 1e-6)
    b.add_capacitor("C2", "sw", "vin", 1e-7)   # floating cap
    b.add_inductor("L1", "sw", "gnd", 1e-3)
    cache = pulsim.PwlStateSpaceCache(b.graph, b.pool)
    cache.build(1e-6)          # dt > 0: caps and inductors stamped
    assert cache.num_built_segments() == 2     # 2^1 switch states
    res = pulsim.simulate(b, t_end=1e-4, dt=1e-6)
    assert res.num_steps() > 0
    assert all(abs(v) < 1e6 for v in res.states[-1])


def test_device_row_empty_by_construction_is_not_blamed_as_unwired():
    # This is the assertion that actually depends on the diagnostics
    # change (review finding F7: the previous "healthy circuit" test
    # would have passed with the whole feature reverted).
    #
    # The SAME healthy circuit, built at dt = 0, leaves the
    # inductor's row empty because the static assembler omits
    # companion stamps — not because L1 is disconnected. The message
    # must say so instead of telling the user to rewire a correct
    # part.
    b = pulsim.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_resistor("R1", "vin", "gnd", 10.0)
    b.add_inductor("L1", "vin", "gnd", 1e-3)
    cache = pulsim.PwlStateSpaceCache(b.graph, b.pool)
    with pytest.raises(RuntimeError) as exc:
        cache.build(0.0)
    msg = str(exc.value)
    assert "L1" in msg                       # names the device
    assert "no equation in this system" in msg
    assert "dt > 0" in msg                   # and the real fix
    assert "no device ties it" not in msg    # NOT a wiring accusation
    assert "bleeder" not in msg


def test_uncoupled_transformer_is_not_rejected_by_the_lti_guard():
    # Regression guard: the coupled-inductor refusal must key on
    # ACTUAL mutual inductance, not on a registered coupling.
    # `add_transformer(..., k=0)` is a documented way to model two
    # independent windings — there M really is 0, the mass matrix
    # really is diagonal, and the extraction is exact.
    b = pulsim.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_resistor("Rp", "vin", "p1", 0.1)
    b.add_transformer("T1", "p1", "gnd", "s1", "gnd", 1e-3, 4e-3, 0.0)
    b.add_resistor("Rs", "s1", "gnd", 10.0)
    cache = pulsim.PwlStateSpaceCache(b.graph, b.pool)
    cache.build_lazy(1e-6)
    A, b_const, _rows, _is_cap, _proj = cache.compute_lti_state_space(
        pulsim.SwitchStateMask(0))
    assert A.shape[0] == A.shape[1] == 2        # two inductor states
    assert np.isfinite(A).all()
    # Shape and finiteness are not the claim. The claim is that with
    # M = 0 the mass matrix really IS diagonal, so the extraction is
    # EXACT — assert the analytic answer, which is what would catch
    # a future predicate that got widened too far.
    #   di_Lp/dt = -(Rp/L_p)·i_Lp + Vin/L_p ,  di_Ls/dt = -(Rs/L_s)·i_Ls
    np.testing.assert_allclose(A[0, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(A[1, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(A[0, 0], -0.1 / 1e-3, rtol=1e-9)
    np.testing.assert_allclose(A[1, 1], -10.0 / 4e-3, rtol=1e-9)
    assert np.isfinite(b_const).all()

    # ...while a genuinely coupled pair IS refused.
    b2 = pulsim.CircuitBuilder()
    b2.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b2.add_resistor("Rp", "vin", "p1", 0.1)
    b2.add_transformer("T1", "p1", "gnd", "s1", "gnd", 1e-3, 4e-3, 0.98)
    b2.add_resistor("Rs", "s1", "gnd", 10.0)
    c2 = pulsim.PwlStateSpaceCache(b2.graph, b2.pool)
    c2.build_lazy(1e-6)
    with pytest.raises(RuntimeError, match="magnetically coupled"):
        c2.compute_lti_state_space(pulsim.SwitchStateMask(0))
