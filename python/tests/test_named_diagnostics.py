"""v2.0 Phase 1: kernel errors name the offending node / device.

Audit findings `kernel-has-no-name-context-for-errors` (the Graph now
carries branch names) and `singular-errors-dont-name-the-node` (the
error paths use them). Before this, an unsolvable circuit produced
only a mask bitstring — on a 200-switch MMC that is unactionable.
"""

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
    # perfectly normal circuit.
    b = pulsim.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_resistor("R1", "vin", "gnd", 10.0)
    cache = pulsim.PwlStateSpaceCache(b.graph, b.pool)
    cache.build(0.0)
    assert cache.num_built_segments() >= 1
