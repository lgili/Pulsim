"""Regression coverage for ``add-python-builder-ergonomics`` (v1.5).

Covers:

* Initial conditions on capacitors / inductors via
  ``set_initial`` + auto-population in ``simulate``.
* GUI-style human aliases via ``set_alias`` + transparent
  resolution through ``result.v`` / ``.i``.
* The ``Cancelled`` exception type.

The full proposal also asks for kernel-deep ``should_continue``
cancellation on ``compute_dc_op`` / ``run_ac_sweep`` /
``compute_temperature`` — that requires C++ changes (tasks 3.1-3.4)
and is out of scope for the Python-side v1 of this PR. The
:class:`Cancelled` exception is shipped here so callers can
already wrap their own polling loops over those analyses.
"""
from __future__ import annotations

import numpy as np
import pytest

import pulsim as ps


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------
class TestInitialConditions:
    def test_inductor_i0_via_set_initial(self):
        # An ideal LR circuit with V=0 and i_L(0)=0.5 A should
        # decay i_L(t) = i_L(0)·exp(-R·t/L). Pick R=1 Ω, L=1 mH so
        # τ = 1 ms. At t=0 we expect 0.5 A.
        b = ps.CircuitBuilder()
        b.add_voltage_source("V1", "a", "gnd", 0.0)
        b.add_inductor("L1", "a", "b", 1e-3)
        b.add_resistor("R1", "b", "gnd", 1.0)
        b.set_initial("L1", 0.5)
        # IC reflects in the synthesised state vector. Layout
        # is [v_nodes..., i_sources..., i_inductors...].
        x0 = b.initial_state()
        assert any(abs(v - 0.5) < 1e-12 for v in x0), (
            f"expected 0.5 somewhere in initial_state, got {list(x0)}")

        res = ps.simulate(b, t_end=5e-3, dt=1e-5)
        # First sample of L1 current ≈ 0.5 A (within 1 mA).
        i0_actual = res.i("L1")[0]
        assert abs(i0_actual - 0.5) < 1e-3, (
            f"i_L(0) = {i0_actual:.6f} A, expected ≈ 0.5 A")

    def test_set_initial_unknown_device(self):
        b = ps.CircuitBuilder()
        b.add_resistor("R1", "a", "gnd", 1.0)
        # C++ binding raises IndexError (out_of_range translation).
        with pytest.raises((IndexError, KeyError), match="X1"):
            b.set_initial("X1", 0.5)

    def test_explicit_initial_state_wins(self):
        # Spec scenario: "Explicit initial_state wins over builder ICs"
        # — passing initial_state=zeros should override the recorded IC.
        b = ps.CircuitBuilder()
        b.add_voltage_source("V1", "a", "gnd", 0.0)
        b.add_inductor("L1", "a", "b", 1e-3)
        b.add_resistor("R1", "b", "gnd", 1.0)
        b.set_initial("L1", 0.5)
        x0 = np.zeros(b.initial_state().shape)
        res = ps.simulate(b, t_end=1e-3, dt=1e-5, initial_state=x0)
        assert abs(res.i("L1")[0]) < 1e-9, (
            "explicit initial_state=zeros should override the IC")

    def test_initial_state_zero_when_no_ic(self):
        b = ps.CircuitBuilder()
        b.add_resistor("R1", "a", "gnd", 1.0)
        b.add_capacitor("C1", "a", "gnd", 1e-6)
        x0 = b.initial_state()
        # All zeros (no ICs set).
        assert np.allclose(x0, 0.0)


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------
class TestAliases:
    def test_alias_resolves_through_result_v(self):
        # Spec scenario: "Alias resolves through node_id_of" — we
        # check via the result-level accessor.
        b = ps.CircuitBuilder()
        b.add_voltage_source("V1", "node_42", "gnd", 12.0)
        b.add_resistor("R1", "node_42", "gnd", 1.0)
        b.set_alias("vin", node="node_42")

        res = ps.simulate(b, t_end=1e-3, dt=1e-5)
        # Both names yield the same trace.
        np.testing.assert_array_equal(res.v("vin"), res.v("node_42"))

    def test_alias_dict_round_trip(self):
        b = ps.CircuitBuilder()
        b.add_voltage_source("V1", "node_42", "gnd", 12.0)
        b.add_resistor("R1", "node_42", "gnd", 1.0)
        b.set_alias("vin", node="node_42")
        b.set_alias("ground_clone", node="gnd")
        aliases = b.aliases()
        assert aliases == {
            "vin": ("node", "node_42"),
            "ground_clone": ("node", "gnd"),
        }

    def test_alias_both_node_and_branch_rejected(self):
        b = ps.CircuitBuilder()
        b.add_resistor("R1", "a", "gnd", 1.0)
        with pytest.raises(ValueError, match="exactly one"):
            b.set_alias("x", node="a", branch="R1")

    def test_alias_neither_rejected(self):
        b = ps.CircuitBuilder()
        b.add_resistor("R1", "a", "gnd", 1.0)
        with pytest.raises(ValueError, match="exactly one"):
            b.set_alias("x")

    def test_alias_collision_with_canonical_node_rejected(self):
        b = ps.CircuitBuilder()
        b.add_voltage_source("V1", "vout", "gnd", 12.0)
        b.add_resistor("R1", "vout", "gnd", 1.0)
        with pytest.raises(ValueError, match="canonical"):
            b.set_alias("vout", node="gnd")

    def test_alias_empty_string_rejected(self):
        b = ps.CircuitBuilder()
        b.add_resistor("R1", "a", "gnd", 1.0)
        with pytest.raises(ValueError):
            b.set_alias("", node="a")
        with pytest.raises(ValueError):
            b.set_alias("x", node="")


# ---------------------------------------------------------------------------
# Cancelled exception
# ---------------------------------------------------------------------------
class TestCancelled:
    def test_cancelled_attributes(self):
        exc = ps.Cancelled("compute_dc_op", point_index=3)
        assert exc.where == "compute_dc_op"
        assert exc.point_index == 3
        assert "compute_dc_op" in str(exc)
        assert "3" in str(exc)

    def test_cancelled_is_runtime_error(self):
        # Existing `except RuntimeError:` catches keep working.
        try:
            raise ps.Cancelled("test")
        except RuntimeError:
            pass  # expected
        else:
            pytest.fail("Cancelled must be a RuntimeError subclass")
