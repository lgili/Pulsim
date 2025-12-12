"""Validation tests for DC resistor networks.

Tests voltage dividers, series, and parallel resistors.
Tolerance: 0.01% maximum relative error (exact solutions).
"""

import pytest
import numpy as np
import pulsim as ps
from ..framework.base import ValidationLevel, CircuitDefinition, ValidationTest


# =============================================================================
# Voltage Divider Tests
# =============================================================================

V_SOURCE = 10.0  # V


class TestVoltageDivider:
    """Test voltage divider circuits."""

    def test_equal_resistors(self):
        """Test voltage divider with equal resistors: V_out = V_in / 2."""
        R1, R2 = 1000.0, 1000.0

        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("v_in")
        n2 = ckt.add_node("v_out")

        ckt.add_voltage_source("V1", n1, gnd, V_SOURCE)
        ckt.add_resistor("R1", n1, n2, R1)
        ckt.add_resistor("R2", n2, gnd, R2)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success

        v_out = dc_result.newton_result.solution[1]
        expected = V_SOURCE * R2 / (R1 + R2)  # = 5.0V

        rel_error = abs(v_out - expected) / expected
        print(f"\nV_out = {v_out:.6f}V, expected = {expected:.6f}V, error = {rel_error*100:.6f}%")

        assert rel_error < 0.0001, f"Voltage divider error too large: {rel_error*100:.6f}%"

    def test_unequal_resistors(self):
        """Test voltage divider with unequal resistors."""
        R1, R2 = 3000.0, 1000.0  # V_out = 10 * 1000/4000 = 2.5V

        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("v_in")
        n2 = ckt.add_node("v_out")

        ckt.add_voltage_source("V1", n1, gnd, V_SOURCE)
        ckt.add_resistor("R1", n1, n2, R1)
        ckt.add_resistor("R2", n2, gnd, R2)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success

        v_out = dc_result.newton_result.solution[1]
        expected = V_SOURCE * R2 / (R1 + R2)

        rel_error = abs(v_out - expected) / expected
        print(f"\nV_out = {v_out:.6f}V, expected = {expected:.6f}V, error = {rel_error*100:.6f}%")

        assert rel_error < 0.0001

    def test_large_ratio_divider(self):
        """Test voltage divider with large resistance ratio."""
        R1, R2 = 100000.0, 100.0  # V_out ≈ 0.01V

        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("v_in")
        n2 = ckt.add_node("v_out")

        ckt.add_voltage_source("V1", n1, gnd, V_SOURCE)
        ckt.add_resistor("R1", n1, n2, R1)
        ckt.add_resistor("R2", n2, gnd, R2)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success

        v_out = dc_result.newton_result.solution[1]
        expected = V_SOURCE * R2 / (R1 + R2)

        rel_error = abs(v_out - expected) / expected if expected != 0 else abs(v_out)
        print(f"\nV_out = {v_out:.8f}V, expected = {expected:.8f}V, error = {rel_error*100:.6f}%")

        assert rel_error < 0.0001


class TestSeriesResistors:
    """Test series resistor networks."""

    def test_two_series(self):
        """Test two resistors in series: I = V / (R1 + R2)."""
        R1, R2 = 1000.0, 2000.0

        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("v_source")
        n2 = ckt.add_node("v_mid")

        ckt.add_voltage_source("V1", n1, gnd, V_SOURCE)
        ckt.add_resistor("R1", n1, n2, R1)
        ckt.add_resistor("R2", n2, gnd, R2)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success

        v_mid = dc_result.newton_result.solution[1]
        # V_mid = V * R2 / (R1 + R2) = 10 * 2000/3000 = 6.667V
        expected = V_SOURCE * R2 / (R1 + R2)

        rel_error = abs(v_mid - expected) / expected
        print(f"\nV_mid = {v_mid:.6f}V, expected = {expected:.6f}V, error = {rel_error*100:.6f}%")

        assert rel_error < 0.0001

    def test_three_series(self):
        """Test three resistors in series."""
        R1, R2, R3 = 1000.0, 2000.0, 3000.0

        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("v_source")
        n2 = ckt.add_node("v_1")
        n3 = ckt.add_node("v_2")

        ckt.add_voltage_source("V1", n1, gnd, V_SOURCE)
        ckt.add_resistor("R1", n1, n2, R1)
        ckt.add_resistor("R2", n2, n3, R2)
        ckt.add_resistor("R3", n3, gnd, R3)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success

        v1 = dc_result.newton_result.solution[1]
        v2 = dc_result.newton_result.solution[2]

        R_total = R1 + R2 + R3
        expected_v1 = V_SOURCE * (R2 + R3) / R_total
        expected_v2 = V_SOURCE * R3 / R_total

        error_v1 = abs(v1 - expected_v1) / expected_v1
        error_v2 = abs(v2 - expected_v2) / expected_v2

        print(f"\nV1 = {v1:.6f}V (expected {expected_v1:.6f}V), error = {error_v1*100:.6f}%")
        print(f"V2 = {v2:.6f}V (expected {expected_v2:.6f}V), error = {error_v2*100:.6f}%")

        assert error_v1 < 0.0001
        assert error_v2 < 0.0001


class TestParallelResistors:
    """Test parallel resistor networks."""

    def test_two_parallel(self):
        """Test two resistors in parallel: R_eq = R1*R2/(R1+R2)."""
        R1, R2 = 1000.0, 1000.0  # R_eq = 500 Ohms

        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("v_source")
        n2 = ckt.add_node("v_parallel")

        # Series resistor to limit current
        R_series = 500.0
        ckt.add_voltage_source("V1", n1, gnd, V_SOURCE)
        ckt.add_resistor("Rs", n1, n2, R_series)
        ckt.add_resistor("R1", n2, gnd, R1)
        ckt.add_resistor("R2", n2, gnd, R2)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success

        v_parallel = dc_result.newton_result.solution[1]
        # R_eq = 500, V_parallel = V * R_eq / (R_series + R_eq) = 10 * 500/1000 = 5V
        R_eq = R1 * R2 / (R1 + R2)
        expected = V_SOURCE * R_eq / (R_series + R_eq)

        rel_error = abs(v_parallel - expected) / expected
        print(f"\nV_parallel = {v_parallel:.6f}V, expected = {expected:.6f}V, error = {rel_error*100:.6f}%")

        assert rel_error < 0.0001

    def test_three_parallel(self):
        """Test three resistors in parallel."""
        R1, R2, R3 = 1000.0, 2000.0, 3000.0
        # 1/R_eq = 1/1000 + 1/2000 + 1/3000 = 6/6000 + 3/6000 + 2/6000 = 11/6000
        # R_eq = 6000/11 = 545.45 Ohms

        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("v_source")
        n2 = ckt.add_node("v_parallel")

        R_series = 454.55  # For easy calculation: total = 1000 Ohms
        ckt.add_voltage_source("V1", n1, gnd, V_SOURCE)
        ckt.add_resistor("Rs", n1, n2, R_series)
        ckt.add_resistor("R1", n2, gnd, R1)
        ckt.add_resistor("R2", n2, gnd, R2)
        ckt.add_resistor("R3", n2, gnd, R3)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success

        v_parallel = dc_result.newton_result.solution[1]
        R_eq = 1.0 / (1/R1 + 1/R2 + 1/R3)
        expected = V_SOURCE * R_eq / (R_series + R_eq)

        rel_error = abs(v_parallel - expected) / expected
        print(f"\nR_eq = {R_eq:.2f} Ohms")
        print(f"V_parallel = {v_parallel:.6f}V, expected = {expected:.6f}V, error = {rel_error*100:.6f}%")

        assert rel_error < 0.0001


class TestCurrentSource:
    """Test current source circuits."""

    def test_current_through_resistor(self):
        """Test current source driving a resistor: V = I * R."""
        I_SOURCE = 1e-3  # 1 mA
        R = 10000.0  # 10k Ohm

        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("v_node")

        # Current flows FROM npos TO nneg in pulsim convention
        # So (n1, gnd, I) means current flows from n1 to gnd = current INTO n1
        ckt.add_current_source("I1", n1, gnd, I_SOURCE)
        ckt.add_resistor("R1", n1, gnd, R)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success

        v_node = dc_result.newton_result.solution[0]
        expected = I_SOURCE * R  # = 10V

        rel_error = abs(v_node - expected) / expected
        print(f"\nV_node = {v_node:.6f}V, expected = {expected:.6f}V, error = {rel_error*100:.6f}%")

        assert rel_error < 0.0001


class TestMultipleVoltageSource:
    """Test circuits with multiple voltage sources."""

    def test_superposition(self):
        """Test circuit with two voltage sources - verify superposition."""
        V1, V2 = 10.0, 5.0
        R1, R2, R3 = 1000.0, 2000.0, 1000.0

        # Circuit: V1 -- R1 -- node -- R2 -- V2
        #                       |
        #                      R3
        #                       |
        #                      GND
        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("v1_node")
        n2 = ckt.add_node("mid_node")
        n3 = ckt.add_node("v2_node")

        ckt.add_voltage_source("V1", n1, gnd, V1)
        ckt.add_resistor("R1", n1, n2, R1)
        ckt.add_resistor("R3", n2, gnd, R3)
        ckt.add_resistor("R2", n2, n3, R2)
        ckt.add_voltage_source("V2", n3, gnd, V2)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success

        v_mid = dc_result.newton_result.solution[1]

        # By superposition:
        # V_mid = V1 * (R2||R3)/(R1 + R2||R3) + V2 * (R1||R3)/(R2 + R1||R3)
        # Or solve directly using node analysis
        # At node 2: (V1-V_mid)/R1 + (V2-V_mid)/R2 = V_mid/R3
        # V_mid * (1/R1 + 1/R2 + 1/R3) = V1/R1 + V2/R2
        sum_conductance = 1/R1 + 1/R2 + 1/R3
        expected = (V1/R1 + V2/R2) / sum_conductance

        rel_error = abs(v_mid - expected) / expected
        print(f"\nV_mid = {v_mid:.6f}V, expected = {expected:.6f}V, error = {rel_error*100:.6f}%")

        assert rel_error < 0.0001


class TestWheatstone:
    """Test Wheatstone bridge circuit."""

    def test_balanced_bridge(self):
        """Test balanced Wheatstone bridge: V_out = 0."""
        # When R1/R2 = R3/R4, bridge is balanced
        R1, R2, R3, R4 = 1000.0, 2000.0, 1000.0, 2000.0

        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        v_in = ckt.add_node("v_in")
        v_a = ckt.add_node("v_a")
        v_b = ckt.add_node("v_b")

        ckt.add_voltage_source("V1", v_in, gnd, V_SOURCE)
        ckt.add_resistor("R1", v_in, v_a, R1)
        ckt.add_resistor("R2", v_a, gnd, R2)
        ckt.add_resistor("R3", v_in, v_b, R3)
        ckt.add_resistor("R4", v_b, gnd, R4)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success

        v_diff = dc_result.newton_result.solution[1] - dc_result.newton_result.solution[2]

        print(f"\nV_a = {dc_result.newton_result.solution[1]:.6f}V")
        print(f"V_b = {dc_result.newton_result.solution[2]:.6f}V")
        print(f"V_diff = {v_diff:.9f}V (expected 0V)")

        assert abs(v_diff) < 1e-9, f"Balanced bridge voltage not zero: {v_diff}V"

    def test_unbalanced_bridge(self):
        """Test unbalanced Wheatstone bridge."""
        R1, R2, R3, R4 = 1000.0, 2000.0, 1500.0, 2000.0  # Unbalanced

        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        v_in = ckt.add_node("v_in")
        v_a = ckt.add_node("v_a")
        v_b = ckt.add_node("v_b")

        ckt.add_voltage_source("V1", v_in, gnd, V_SOURCE)
        ckt.add_resistor("R1", v_in, v_a, R1)
        ckt.add_resistor("R2", v_a, gnd, R2)
        ckt.add_resistor("R3", v_in, v_b, R3)
        ckt.add_resistor("R4", v_b, gnd, R4)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success

        v_a = dc_result.newton_result.solution[1]
        v_b = dc_result.newton_result.solution[2]

        # V_a = V_in * R2/(R1+R2) = 10 * 2000/3000 = 6.667V
        # V_b = V_in * R4/(R3+R4) = 10 * 2000/3500 = 5.714V
        expected_a = V_SOURCE * R2 / (R1 + R2)
        expected_b = V_SOURCE * R4 / (R3 + R4)

        error_a = abs(v_a - expected_a) / expected_a
        error_b = abs(v_b - expected_b) / expected_b

        print(f"\nV_a = {v_a:.6f}V (expected {expected_a:.6f}V), error = {error_a*100:.6f}%")
        print(f"V_b = {v_b:.6f}V (expected {expected_b:.6f}V), error = {error_b*100:.6f}%")
        print(f"V_diff = {v_a - v_b:.6f}V")

        assert error_a < 0.0001
        assert error_b < 0.0001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
