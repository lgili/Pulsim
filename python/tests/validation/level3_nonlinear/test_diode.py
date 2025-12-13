"""Validation tests for diode circuits.

Tests ideal diode behavior in forward and reverse bias.
Tolerance: 5% maximum relative error.
"""

import pytest
import pulsim as ps


class TestIdealDiodeForward:
    """Test ideal diode in forward bias."""

    def test_diode_forward_voltage_drop(self):
        """Test diode forward conduction: most voltage drops across resistor."""
        V_SOURCE = 5.0
        R = 1000.0

        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("v_source")
        n2 = ckt.add_node("v_after_diode")

        ckt.add_voltage_source("V1", n1, gnd, V_SOURCE)
        ckt.add_diode("D1", n1, n2)  # Forward: anode at n1 (higher V)
        ckt.add_resistor("R1", n2, gnd, R)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success, f"DC failed: {dc_result.message}"

        v_diode_cathode = dc_result.newton_result.solution[1]

        # For ideal diode with g_on >> g_off:
        # When conducting, V_anode ≈ V_cathode (small drop)
        # Most of V_source appears across R
        # I ≈ V_source / R (small diode drop)

        print(f"\nV_source = {V_SOURCE}V")
        print(f"V_after_diode = {v_diode_cathode:.4f}V")
        print(f"Diode drop = {V_SOURCE - v_diode_cathode:.4f}V")

        # The ideal diode should have minimal forward drop
        # Expecting V_after_diode ≈ V_source (within a few %)
        assert v_diode_cathode > 0.9 * V_SOURCE, \
            f"Diode forward voltage too low: {v_diode_cathode}V"

    def test_diode_forward_current(self):
        """Test forward current through diode circuit."""
        V_SOURCE = 10.0
        R = 1000.0

        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("v_source")
        n2 = ckt.add_node("v_cathode")

        ckt.add_voltage_source("V1", n1, gnd, V_SOURCE)
        ckt.add_diode("D1", n1, n2)
        ckt.add_resistor("R1", n2, gnd, R)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success

        v_cathode = dc_result.newton_result.solution[1]

        # Current through resistor = V_cathode / R
        i_load = v_cathode / R
        expected_i = V_SOURCE / R  # Ideal case

        print(f"\nI_load = {i_load*1000:.4f} mA")
        print(f"Expected (ideal) = {expected_i*1000:.4f} mA")

        # Should be within 5% of ideal
        rel_error = abs(i_load - expected_i) / expected_i
        assert rel_error < 0.05, f"Current error too large: {rel_error*100:.2f}%"


class TestIdealDiodeReverse:
    """Test ideal diode in reverse bias."""

    def test_diode_reverse_blocks(self):
        """Test diode blocks current in reverse bias."""
        V_SOURCE = 5.0
        R = 1000.0

        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("v_source")
        n2 = ckt.add_node("v_before_diode")

        ckt.add_voltage_source("V1", n1, gnd, V_SOURCE)
        ckt.add_resistor("R1", n1, n2, R)
        # Diode in reverse: cathode at n2 (higher V), anode at gnd
        ckt.add_diode("D1", gnd, n2)  # Reverse biased

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success, f"DC failed: {dc_result.message}"

        v_before_diode = dc_result.newton_result.solution[1]

        # When reverse biased, diode blocks, so V_before_diode ≈ V_source
        # (very little current through R, so almost no voltage drop)
        print(f"\nV_source = {V_SOURCE}V")
        print(f"V_before_diode = {v_before_diode:.4f}V")
        print(f"Voltage drop across R = {V_SOURCE - v_before_diode:.6f}V")

        # Should be very close to V_source
        rel_error = abs(v_before_diode - V_SOURCE) / V_SOURCE
        assert rel_error < 0.01, f"Reverse diode not blocking: {rel_error*100:.4f}%"

    def test_diode_leakage_current(self):
        """Test diode leakage current in reverse bias is negligible."""
        V_SOURCE = 10.0
        R = 1000.0

        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("v_source")
        n2 = ckt.add_node("v_anode")

        ckt.add_voltage_source("V1", n1, gnd, V_SOURCE)
        ckt.add_resistor("R1", n1, n2, R)
        ckt.add_diode("D1", gnd, n2)  # Reverse: cathode at n2

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success

        v_anode = dc_result.newton_result.solution[1]

        # Leakage current = (V_source - V_anode) / R
        i_leakage = (V_SOURCE - v_anode) / R

        print(f"\nLeakage current = {i_leakage*1e9:.4f} nA")

        # Leakage should be very small (g_off = 1e-9 S by default)
        assert i_leakage < 1e-6, f"Leakage too high: {i_leakage:.2e} A"


class TestDiodeRectifier:
    """Test diode rectifier behavior."""

    def test_half_wave_rectifier(self):
        """Test half-wave rectifier behavior with AC-like input."""
        R = 1000.0

        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("v_in")
        n2 = ckt.add_node("v_out")

        # Start with positive voltage
        V_POS = 5.0
        ckt.add_voltage_source("V1", n1, gnd, V_POS)
        ckt.add_diode("D1", n1, n2)
        ckt.add_resistor("R1", n2, gnd, R)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success

        v_out_pos = dc_result.newton_result.solution[1]

        print(f"\nWith V_in = +{V_POS}V: V_out = {v_out_pos:.4f}V")

        # With positive input, diode conducts, V_out ≈ V_in
        assert v_out_pos > 0.9 * V_POS, "Diode not conducting with positive input"

        # Now test with negative voltage (new circuit)
        ckt2 = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt2.add_node("v_in")
        n2 = ckt2.add_node("v_out")

        V_NEG = -5.0
        ckt2.add_voltage_source("V1", n1, gnd, V_NEG)
        ckt2.add_diode("D1", n1, n2)
        ckt2.add_resistor("R1", n2, gnd, R)

        dc_result2 = ps.dc_operating_point(ckt2)
        assert dc_result2.success

        v_out_neg = dc_result2.newton_result.solution[1]

        print(f"With V_in = {V_NEG}V: V_out = {v_out_neg:.6f}V")

        # With negative input, diode blocks, V_out ≈ 0
        assert abs(v_out_neg) < 0.01, f"Diode not blocking with negative input: {v_out_neg}V"


class TestDiodeParameters:
    """Test diode with custom parameters."""

    def test_custom_conductances(self):
        """Test diode with custom g_on and g_off."""
        V_SOURCE = 5.0
        R = 1000.0
        G_ON = 1e4   # 0.1 mOhm on-resistance
        G_OFF = 1e-12  # Very high off-resistance

        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("v_source")
        n2 = ckt.add_node("v_out")

        ckt.add_voltage_source("V1", n1, gnd, V_SOURCE)
        ckt.add_diode("D1", n1, n2, g_on=G_ON, g_off=G_OFF)
        ckt.add_resistor("R1", n2, gnd, R)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success

        v_out = dc_result.newton_result.solution[1]

        # With very high g_on, forward drop should be minimal
        diode_drop = V_SOURCE - v_out
        V_SOURCE / (1 + G_ON * R)  # Approximate

        print(f"\nDiode drop with g_on={G_ON}: {diode_drop:.6f}V")

        # Should be very small
        assert diode_drop < 0.01, f"Diode drop too high: {diode_drop}V"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
