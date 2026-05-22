"""Validation tests for ideal switch circuits.

Tests switch behavior in open and closed states.
Tolerance: 1% maximum relative error.
"""

import pytest
import pulsim as ps


class TestIdealSwitchClosed:
    """Test ideal switch in closed state."""

    def test_switch_closed_conducts(self):
        """Test closed switch conducts current."""
        V_SOURCE = 10.0
        R = 1000.0

        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("v_source")
        n2 = ckt.add_node("v_load")

        ckt.add_voltage_source("V1", n1, gnd, V_SOURCE)
        ckt.add_switch("S1", n1, n2, closed=True)
        ckt.add_resistor("R1", n2, gnd, R)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success, f"DC failed: {dc_result.message}"

        v_load = dc_result.newton_result.solution[1]

        print(f"\nV_source = {V_SOURCE}V")
        print(f"V_load (switch closed) = {v_load:.6f}V")
        print(f"Switch drop = {V_SOURCE - v_load:.6f}V")

        # Closed switch: V_load ≈ V_source
        rel_error = abs(v_load - V_SOURCE) / V_SOURCE
        assert rel_error < 0.01, f"Closed switch drop too large: {rel_error*100:.4f}%"

    def test_switch_closed_current(self):
        """Test current through closed switch."""
        V_SOURCE = 5.0
        R = 500.0

        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("v_source")
        n2 = ckt.add_node("v_load")

        ckt.add_voltage_source("V1", n1, gnd, V_SOURCE)
        ckt.add_switch("S1", n1, n2, closed=True)
        ckt.add_resistor("R1", n2, gnd, R)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success

        v_load = dc_result.newton_result.solution[1]
        i_load = v_load / R
        expected_i = V_SOURCE / R

        print(f"\nI_load = {i_load*1000:.4f} mA")
        print(f"Expected = {expected_i*1000:.4f} mA")

        rel_error = abs(i_load - expected_i) / expected_i
        assert rel_error < 0.01, f"Current error: {rel_error*100:.4f}%"


class TestIdealSwitchOpen:
    """Test ideal switch in open state."""

    def test_switch_open_blocks(self):
        """Test open switch blocks current."""
        V_SOURCE = 10.0
        R = 1000.0

        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("v_source")
        n2 = ckt.add_node("v_load")

        ckt.add_voltage_source("V1", n1, gnd, V_SOURCE)
        ckt.add_switch("S1", n1, n2, closed=False)
        ckt.add_resistor("R1", n2, gnd, R)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success, f"DC failed: {dc_result.message}"

        v_load = dc_result.newton_result.solution[1]

        print(f"\nV_source = {V_SOURCE}V")
        print(f"V_load (switch open) = {v_load:.9f}V")

        # Open switch: V_load ≈ 0 (no current through R)
        assert abs(v_load) < 0.001, f"Open switch leakage too high: {v_load}V"

    def test_switch_open_leakage(self):
        """Test leakage current through open switch."""
        V_SOURCE = 100.0  # Higher voltage to measure leakage
        R = 1000.0

        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("v_source")
        n2 = ckt.add_node("v_load")

        ckt.add_voltage_source("V1", n1, gnd, V_SOURCE)
        ckt.add_switch("S1", n1, n2, closed=False)
        ckt.add_resistor("R1", n2, gnd, R)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success

        v_load = dc_result.newton_result.solution[1]
        i_leakage = v_load / R

        print(f"\nLeakage current = {i_leakage*1e9:.4f} nA")

        # Leakage should be negligible (g_off = 1e-12 S by default)
        assert abs(i_leakage) < 1e-6, f"Leakage too high: {i_leakage:.2e} A"


class TestSwitchTransient:
    """Test switch behavior in transient simulation."""

    def test_rc_with_switch(self):
        """Test RC circuit with switch controlling charge."""
        V_SOURCE = 5.0
        R = 1000.0
        C = 1e-6  # 1 µF
        TAU = R * C

        # Circuit with switch closed - capacitor charges
        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("v_source")
        n2 = ckt.add_node("v_switch")
        n3 = ckt.add_node("v_cap")

        ckt.add_voltage_source("V1", n1, gnd, V_SOURCE)
        ckt.add_switch("S1", n1, n2, closed=True)
        ckt.add_resistor("R1", n2, n3, R)
        ckt.add_capacitor("C1", n3, gnd, C)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success

        times, states, success, _ = ps.run_transient(
            ckt, 0.0, 5*TAU, TAU/100, dc_result.newton_result.solution
        )

        v_cap_final = states[-1][2]

        print(f"\nV_cap final (switch closed) = {v_cap_final:.4f}V")
        print(f"Expected = {V_SOURCE}V")

        # With switch closed, capacitor should charge to V_source
        rel_error = abs(v_cap_final - V_SOURCE) / V_SOURCE
        assert rel_error < 0.01, f"Final voltage error: {rel_error*100:.4f}%"


class TestSwitchParameters:
    """Test switch with custom parameters."""

    def test_custom_conductances(self):
        """Test switch with custom g_on and g_off."""
        V_SOURCE = 10.0
        R = 1000.0
        G_ON = 1e8   # Very low on-resistance
        G_OFF = 1e-15  # Very high off-resistance

        # Test closed
        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt.add_node("v_source")
        n2 = ckt.add_node("v_load")

        ckt.add_voltage_source("V1", n1, gnd, V_SOURCE)
        ckt.add_switch("S1", n1, n2, closed=True, g_on=G_ON, g_off=G_OFF)
        ckt.add_resistor("R1", n2, gnd, R)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success

        v_load = dc_result.newton_result.solution[1]
        switch_drop = V_SOURCE - v_load

        print(f"\nWith g_on={G_ON:.0e}:")
        print(f"  V_load = {v_load:.9f}V")
        print(f"  Switch drop = {switch_drop:.9f}V")

        # Very high g_on means very small drop
        assert switch_drop < 1e-4, f"Switch drop too high: {switch_drop}V"


class TestSwitchStateControl:
    """Test switch state control methods."""

    def test_switch_toggle(self):
        """Test switch open/close methods."""
        V_SOURCE = 5.0
        R = 1000.0

        # Test with initially closed switch
        ckt1 = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt1.add_node("v_source")
        n2 = ckt1.add_node("v_load")

        ckt1.add_voltage_source("V1", n1, gnd, V_SOURCE)
        ckt1.add_switch("S1", n1, n2, closed=True)
        ckt1.add_resistor("R1", n2, gnd, R)

        dc_closed = ps.dc_operating_point(ckt1)
        assert dc_closed.success
        v_closed = dc_closed.newton_result.solution[1]

        # Test with initially open switch
        ckt2 = ps.Circuit()
        gnd = ps.Circuit.ground()
        n1 = ckt2.add_node("v_source")
        n2 = ckt2.add_node("v_load")

        ckt2.add_voltage_source("V1", n1, gnd, V_SOURCE)
        ckt2.add_switch("S1", n1, n2, closed=False)
        ckt2.add_resistor("R1", n2, gnd, R)

        dc_open = ps.dc_operating_point(ckt2)
        assert dc_open.success
        v_open = dc_open.newton_result.solution[1]

        print(f"\nV_load (closed) = {v_closed:.6f}V")
        print(f"V_load (open) = {v_open:.9f}V")

        # Closed should be near V_SOURCE, open should be near 0
        assert v_closed > 0.99 * V_SOURCE
        assert abs(v_open) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
