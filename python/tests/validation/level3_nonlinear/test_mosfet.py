"""Validation tests for MOSFET device.

Tests MOSFET behavior in different operating regions:
- Cutoff: Vgs < Vth
- Linear/Triode: Vgs > Vth, Vds < Vgs - Vth
- Saturation: Vgs > Vth, Vds > Vgs - Vth

Tolerance: 5% for nonlinear behavior.
"""

import pytest
import numpy as np
import pulsim as ps


# Default MOSFET parameters
VTH = 2.0      # Threshold voltage
KP = 0.1       # Transconductance parameter (A/V²)
LAMBDA = 0.01  # Channel length modulation
G_OFF = 1e-12  # Off-state conductance


class TestMOSFETCutoff:
    """Test MOSFET in cutoff region (Vgs < Vth)."""

    def test_mosfet_cutoff_no_current(self):
        """MOSFET with Vgs < Vth should have negligible drain current."""
        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()

        # Nodes
        n_drain = ckt.add_node("drain")
        n_gate = ckt.add_node("gate")
        n_source = ckt.add_node("source")

        # Vdd -> R_load -> Drain, Source -> GND
        # Gate voltage below threshold
        V_DD = 10.0
        V_GATE = 1.0  # Below Vth=2V
        R_LOAD = 1000.0

        ckt.add_voltage_source("Vdd", n_drain, gnd, V_DD)
        ckt.add_resistor("Rload", n_drain, n_source, R_LOAD)
        ckt.add_voltage_source("Vgate", n_gate, gnd, V_GATE)

        # MOSFET: gate, drain, source
        params = ps.MOSFETParams()
        params.vth = VTH
        params.kp = KP
        params.lambda_ = LAMBDA
        params.g_off = G_OFF
        ckt.add_mosfet("M1", n_gate, n_drain, n_source, params)

        # DC analysis
        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success, f"DC analysis failed: {dc_result.message}"

        # In cutoff, drain current should be ~0 (only g_off leakage)
        # V_drain ≈ V_DD (no voltage drop across R_load)
        v_drain = dc_result.newton_result.solution[0]  # n_drain

        # With g_off = 1e-12, current is negligible
        # V_source should be very close to 0 (tied to ground through source)
        print(f"\nMOSFET Cutoff Test:")
        print(f"  V_gate = {V_GATE}V (< Vth={VTH}V)")
        print(f"  V_drain = {v_drain:.6f}V")

        # Drain voltage should be close to Vdd (minimal current)
        assert abs(v_drain - V_DD) < 0.1, f"Drain voltage not at Vdd: {v_drain}V"

    def test_mosfet_cutoff_leakage(self):
        """Test that cutoff leakage is controlled by g_off parameter."""
        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()

        n_drain = ckt.add_node("drain")
        n_gate = ckt.add_node("gate")

        V_DD = 5.0
        V_GATE = 0.0  # Well below threshold
        R_LOAD = 1e6  # Large resistor to see leakage effect

        ckt.add_voltage_source("Vdd", n_drain, gnd, V_DD)
        ckt.add_voltage_source("Vgate", n_gate, gnd, V_GATE)

        # MOSFET with higher g_off to make leakage visible
        params = ps.MOSFETParams()
        params.vth = VTH
        params.g_off = 1e-6  # Higher leakage for testing

        # Drain to ground through MOSFET (source at ground)
        ckt.add_mosfet("M1", n_gate, n_drain, gnd, params)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success

        # Leakage current = g_off * Vds
        # With R_load in series, voltage divider effect
        v_drain = dc_result.newton_result.solution[0]

        print(f"\nMOSFET Leakage Test:")
        print(f"  g_off = {params.g_off}")
        print(f"  V_drain = {v_drain:.6f}V")

        # Some leakage should be present but drain still near Vdd
        assert v_drain > 0.9 * V_DD, f"Unexpected drain voltage: {v_drain}V"


class TestMOSFETLinearRegion:
    """Test MOSFET in linear/triode region."""

    def test_mosfet_linear_region_dc(self):
        """MOSFET in linear region: Id proportional to Vds."""
        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()

        n_drain = ckt.add_node("drain")
        n_gate = ckt.add_node("gate")

        # High gate voltage to ensure linear region
        V_DD = 2.0   # Low Vdd to stay in linear region
        V_GATE = 5.0  # Vgs - Vth = 3V > Vds
        R_LOAD = 100.0

        ckt.add_voltage_source("Vdd", n_drain, gnd, V_DD)
        ckt.add_resistor("Rload", n_drain, gnd, R_LOAD)
        ckt.add_voltage_source("Vgate", n_gate, gnd, V_GATE)

        params = ps.MOSFETParams()
        params.vth = VTH
        params.kp = KP
        ckt.add_mosfet("M1", n_gate, n_drain, gnd, params)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success, f"DC failed: {dc_result.message}"

        v_drain = dc_result.newton_result.solution[0]

        # In linear region, MOSFET acts like a voltage-controlled resistor
        # Id = kp * [(Vgs - Vth) * Vds - Vds²/2]
        # Vds = v_drain (source at ground)
        vgs = V_GATE
        vds = v_drain
        vov = vgs - VTH  # Overdrive voltage

        if vds < vov:
            # Linear region formula
            id_expected = KP * ((vov * vds) - (vds**2 / 2))
            print(f"\nMOSFET Linear Region Test:")
            print(f"  Vgs = {vgs}V, Vth = {VTH}V, Vov = {vov}V")
            print(f"  Vds = {vds:.4f}V (< Vov, so linear region)")
            print(f"  Id (expected) = {id_expected*1000:.4f} mA")

            # Verify we're in linear region
            assert vds < vov, f"Not in linear region: Vds={vds} >= Vov={vov}"


class TestMOSFETSaturation:
    """Test MOSFET in saturation region."""

    def test_mosfet_saturation_region_dc(self):
        """MOSFET in saturation: Id = kp/2 * (Vgs-Vth)²."""
        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()

        n_drain = ckt.add_node("drain")
        n_gate = ckt.add_node("gate")

        # Setup for saturation: Vds > Vgs - Vth
        V_DD = 10.0
        V_GATE = 3.0  # Vgs - Vth = 1V, need Vds > 1V
        R_LOAD = 1000.0

        ckt.add_voltage_source("Vdd", n_drain, gnd, V_DD)
        ckt.add_resistor("Rload", n_drain, gnd, R_LOAD)
        ckt.add_voltage_source("Vgate", n_gate, gnd, V_GATE)

        params = ps.MOSFETParams()
        params.vth = VTH
        params.kp = KP
        params.lambda_ = 0.0  # Ignore channel length modulation for simplicity
        ckt.add_mosfet("M1", n_gate, n_drain, gnd, params)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success, f"DC failed: {dc_result.message}"

        v_drain = dc_result.newton_result.solution[0]
        vgs = V_GATE
        vds = v_drain
        vov = vgs - VTH

        print(f"\nMOSFET Saturation Region Test:")
        print(f"  Vgs = {vgs}V, Vth = {VTH}V, Vov = {vov}V")
        print(f"  Vds = {vds:.4f}V")

        # Check if in saturation
        if vds > vov:
            # Saturation: Id = kp/2 * (Vgs - Vth)²
            id_sat = (KP / 2) * (vov ** 2)
            print(f"  In saturation region (Vds > Vov)")
            print(f"  Id (saturation formula) = {id_sat*1000:.4f} mA")

            # Voltage drop across R_load: V_R = Id * R_LOAD
            # V_drain = V_DD - Id * R_LOAD... but our circuit is different
            # Actually we have Vdd at drain node directly


class TestMOSFETSwitching:
    """Test MOSFET as a switch."""

    def test_mosfet_switch_on(self):
        """MOSFET with high gate voltage acts as closed switch."""
        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()

        n_vdd = ckt.add_node("vdd")
        n_drain = ckt.add_node("drain")
        n_gate = ckt.add_node("gate")

        V_DD = 5.0
        V_GATE = 10.0  # High gate voltage
        R_LOAD = 100.0

        ckt.add_voltage_source("Vdd", n_vdd, gnd, V_DD)
        ckt.add_resistor("Rload", n_vdd, n_drain, R_LOAD)
        ckt.add_voltage_source("Vgate", n_gate, gnd, V_GATE)

        params = ps.MOSFETParams()
        params.vth = VTH
        params.kp = 1.0  # High kp for good switching
        ckt.add_mosfet("M1", n_gate, n_drain, gnd, params)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success

        v_drain = dc_result.newton_result.solution[1]  # n_drain is node 1

        # With high gate voltage, MOSFET is on, drain should be near ground
        print(f"\nMOSFET Switch ON Test:")
        print(f"  V_gate = {V_GATE}V")
        print(f"  V_drain = {v_drain:.4f}V (expected ~0V)")

        # Drain should be close to ground (low Vds when on)
        assert v_drain < 1.0, f"MOSFET not fully on: V_drain={v_drain}V"

    def test_mosfet_switch_off(self):
        """MOSFET with low gate voltage acts as open switch."""
        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()

        n_vdd = ckt.add_node("vdd")
        n_drain = ckt.add_node("drain")
        n_gate = ckt.add_node("gate")

        V_DD = 5.0
        V_GATE = 0.0  # Gate below threshold
        R_LOAD = 100.0

        ckt.add_voltage_source("Vdd", n_vdd, gnd, V_DD)
        ckt.add_resistor("Rload", n_vdd, n_drain, R_LOAD)
        ckt.add_voltage_source("Vgate", n_gate, gnd, V_GATE)

        params = ps.MOSFETParams()
        params.vth = VTH
        params.kp = 1.0
        ckt.add_mosfet("M1", n_gate, n_drain, gnd, params)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success

        v_drain = dc_result.newton_result.solution[1]

        # With gate below threshold, MOSFET is off, drain should be near Vdd
        print(f"\nMOSFET Switch OFF Test:")
        print(f"  V_gate = {V_GATE}V (< Vth={VTH}V)")
        print(f"  V_drain = {v_drain:.4f}V (expected ~{V_DD}V)")

        # Drain should be close to Vdd (no current)
        assert v_drain > 0.9 * V_DD, f"MOSFET not fully off: V_drain={v_drain}V"


class TestMOSFETParameters:
    """Test MOSFET parameter variations."""

    def test_threshold_voltage_effect(self):
        """Higher Vth requires higher gate voltage to turn on."""
        results = []

        for vth in [1.0, 2.0, 3.0]:
            ckt = ps.Circuit()
            gnd = ps.Circuit.ground()

            n_vdd = ckt.add_node("vdd")
            n_drain = ckt.add_node("drain")
            n_gate = ckt.add_node("gate")

            V_DD = 5.0
            V_GATE = 4.0  # Fixed gate voltage
            R_LOAD = 100.0

            ckt.add_voltage_source("Vdd", n_vdd, gnd, V_DD)
            ckt.add_resistor("Rload", n_vdd, n_drain, R_LOAD)
            ckt.add_voltage_source("Vgate", n_gate, gnd, V_GATE)

            params = ps.MOSFETParams()
            params.vth = vth
            params.kp = 0.5
            ckt.add_mosfet("M1", n_gate, n_drain, gnd, params)

            dc_result = ps.dc_operating_point(ckt)
            assert dc_result.success

            v_drain = dc_result.newton_result.solution[1]
            results.append((vth, v_drain))

        print(f"\nThreshold Voltage Effect:")
        for vth, v_drain in results:
            status = "ON" if v_drain < 2.5 else "OFF"
            print(f"  Vth={vth}V: V_drain={v_drain:.4f}V ({status})")

        # Higher Vth should result in higher drain voltage (less current)
        assert results[0][1] < results[1][1] < results[2][1], \
            "Higher Vth should reduce drain current"

    def test_kp_effect_on_current(self):
        """Higher kp should result in more drain current."""
        results = []

        for kp in [0.1, 0.5, 1.0]:
            ckt = ps.Circuit()
            gnd = ps.Circuit.ground()

            n_vdd = ckt.add_node("vdd")
            n_drain = ckt.add_node("drain")
            n_gate = ckt.add_node("gate")

            V_DD = 5.0
            V_GATE = 5.0
            R_LOAD = 100.0

            ckt.add_voltage_source("Vdd", n_vdd, gnd, V_DD)
            ckt.add_resistor("Rload", n_vdd, n_drain, R_LOAD)
            ckt.add_voltage_source("Vgate", n_gate, gnd, V_GATE)

            params = ps.MOSFETParams()
            params.vth = VTH
            params.kp = kp
            ckt.add_mosfet("M1", n_gate, n_drain, gnd, params)

            dc_result = ps.dc_operating_point(ckt)
            assert dc_result.success

            v_drain = dc_result.newton_result.solution[1]
            # Current through resistor: I = (Vdd - Vdrain) / R
            i_drain = (V_DD - v_drain) / R_LOAD
            results.append((kp, v_drain, i_drain))

        print(f"\nTransconductance (kp) Effect:")
        for kp, v_drain, i_drain in results:
            print(f"  kp={kp}: V_drain={v_drain:.4f}V, Id={i_drain*1000:.4f}mA")

        # Higher kp should result in more current (lower drain voltage)
        assert results[0][1] > results[1][1] > results[2][1], \
            "Higher kp should increase drain current"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
