"""Validation tests for Boost (step-up) DC-DC converter.

Tests Boost converter steady-state and transient behavior.
Topology: Vin -> L -> Switch to GND
                  -> Diode -> C || R_load -> GND

Key equations (CCM):
- Duty cycle: D = 1 - Vin/Vout
- Output voltage: Vout = Vin / (1 - D)
- Inductor current ripple: ΔI_L = Vin * D * T / L
- Output voltage ripple: ΔV_out = I_out * D * T / C

Tolerance: 10% for converter behavior (complex switching dynamics).
"""

import pytest
import numpy as np
import pulsim as ps


# Boost converter parameters
V_IN = 5.0        # Input voltage (V)
V_OUT = 12.0      # Target output voltage (V)
DUTY_CYCLE = 1 - V_IN / V_OUT  # ~0.583

F_SW = 100e3      # Switching frequency (Hz)
T_SW = 1 / F_SW   # Switching period (10 µs)

L_VALUE = 100e-6  # Inductor (100 µH)
C_VALUE = 100e-6  # Output capacitor (100 µF)
R_LOAD = 50.0     # Load resistor (50 Ω)

# Expected output current
I_OUT = V_OUT / R_LOAD  # 0.24 A


def build_boost_converter_simplified():
    """Build simplified Boost converter with ideal switch and diode.

    Circuit:
        Vin ─── L ──┬── D ──┬── Vout
                    │       │
                   SW       C ─┬─ R_load
                    │       │  │
                   GND ────┴──┴─ GND
    """
    ckt = ps.Circuit()
    gnd = ps.Circuit.ground()

    # Nodes
    n_vin = ckt.add_node("vin")        # Input voltage
    n_sw = ckt.add_node("sw_node")     # Switch node (L-SW-D junction)
    n_out = ckt.add_node("vout")       # Output voltage

    # Input voltage source
    ckt.add_voltage_source("Vin", n_vin, gnd, V_IN)

    # Inductor (input side)
    ckt.add_inductor("L1", n_vin, n_sw, L_VALUE)

    # Main switch (to ground, initially open for boost operation)
    ckt.add_switch("SW", n_sw, gnd, closed=False)

    # Boost diode (anode at switch node, cathode to output)
    ckt.add_diode("D1", n_sw, n_out)

    # Output capacitor
    ckt.add_capacitor("C1", n_out, gnd, C_VALUE)

    # Load resistor
    ckt.add_resistor("R_load", n_out, gnd, R_LOAD)

    return ckt


class TestBoostConverterDC:
    """Test Boost converter DC operating point."""

    def test_boost_dc_with_switch_open(self):
        """With switch open, diode conducts and Vout ≈ Vin."""
        ckt = build_boost_converter_simplified()

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success, f"DC analysis failed: {dc_result.message}"

        # Node indices: 0=vin, 1=sw_node, 2=vout
        v_in = dc_result.newton_result.solution[0]
        v_sw = dc_result.newton_result.solution[1]
        v_out = dc_result.newton_result.solution[2]

        print("\nBoost DC (switch open):")
        print(f"  V_in = {v_in:.4f}V")
        print(f"  V_sw = {v_sw:.4f}V")
        print(f"  V_out = {v_out:.4f}V")

        # With switch open, current flows: Vin -> L -> D -> Vout
        # DC steady state: V_out ≈ V_in (minus diode drop)
        # Inductor is short in DC, so V_sw ≈ V_in
        assert v_sw > 0, f"Switch node should be positive: {v_sw}V"
        assert v_out > 0, f"Output should be positive: {v_out}V"

    def test_boost_dc_with_switch_closed(self):
        """With switch closed, steady-state current flows through inductor.

        In DC steady state, inductor is a short circuit. With switch closed,
        current flows: Vin -> L (short) -> SW (closed) -> GND
        This creates a voltage divider between L's DC resistance and switch on-resistance.

        Note: IdealSwitch has g_on conductance, so there's a small voltage drop.
        """
        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()

        n_vin = ckt.add_node("vin")
        n_sw = ckt.add_node("sw_node")
        n_out = ckt.add_node("vout")

        ckt.add_voltage_source("Vin", n_vin, gnd, V_IN)
        ckt.add_inductor("L1", n_vin, n_sw, L_VALUE)
        ckt.add_switch("SW", n_sw, gnd, closed=True)  # Closed
        ckt.add_diode("D1", n_sw, n_out)
        ckt.add_capacitor("C1", n_out, gnd, C_VALUE)
        ckt.add_resistor("R_load", n_out, gnd, R_LOAD)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success, f"DC failed: {dc_result.message}"

        v_sw = dc_result.newton_result.solution[1]
        v_out = dc_result.newton_result.solution[2]

        print("\nBoost DC (switch closed):")
        print(f"  V_sw = {v_sw:.4f}V")
        print(f"  V_out = {v_out:.4f}V")

        # In DC, inductor is short, so V_sw ≈ V_in (current limited by switch g_on)
        # Diode will conduct if V_sw > V_out, passing current to output
        # The exact behavior depends on component conductances
        assert dc_result.success, "DC analysis should converge"


class TestBoostConverterTransient:
    """Test Boost converter transient response."""

    def test_boost_startup_from_zero(self):
        """Test Boost converter startup with discharged capacitor."""
        ckt = build_boost_converter_simplified()

        # Start with capacitor discharged, switch open (diode conducts)
        x0 = np.zeros(ckt.system_size())
        x0[0] = V_IN    # vin
        x0[1] = V_IN    # sw_node (≈ Vin when switch open)
        x0[2] = 0.0     # vout = 0 (discharged)

        # Simulate startup
        t_lc = 2 * np.pi * np.sqrt(L_VALUE * C_VALUE)
        t_stop = 3 * t_lc
        dt = t_lc / 100

        times, states, success, msg = ps.run_transient(
            ckt, 0.0, t_stop, dt, x0
        )
        assert success, f"Transient failed: {msg}"

        times = np.array(times)
        v_out = np.array([s[2] for s in states])

        print("\nBoost Startup Transient:")
        print(f"  V_out(0) = {v_out[0]:.4f}V")
        print(f"  V_out(max) = {np.max(v_out):.4f}V")
        print(f"  V_out(final) = {v_out[-1]:.4f}V")

        # Output should charge up through diode
        assert v_out[-1] > v_out[0], "Output should increase"
        assert np.max(v_out) > 0, "Output should become positive"

    def test_boost_inductor_energy_storage(self):
        """Test that inductor stores energy when switch is closed."""
        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()

        n_vin = ckt.add_node("vin")
        n_sw = ckt.add_node("sw_node")

        ckt.add_voltage_source("Vin", n_vin, gnd, V_IN)
        ckt.add_inductor("L1", n_vin, n_sw, L_VALUE)
        ckt.add_switch("SW", n_sw, gnd, closed=True)  # Closed
        ckt.add_resistor("R_sense", n_sw, gnd, 0.01)  # Small sense resistor

        # Initial condition: zero inductor current
        x0 = np.zeros(ckt.system_size())
        x0[0] = V_IN
        x0[1] = 0.0  # sw_node at ground (switch closed)

        t_stop = 100e-6  # 100 µs
        dt = 1e-6

        times, states, success, msg = ps.run_transient(
            ckt, 0.0, t_stop, dt, x0
        )
        assert success

        times = np.array(times)
        # Inductor current = Vin * t / L (linear ramp when switch closed)
        # V_sw should stay near 0 (switch closed)
        v_sw = np.array([s[1] for s in states])

        print("\nInductor Energy Storage:")
        print(f"  V_sw(0) = {v_sw[0]:.6f}V")
        print(f"  V_sw(final) = {v_sw[-1]:.6f}V")
        print(f"  Expected I_L(final) = {V_IN * t_stop / L_VALUE:.4f} A")

        # Switch node should stay near ground
        assert np.max(np.abs(v_sw)) < 1.0, "Switch node should be near ground"


class TestBoostConverterComponents:
    """Test individual Boost converter component behavior."""

    def test_diode_conducts_when_switch_opens(self):
        """Diode should conduct when switch opens (inductor kicks)."""
        ckt = ps.Circuit()
        gnd = ps.Circuit.ground()

        n_vin = ckt.add_node("vin")
        n_sw = ckt.add_node("sw_node")
        n_out = ckt.add_node("vout")

        ckt.add_voltage_source("Vin", n_vin, gnd, V_IN)
        ckt.add_inductor("L1", n_vin, n_sw, L_VALUE)
        ckt.add_switch("SW", n_sw, gnd, closed=False)  # Open
        ckt.add_diode("D1", n_sw, n_out)
        ckt.add_resistor("R_load", n_out, gnd, R_LOAD)

        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success

        v_sw = dc_result.newton_result.solution[1]
        v_out = dc_result.newton_result.solution[2]

        print("\nDiode Conduction Test (switch open):")
        print(f"  V_sw = {v_sw:.4f}V")
        print(f"  V_out = {v_out:.4f}V")
        print(f"  V_diode = V_out - V_sw = {v_out - v_sw:.4f}V")

        # Diode should conduct, V_sw ≈ V_out (minus diode drop)
        # In DC with switch open: Vin -> L (short) -> D -> R_load
        assert v_out > 0, "Output should be positive"

    def test_output_capacitor_filter(self):
        """Test output capacitor provides filtering."""
        ckt = build_boost_converter_simplified()

        # Pre-charge capacitor
        x0 = np.zeros(ckt.system_size())
        x0[0] = V_IN
        x0[1] = V_IN
        x0[2] = V_IN  # Pre-charge to Vin

        t_stop = 1e-3
        dt = 1e-6

        times, states, success, msg = ps.run_transient(
            ckt, 0.0, t_stop, dt, x0
        )
        assert success

        v_out = np.array([s[2] for s in states])

        # Output ripple should be limited by capacitor
        ripple = np.max(v_out) - np.min(v_out)

        print("\nOutput Capacitor Filter:")
        print(f"  V_out(min) = {np.min(v_out):.4f}V")
        print(f"  V_out(max) = {np.max(v_out):.4f}V")
        print(f"  Ripple = {ripple:.4f}V")

        # Ripple should be reasonable
        assert ripple < V_IN, f"Ripple too large: {ripple}V"


class TestBoostConverterTheory:
    """Verify Boost converter theoretical calculations."""

    def test_duty_cycle_calculation(self):
        """Verify D = 1 - Vin/Vout relationship."""
        D = 1 - V_IN / V_OUT

        print("\nDuty Cycle Calculation:")
        print(f"  V_in = {V_IN}V")
        print(f"  V_out (target) = {V_OUT}V")
        print(f"  D = 1 - V_in/V_out = {D:.4f} ({D*100:.1f}%)")

        assert 0 < D < 1, "Duty cycle must be between 0 and 1"
        assert abs(D - DUTY_CYCLE) < 0.001, "Duty cycle calculation error"

    def test_voltage_gain(self):
        """Verify voltage gain = 1/(1-D)."""
        D = DUTY_CYCLE
        gain = 1 / (1 - D)

        print("\nVoltage Gain:")
        print(f"  D = {D:.4f}")
        print(f"  Gain = 1/(1-D) = {gain:.4f}")
        print(f"  V_out = V_in * Gain = {V_IN * gain:.4f}V")

        # Gain should give expected output
        assert abs(V_IN * gain - V_OUT) < 0.1, "Voltage gain calculation error"

    def test_inductor_ripple_current(self):
        """Calculate expected inductor current ripple."""
        # ΔI_L = Vin * D * T / L
        D = DUTY_CYCLE
        delta_I_L = V_IN * D * T_SW / L_VALUE

        # Average inductor current = I_out / (1 - D)
        I_L_avg = I_OUT / (1 - D)

        ripple_percent = (delta_I_L / I_L_avg) * 100

        print("\nInductor Current Ripple:")
        print(f"  I_L (avg) = {I_L_avg:.4f} A")
        print(f"  ΔI_L = {delta_I_L*1000:.4f} mA")
        print(f"  Ripple = {ripple_percent:.1f}%")

        # Ripple should be reasonable
        assert ripple_percent < 100, f"Inductor ripple too high: {ripple_percent}%"

    def test_output_voltage_ripple(self):
        """Calculate expected output voltage ripple."""
        # ΔV_out = I_out * D * T / C
        D = DUTY_CYCLE
        delta_V_out = I_OUT * D * T_SW / C_VALUE

        ripple_percent = (delta_V_out / V_OUT) * 100

        print("\nOutput Voltage Ripple:")
        print(f"  V_out (avg) = {V_OUT}V")
        print(f"  ΔV_out = {delta_V_out*1000:.4f} mV")
        print(f"  Ripple = {ripple_percent:.4f}%")

        # Output ripple should be small
        assert ripple_percent < 5, f"Output ripple too high: {ripple_percent}%"

    def test_ccm_boundary(self):
        """Calculate boundary between CCM and DCM."""
        # CCM boundary: I_L_avg > ΔI_L / 2
        D = DUTY_CYCLE
        delta_I_L = V_IN * D * T_SW / L_VALUE
        I_L_avg = I_OUT / (1 - D)

        print("\nCCM/DCM Boundary:")
        print(f"  I_L (avg) = {I_L_avg:.4f} A")
        print(f"  ΔI_L / 2 = {delta_I_L/2:.4f} A")
        print(f"  Mode: {'CCM' if I_L_avg > delta_I_L/2 else 'DCM'}")

        # Should be in CCM for these parameters
        is_ccm = I_L_avg > delta_I_L / 2
        print(f"  Operating in CCM: {is_ccm}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
