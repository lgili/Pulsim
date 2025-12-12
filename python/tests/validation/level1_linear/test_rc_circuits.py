"""Validation tests for RC circuits.

Tests RC step response and discharge against analytical solutions.
Tolerance: 1% maximum relative error.
"""

import pytest
import numpy as np
import pulsim as ps


# Circuit parameters
V_SOURCE = 5.0  # V
R_VALUE = 1000.0  # Ohms
C_VALUE = 1e-6  # F (1 µF)
TAU = R_VALUE * C_VALUE  # Time constant = 1 ms


def build_rc_step_circuit():
    """Build RC circuit for step response: V -> R -> C -> GND."""
    ckt = ps.Circuit()
    gnd = ps.Circuit.ground()
    n1 = ckt.add_node("v_source")
    n2 = ckt.add_node("v_cap")

    ckt.add_voltage_source("V1", n1, gnd, V_SOURCE)
    ckt.add_resistor("R1", n1, n2, R_VALUE)
    ckt.add_capacitor("C1", n2, gnd, C_VALUE)

    return ckt


def rc_step_analytical(t: np.ndarray) -> np.ndarray:
    """Analytical solution for RC step response.

    V(t) = Vf * (1 - exp(-t/τ))
    """
    return V_SOURCE * (1 - np.exp(-t / TAU))


def build_rc_discharge_circuit():
    """Build RC circuit for discharge: C(V0) -> R -> GND.

    Initial condition: capacitor charged to V_SOURCE.
    """
    ckt = ps.Circuit()
    gnd = ps.Circuit.ground()
    n1 = ckt.add_node("v_cap")

    # Capacitor with initial condition
    ckt.add_capacitor("C1", n1, gnd, C_VALUE, ic=V_SOURCE)
    ckt.add_resistor("R1", n1, gnd, R_VALUE)

    return ckt


def rc_discharge_analytical(t: np.ndarray) -> np.ndarray:
    """Analytical solution for RC discharge.

    V(t) = V0 * exp(-t/τ)
    """
    return V_SOURCE * np.exp(-t / TAU)


class TestRCStepResponse:
    """Test RC step response against analytical solution."""

    def test_step_response_accuracy(self):
        """Validate RC step response within 1% tolerance."""
        ckt = build_rc_step_circuit()

        # For step response, start with capacitor at 0V (not DC steady state)
        x0 = np.zeros(ckt.system_size())
        x0[0] = V_SOURCE  # v_source node = 5V
        x0[1] = 0.0       # v_cap node = 0V (initial condition)

        times, states, success, msg = ps.run_transient(
            ckt, 0.0, 5*TAU, TAU/100, x0
        )
        assert success, f"Transient failed: {msg}"

        times = np.array(times)
        v_sim = np.array([s[1] for s in states])
        v_analytical = rc_step_analytical(times)

        # Calculate error
        max_error = np.max(np.abs(v_sim - v_analytical))
        max_rel_error = max_error / V_SOURCE

        print(f"\nRC Step Response Accuracy:")
        print(f"  Max absolute error: {max_error:.6f}V")
        print(f"  Max relative error: {max_rel_error*100:.4f}%")

        assert max_rel_error < 0.01, (
            f"RC step response failed validation:\n"
            f"  Max relative error: {max_rel_error*100:.4f}%\n"
            f"  Tolerance: 1.00%"
        )

    def test_step_response_at_tau(self):
        """Verify voltage at t=τ is ~63.2% of final value."""
        ckt = build_rc_step_circuit()

        # Start with IC=0
        x0 = np.zeros(ckt.system_size())
        x0[0] = V_SOURCE

        times, states, success, _ = ps.run_transient(
            ckt, 0.0, 2*TAU, TAU/100, x0
        )

        times = np.array(times)
        idx_tau = np.argmin(np.abs(times - TAU))
        v_at_tau = states[idx_tau][1]

        expected = V_SOURCE * (1 - np.exp(-1))  # 63.2%
        rel_error = abs(v_at_tau - expected) / expected

        print(f"\nAt t=τ: V={v_at_tau:.4f}V, expected={expected:.4f}V, error={rel_error*100:.2f}%")

        assert rel_error < 0.02, f"Voltage at t=τ error too large: {rel_error*100:.2f}%"

    def test_step_response_final_value(self):
        """Verify final value approaches source voltage."""
        ckt = build_rc_step_circuit()

        x0 = np.zeros(ckt.system_size())
        x0[0] = V_SOURCE

        times, states, success, _ = ps.run_transient(
            ckt, 0.0, 10*TAU, TAU/50, x0
        )

        final_voltage = states[-1][1]
        rel_error = abs(final_voltage - V_SOURCE) / V_SOURCE

        print(f"\nFinal value: {final_voltage:.6f}V, expected: {V_SOURCE}V, error: {rel_error*100:.4f}%")

        assert rel_error < 0.001, f"Final value error too large: {rel_error*100:.4f}%"


class TestRCDischarge:
    """Test RC discharge against analytical solution."""

    def test_discharge_accuracy(self):
        """Validate RC discharge within 1% tolerance."""
        ckt = build_rc_discharge_circuit()

        # Start with capacitor at V_SOURCE
        x0 = np.zeros(ckt.system_size())
        x0[0] = V_SOURCE

        times, states, success, msg = ps.run_transient(
            ckt, 0.0, 5*TAU, TAU/100, x0
        )
        assert success, f"Transient failed: {msg}"

        times = np.array(times)
        v_sim = np.array([s[0] for s in states])
        v_analytical = rc_discharge_analytical(times)

        max_error = np.max(np.abs(v_sim - v_analytical))
        max_rel_error = max_error / V_SOURCE

        print(f"\nRC Discharge Accuracy:")
        print(f"  Max absolute error: {max_error:.6f}V")
        print(f"  Max relative error: {max_rel_error*100:.4f}%")

        assert max_rel_error < 0.01, (
            f"RC discharge failed validation:\n"
            f"  Max relative error: {max_rel_error*100:.4f}%"
        )

    def test_discharge_at_tau(self):
        """Verify voltage at t=τ is ~36.8% of initial value."""
        ckt = build_rc_discharge_circuit()

        x0 = np.zeros(ckt.system_size())
        x0[0] = V_SOURCE

        times, states, success, _ = ps.run_transient(
            ckt, 0.0, 2*TAU, TAU/100, x0
        )

        times = np.array(times)
        idx_tau = np.argmin(np.abs(times - TAU))
        v_at_tau = states[idx_tau][0]

        expected = V_SOURCE * np.exp(-1)  # 36.8%
        rel_error = abs(v_at_tau - expected) / expected

        print(f"\nAt t=τ: V={v_at_tau:.4f}V, expected={expected:.4f}V, error={rel_error*100:.2f}%")

        assert rel_error < 0.02, f"Voltage at t=τ error too large: {rel_error*100:.2f}%"


class TestRCWithPulsimAnalytical:
    """Test using Pulsim's built-in RCAnalytical class."""

    def test_step_response_vs_pulsim_analytical(self):
        """Compare simulation against Pulsim's RCAnalytical."""
        # RCAnalytical(R, C, V_initial, V_final)
        analytical = ps.RCAnalytical(R_VALUE, C_VALUE, 0.0, V_SOURCE)

        ckt = build_rc_step_circuit()

        # Start with IC=0
        x0 = np.zeros(ckt.system_size())
        x0[0] = V_SOURCE

        times, states, success, _ = ps.run_transient(
            ckt, 0.0, 5*TAU, TAU/100, x0
        )

        # Calculate max absolute error, normalized by V_SOURCE
        max_abs_error = 0.0
        for i, t in enumerate(times):
            v_sim = states[i][1]
            v_analytical = analytical.voltage(t)
            error = abs(v_sim - v_analytical)
            max_abs_error = max(max_abs_error, error)

        max_rel_error = max_abs_error / V_SOURCE

        print(f"\nMax absolute error vs RCAnalytical: {max_abs_error:.6f}V")
        print(f"Max relative error: {max_rel_error*100:.4f}%")
        assert max_rel_error < 0.01, f"Error vs RCAnalytical too large: {max_rel_error*100:.4f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
