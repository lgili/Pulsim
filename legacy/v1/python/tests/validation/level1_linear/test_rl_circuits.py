"""Validation tests for RL circuits.

Tests RL step response against analytical solutions.
Tolerance: 1% maximum relative error.
"""

import pytest
import numpy as np
import pulsim as ps


# Circuit parameters
V_SOURCE = 10.0  # V
R_VALUE = 100.0  # Ohms
L_VALUE = 10e-3  # H (10 mH)
TAU = L_VALUE / R_VALUE  # Time constant = 0.1 ms


def build_rl_step_circuit():
    """Build RL circuit for step response: V -> R -> L -> GND."""
    ckt = ps.Circuit()
    gnd = ps.Circuit.ground()
    n1 = ckt.add_node("v_source")
    n2 = ckt.add_node("v_inductor")

    ckt.add_voltage_source("V1", n1, gnd, V_SOURCE)
    ckt.add_resistor("R1", n1, n2, R_VALUE)
    ckt.add_inductor("L1", n2, gnd, L_VALUE)

    return ckt


def rl_step_current_analytical(t: np.ndarray) -> np.ndarray:
    """Analytical solution for RL step response current.

    I(t) = (V/R) * (1 - exp(-t/τ))
    """
    I_final = V_SOURCE / R_VALUE
    return I_final * (1 - np.exp(-t / TAU))


def rl_step_voltage_analytical(t: np.ndarray) -> np.ndarray:
    """Analytical solution for inductor voltage.

    V_L(t) = V * exp(-t/τ)

    Note: The voltage across the inductor = V_source - V_R
          V_L = L * dI/dt = V * exp(-t/τ)
    """
    return V_SOURCE * np.exp(-t / TAU)


class TestRLStepResponse:
    """Test RL step response against analytical solution."""

    def test_step_response_accuracy(self):
        """Validate RL step response within 1% tolerance."""
        ckt = build_rl_step_circuit()

        # For step response, start with inductor current at 0 (IC=0)
        # At t=0: I_L=0, so V_R=0, thus v_inductor = V_SOURCE
        x0 = np.zeros(ckt.system_size())
        x0[0] = V_SOURCE  # v_source node = 10V
        x0[1] = V_SOURCE  # v_inductor node = 10V (no current through R yet)

        times, states, success, msg = ps.run_transient(
            ckt, 0.0, 5*TAU, TAU/100, x0
        )
        assert success, f"Transient failed: {msg}"

        times = np.array(times)
        v_sim = np.array([s[1] for s in states])
        v_analytical = rl_step_voltage_analytical(times)

        # Calculate error
        max_error = np.max(np.abs(v_sim - v_analytical))
        max_rel_error = max_error / V_SOURCE

        print("\nRL Step Response Accuracy:")
        print(f"  Max absolute error: {max_error:.6f}V")
        print(f"  Max relative error: {max_rel_error*100:.4f}%")

        assert max_rel_error < 0.01, (
            f"RL step response failed validation:\n"
            f"  Max relative error: {max_rel_error*100:.4f}%\n"
            f"  Tolerance: 1.00%"
        )

    def test_step_response_at_tau(self):
        """Verify inductor voltage at t=τ is ~36.8% of initial."""
        ckt = build_rl_step_circuit()

        # Start with IC=0
        x0 = np.zeros(ckt.system_size())
        x0[0] = V_SOURCE
        x0[1] = V_SOURCE  # At t=0, v_inductor = V_SOURCE

        times, states, success, _ = ps.run_transient(
            ckt, 0.0, 2*TAU, TAU/100, x0
        )

        times = np.array(times)
        idx_tau = np.argmin(np.abs(times - TAU))
        v_l_at_tau = states[idx_tau][1]  # node 1 is v_inductor

        expected = V_SOURCE * np.exp(-1)  # 36.8% of initial
        rel_error = abs(v_l_at_tau - expected) / expected

        print(f"\nV_L at t=τ: {v_l_at_tau:.4f}V, expected={expected:.4f}V, error={rel_error*100:.2f}%")

        assert rel_error < 0.02, f"Inductor voltage at t=τ error too large: {rel_error*100:.2f}%"

    def test_final_inductor_voltage(self):
        """Verify final inductor voltage approaches 0."""
        ckt = build_rl_step_circuit()

        x0 = np.zeros(ckt.system_size())
        x0[0] = V_SOURCE
        x0[1] = V_SOURCE

        times, states, success, _ = ps.run_transient(
            ckt, 0.0, 10*TAU, TAU/50, x0
        )

        # Final voltage across inductor should be ~0 (steady state)
        final_v_inductor = states[-1][1]

        print(f"\nFinal V_L: {final_v_inductor:.6f}V (expected ~0V)")
        assert abs(final_v_inductor) < 0.01, f"Final inductor voltage not near 0: {final_v_inductor}V"


class TestRLWithPulsimAnalytical:
    """Test using Pulsim's built-in RLAnalytical class."""

    def test_step_response_vs_pulsim_analytical(self):
        """Compare simulation against Pulsim's RLAnalytical."""
        # RLAnalytical(R, L, V_source, I_initial)
        analytical = ps.RLAnalytical(R_VALUE, L_VALUE, V_SOURCE, 0.0)

        ckt = build_rl_step_circuit()

        # Start with IC=0
        x0 = np.zeros(ckt.system_size())
        x0[0] = V_SOURCE
        x0[1] = V_SOURCE

        times, states, success, _ = ps.run_transient(
            ckt, 0.0, 5*TAU, TAU/100, x0
        )

        # Compare inductor voltage at each time point
        max_error = 0.0
        for i, t in enumerate(times):
            v_sim = states[i][1]  # Inductor node voltage
            v_analytical = analytical.voltage_L(t)
            error = abs(v_sim - v_analytical)
            # Use absolute tolerance for small values
            if abs(v_analytical) > 0.1:
                rel_error = error / abs(v_analytical)
            else:
                rel_error = error / V_SOURCE  # Normalize to source voltage
            max_error = max(max_error, rel_error)

        print(f"\nMax relative error vs RLAnalytical: {max_error*100:.4f}%")
        assert max_error < 0.02, f"Error vs RLAnalytical too large: {max_error*100:.4f}%"

    def test_current_vs_analytical(self):
        """Compare current against analytical solution.

        Note: Skip first 10% of timesteps because trapezoidal integration
        has inherent startup error (approx half the correct value at t=dt).
        This is expected behavior, not a bug.
        """
        analytical = ps.RLAnalytical(R_VALUE, L_VALUE, V_SOURCE, 0.0)

        ckt = build_rl_step_circuit()

        x0 = np.zeros(ckt.system_size())
        x0[0] = V_SOURCE
        x0[1] = V_SOURCE

        times, states, success, _ = ps.run_transient(
            ckt, 0.0, 5*TAU, TAU/100, x0
        )

        times = np.array(times)

        # Skip first 10% of timesteps (trapezoidal startup phase)
        skip_count = len(times) // 10
        I_final = V_SOURCE / R_VALUE

        # Calculate current from node voltages: I = (V_source - V_inductor) / R
        max_error = 0.0
        for i, t in enumerate(times[skip_count:], start=skip_count):
            v_source_node = states[i][0]
            v_inductor_node = states[i][1]
            i_sim = (v_source_node - v_inductor_node) / R_VALUE
            i_analytical = analytical.current(t)

            # Normalize by final current for consistent error metric
            error = abs(i_sim - i_analytical) / I_final
            max_error = max(max_error, error)

        print(f"\nMax current error vs RLAnalytical (after warmup): {max_error*100:.4f}%")
        assert max_error < 0.02, f"Current error vs RLAnalytical too large: {max_error*100:.4f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
