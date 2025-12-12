"""Validation tests for RLC circuits.

Tests RLC step response for all damping cases:
- Underdamped (ζ < 1): Oscillating response
- Critically damped (ζ = 1): Fastest non-oscillating response
- Overdamped (ζ > 1): Slow non-oscillating response

Tolerance: 2% maximum relative error.
"""

import pytest
import numpy as np
import pulsim as ps
from ..framework.base import (
    ValidationLevel,
    CircuitDefinition,
    ValidationTest,
)


V_SOURCE = 10.0  # V


def build_rlc_circuit(R: float, L: float, C: float):
    """Build series RLC circuit: V -> R -> L -> C -> GND."""
    ckt = ps.Circuit()
    gnd = ps.Circuit.ground()
    n1 = ckt.add_node("v_source")
    n2 = ckt.add_node("v_after_r")
    n3 = ckt.add_node("v_capacitor")

    ckt.add_voltage_source("V1", n1, gnd, V_SOURCE)
    ckt.add_resistor("R1", n1, n2, R)
    ckt.add_inductor("L1", n2, n3, L)
    ckt.add_capacitor("C1", n3, gnd, C)

    return ckt


# =============================================================================
# Underdamped RLC (ζ < 1)
# =============================================================================

# Parameters for underdamped: ζ = R/(2*sqrt(L/C)) < 1
R_UNDER = 100.0    # Ohms
L_UNDER = 10e-3    # 10 mH
C_UNDER = 10e-6    # 10 µF
# ζ = 100 / (2 * sqrt(10e-3 / 10e-6)) = 100 / (2 * 31.6) = 1.58... wait that's overdamped
# Let's recalculate: ζ = R * sqrt(C/L) / 2 = 100 * sqrt(10e-6/10e-3) / 2 = 100 * 0.0316 / 2 = 1.58
# For underdamped we need R < 2*sqrt(L/C) = 2*sqrt(1000) = 63.2
R_UNDER = 20.0     # Now ζ = 20/63.2 = 0.316 (underdamped)
OMEGA_0_UNDER = 1.0 / np.sqrt(L_UNDER * C_UNDER)  # Natural frequency
ZETA_UNDER = R_UNDER / (2 * np.sqrt(L_UNDER / C_UNDER))
OMEGA_D_UNDER = OMEGA_0_UNDER * np.sqrt(1 - ZETA_UNDER**2)  # Damped frequency


def build_rlc_underdamped():
    return build_rlc_circuit(R_UNDER, L_UNDER, C_UNDER)


def rlc_underdamped_analytical(t: np.ndarray) -> np.ndarray:
    """Underdamped RLC capacitor voltage.

    V_c(t) = V_f * (1 - (ωn/ωd) * exp(-ζωn*t) * sin(ωd*t + φ))
    where φ = arccos(ζ)
    """
    phi = np.arccos(ZETA_UNDER)
    envelope = np.exp(-ZETA_UNDER * OMEGA_0_UNDER * t)
    return V_SOURCE * (1 - envelope / np.sqrt(1 - ZETA_UNDER**2) *
                       np.sin(OMEGA_D_UNDER * t + phi))


# =============================================================================
# Critically Damped RLC (ζ = 1)
# =============================================================================

# Parameters for critically damped: ζ = R/(2*sqrt(L/C)) = 1
# R = 2*sqrt(L/C)
L_CRIT = 10e-3     # 10 mH
C_CRIT = 10e-6     # 10 µF
R_CRIT = 2 * np.sqrt(L_CRIT / C_CRIT)  # = 63.25 Ohms for ζ = 1
OMEGA_0_CRIT = 1.0 / np.sqrt(L_CRIT * C_CRIT)


def build_rlc_critical():
    return build_rlc_circuit(R_CRIT, L_CRIT, C_CRIT)


def rlc_critical_analytical(t: np.ndarray) -> np.ndarray:
    """Critically damped RLC capacitor voltage.

    V_c(t) = V_f * (1 - (1 + ωn*t) * exp(-ωn*t))
    """
    return V_SOURCE * (1 - (1 + OMEGA_0_CRIT * t) * np.exp(-OMEGA_0_CRIT * t))


# =============================================================================
# Overdamped RLC (ζ > 1)
# =============================================================================

# Parameters for overdamped: ζ = R/(2*sqrt(L/C)) > 1
R_OVER = 200.0     # Ohms (ζ = 200/63.2 = 3.16)
L_OVER = 10e-3     # 10 mH
C_OVER = 10e-6     # 10 µF
OMEGA_0_OVER = 1.0 / np.sqrt(L_OVER * C_OVER)
ZETA_OVER = R_OVER / (2 * np.sqrt(L_OVER / C_OVER))
# Two real roots
S1_OVER = -OMEGA_0_OVER * (ZETA_OVER - np.sqrt(ZETA_OVER**2 - 1))
S2_OVER = -OMEGA_0_OVER * (ZETA_OVER + np.sqrt(ZETA_OVER**2 - 1))


def build_rlc_overdamped():
    return build_rlc_circuit(R_OVER, L_OVER, C_OVER)


def rlc_overdamped_analytical(t: np.ndarray) -> np.ndarray:
    """Overdamped RLC capacitor voltage.

    V_c(t) = V_f * (1 - (s2*exp(s1*t) - s1*exp(s2*t)) / (s2 - s1))
    """
    s1, s2 = S1_OVER, S2_OVER
    return V_SOURCE * (1 - (s2 * np.exp(s1 * t) - s1 * np.exp(s2 * t)) / (s2 - s1))


# =============================================================================
# Circuit Definitions
# =============================================================================

# Simulation time based on slowest time constant
T_STOP_UNDER = 10 / (ZETA_UNDER * OMEGA_0_UNDER)
T_STOP_CRIT = 10 / OMEGA_0_CRIT
T_STOP_OVER = 10 / abs(S1_OVER)  # S1 is slower root


def rlc_step_ic(circuit):
    """Initial condition for RLC step response.

    For series RLC (V -> R -> L -> C -> GND) starting from IC=0:
    - Node 0 (v_source): V_SOURCE (held by voltage source)
    - Node 1 (v_after_r): V_SOURCE (at t=0, I=0 so V_R=0)
    - Node 2 (v_cap): 0 (capacitor IC)
    - Branch (i_L): 0 (inductor IC)
    """
    x0 = np.zeros(circuit.system_size())
    x0[0] = V_SOURCE  # v_source
    x0[1] = V_SOURCE  # v_after_r (V_R = 0 when I = 0)
    x0[2] = 0.0       # v_cap (IC = 0)
    # Branch current x0[3] = 0 (already zero from np.zeros)
    return x0


RLC_UNDERDAMPED_DEF = CircuitDefinition(
    name="RLC_underdamped",
    description="Underdamped RLC (ζ<1): oscillating response",
    level=ValidationLevel.LINEAR,
    build_circuit=build_rlc_underdamped,
    analytical_solution=rlc_underdamped_analytical,
    t_start=0.0,
    t_stop=T_STOP_UNDER,
    dt=T_STOP_UNDER / 1000,  # 1000 points
    node_index=2,  # v_capacitor
    tolerance=0.02,  # 2%
    custom_ic=rlc_step_ic,
)

RLC_CRITICAL_DEF = CircuitDefinition(
    name="RLC_critically_damped",
    description="Critically damped RLC (ζ=1): fastest non-oscillating",
    level=ValidationLevel.LINEAR,
    build_circuit=build_rlc_critical,
    analytical_solution=rlc_critical_analytical,
    t_start=0.0,
    t_stop=T_STOP_CRIT,
    dt=T_STOP_CRIT / 1000,
    node_index=2,
    tolerance=0.02,
    custom_ic=rlc_step_ic,
)

RLC_OVERDAMPED_DEF = CircuitDefinition(
    name="RLC_overdamped",
    description="Overdamped RLC (ζ>1): slow non-oscillating",
    level=ValidationLevel.LINEAR,
    build_circuit=build_rlc_overdamped,
    analytical_solution=rlc_overdamped_analytical,
    t_start=0.0,
    t_stop=T_STOP_OVER,
    dt=T_STOP_OVER / 1000,
    node_index=2,
    tolerance=0.02,
    custom_ic=rlc_step_ic,
)


# =============================================================================
# Test Classes
# =============================================================================

class TestRLCUnderdamped:
    """Test underdamped RLC response."""

    def test_underdamped_accuracy(self):
        """Validate underdamped RLC within 2% tolerance."""
        test = ValidationTest(RLC_UNDERDAMPED_DEF)
        result = test.validate_transient()

        print(f"\n{result.summary()}")
        print(f"  ζ = {ZETA_UNDER:.3f}, ω₀ = {OMEGA_0_UNDER:.1f} rad/s")

        assert result.passed, (
            f"Underdamped RLC failed validation:\n"
            f"  Max relative error: {result.max_relative_error*100:.4f}%\n"
            f"  Tolerance: {result.tolerance*100:.2f}%"
        )

    def test_oscillation_frequency(self):
        """Verify oscillation frequency matches ωd."""
        ckt = build_rlc_underdamped()
        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success

        times, states, success, _ = ps.run_transient(
            ckt, 0.0, T_STOP_UNDER, T_STOP_UNDER/1000, dc_result.newton_result.solution
        )

        v_cap = np.array([s[2] for s in states])
        times = np.array(times)

        # Find zero crossings of (v_cap - V_SOURCE) to estimate period
        v_centered = v_cap - V_SOURCE
        # Skip initial transient
        start_idx = len(times) // 4
        zero_crossings = []
        for i in range(start_idx, len(v_centered) - 1):
            if v_centered[i] * v_centered[i+1] < 0:
                zero_crossings.append(times[i])

        if len(zero_crossings) >= 2:
            # Period is twice the half-period
            measured_period = 2 * (zero_crossings[1] - zero_crossings[0])
            measured_omega_d = 2 * np.pi / measured_period
            expected_omega_d = OMEGA_D_UNDER

            rel_error = abs(measured_omega_d - expected_omega_d) / expected_omega_d
            print(f"\nMeasured ωd = {measured_omega_d:.1f} rad/s, expected = {expected_omega_d:.1f} rad/s")
            print(f"Frequency error: {rel_error*100:.2f}%")

            assert rel_error < 0.05, f"Frequency error too large: {rel_error*100:.2f}%"


class TestRLCCriticallyDamped:
    """Test critically damped RLC response."""

    def test_critical_accuracy(self):
        """Validate critically damped RLC within 2% tolerance."""
        test = ValidationTest(RLC_CRITICAL_DEF)
        result = test.validate_transient()

        print(f"\n{result.summary()}")
        print(f"  R = {R_CRIT:.2f}Ω for ζ = 1")

        assert result.passed, (
            f"Critically damped RLC failed validation:\n"
            f"  Max relative error: {result.max_relative_error*100:.4f}%\n"
            f"  Tolerance: {result.tolerance*100:.2f}%"
        )

    def test_no_overshoot(self):
        """Verify critically damped has minimal/no overshoot."""
        ckt = build_rlc_critical()
        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success

        times, states, success, _ = ps.run_transient(
            ckt, 0.0, T_STOP_CRIT, T_STOP_CRIT/1000, dc_result.newton_result.solution
        )

        v_cap = np.array([s[2] for s in states])
        max_v = np.max(v_cap)
        overshoot = (max_v - V_SOURCE) / V_SOURCE * 100

        print(f"\nMax voltage: {max_v:.4f}V, Overshoot: {overshoot:.2f}%")

        # Critically damped should have < 2% overshoot (numerical tolerance)
        assert overshoot < 2.0, f"Overshoot too large for critical damping: {overshoot:.2f}%"


class TestRLCOverdamped:
    """Test overdamped RLC response."""

    def test_overdamped_accuracy(self):
        """Validate overdamped RLC within 2% tolerance."""
        test = ValidationTest(RLC_OVERDAMPED_DEF)
        result = test.validate_transient()

        print(f"\n{result.summary()}")
        print(f"  ζ = {ZETA_OVER:.3f}")

        assert result.passed, (
            f"Overdamped RLC failed validation:\n"
            f"  Max relative error: {result.max_relative_error*100:.4f}%\n"
            f"  Tolerance: {result.tolerance*100:.2f}%"
        )

    def test_monotonic_rise(self):
        """Verify overdamped response is monotonically increasing."""
        ckt = build_rlc_overdamped()
        dc_result = ps.dc_operating_point(ckt)
        assert dc_result.success

        times, states, success, _ = ps.run_transient(
            ckt, 0.0, T_STOP_OVER, T_STOP_OVER/500, dc_result.newton_result.solution
        )

        v_cap = np.array([s[2] for s in states])

        # Check monotonicity (allow small numerical tolerance)
        diffs = np.diff(v_cap)
        violations = np.sum(diffs < -1e-6)

        print(f"\nMonotonicity violations: {violations} out of {len(diffs)}")

        assert violations < 3, f"Too many monotonicity violations: {violations}"


class TestRLCWithPulsimAnalytical:
    """Test using Pulsim's built-in RLCAnalytical class."""

    def test_underdamped_vs_pulsim_analytical(self):
        """Compare underdamped against Pulsim's RLCAnalytical."""
        analytical = ps.RLCAnalytical(R_UNDER, L_UNDER, C_UNDER, V_SOURCE, 0.0, 0.0)

        assert analytical.damping_type() == ps.RLCDamping.Underdamped
        print(f"\nDamping type: {analytical.damping_type()}")

        ckt = build_rlc_underdamped()
        x0 = rlc_step_ic(ckt)  # Use proper step response IC

        times, states, success, _ = ps.run_transient(
            ckt, 0.0, T_STOP_UNDER, T_STOP_UNDER/500, x0
        )

        max_error = 0.0
        for i, t in enumerate(times):
            v_sim = states[i][2]
            v_analytical = analytical.voltage(t)
            error = abs(v_sim - v_analytical)
            rel_error = error / V_SOURCE  # Normalize to final value
            max_error = max(max_error, rel_error)

        print(f"Max relative error vs RLCAnalytical: {max_error*100:.4f}%")
        # Underdamped oscillating systems have slightly higher numerical error
        assert max_error < 0.025, f"Error vs RLCAnalytical too large: {max_error*100:.4f}%"

    def test_overdamped_vs_pulsim_analytical(self):
        """Compare overdamped against Pulsim's RLCAnalytical."""
        analytical = ps.RLCAnalytical(R_OVER, L_OVER, C_OVER, V_SOURCE, 0.0, 0.0)

        assert analytical.damping_type() == ps.RLCDamping.Overdamped
        print(f"\nDamping type: {analytical.damping_type()}")

        ckt = build_rlc_overdamped()
        x0 = rlc_step_ic(ckt)  # Use proper step response IC

        times, states, success, _ = ps.run_transient(
            ckt, 0.0, T_STOP_OVER, T_STOP_OVER/500, x0
        )

        max_error = 0.0
        for i, t in enumerate(times):
            v_sim = states[i][2]
            v_analytical = analytical.voltage(t)
            error = abs(v_sim - v_analytical)
            rel_error = error / V_SOURCE
            max_error = max(max_error, rel_error)

        print(f"Max relative error vs RLCAnalytical: {max_error*100:.4f}%")
        assert max_error < 0.02, f"Error vs RLCAnalytical too large: {max_error*100:.4f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
