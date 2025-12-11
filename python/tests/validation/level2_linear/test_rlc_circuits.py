"""
RLC Circuit Validation Tests

Validates Pulsim RLC circuit simulations against analytical solutions.
Tests underdamped, critically damped, and overdamped responses.
"""

import pytest
import numpy as np
import pulsim as sl

from ..framework.base import (
    ValidationLevel,
    CircuitDefinition,
    ValidationTest,
)
from ..framework.analytical import AnalyticalSolutions
from ..framework.comparator import ResultComparator


# =============================================================================
# Circuit Builders for Pulsim
# =============================================================================

def build_rlc_series_circuit(
    R: float = 10.0,
    L: float = 10e-3,
    C: float = 1e-6,
    V0: float = 10.0,
    V_C_initial: float = 0.0,
    I_L_initial: float = 0.0,
):
    """Build series RLC circuit for Pulsim."""
    circuit = sl.Circuit()
    circuit.add_voltage_source("V1", "in", "0", V0)
    circuit.add_resistor("R1", "in", "n1", R)
    circuit.add_inductor("L1", "n1", "out", L, ic=I_L_initial)
    circuit.add_capacitor("C1", "out", "0", C, ic=V_C_initial)
    return circuit


# =============================================================================
# SPICE Netlists for NgSpice Reference
# =============================================================================

def get_rlc_netlist(R: float, L: float, C: float, V0: float = 10.0) -> str:
    """Generate SPICE netlist for series RLC circuit."""
    return f"""
* Series RLC Step Response
V1 in 0 DC {V0}
R1 in n1 {R}
L1 n1 out {L} IC=0
C1 out 0 {C} IC=0
.ic V(out)=0 I(L1)=0
"""


# =============================================================================
# Circuit Definitions
# =============================================================================

class RLCCircuitDefinitions:
    """Collection of RLC circuit test definitions."""

    @staticmethod
    def get_damping_info(R: float, L: float, C: float) -> dict:
        """Calculate damping characteristics."""
        omega_0 = 1.0 / np.sqrt(L * C)
        alpha = R / (2.0 * L)
        zeta = alpha / omega_0

        if zeta < 1.0:
            damping_type = "underdamped"
            omega_d = omega_0 * np.sqrt(1 - zeta**2)
            period = 2 * np.pi / omega_d
        elif np.isclose(zeta, 1.0, rtol=1e-2):
            damping_type = "critically_damped"
            omega_d = None
            period = None
        else:
            damping_type = "overdamped"
            omega_d = None
            period = None

        return {
            "omega_0": omega_0,
            "alpha": alpha,
            "zeta": zeta,
            "damping_type": damping_type,
            "omega_d": omega_d,
            "period": period,
        }

    @staticmethod
    def rlc_underdamped(
        R: float = 10.0,
        L: float = 10e-3,
        C: float = 1e-6,
        V0: float = 10.0,
        tstop: float = None,
    ) -> CircuitDefinition:
        """
        Underdamped RLC step response (oscillatory).

        Default values give:
            ω₀ = 10,000 rad/s (f₀ ≈ 1592 Hz)
            α = 500 s⁻¹
            ζ = 0.05 (strongly underdamped)
        """
        info = RLCCircuitDefinitions.get_damping_info(R, L, C)

        if info["zeta"] >= 1.0:
            raise ValueError(f"Parameters give ζ={info['zeta']:.3f}, not underdamped")

        if tstop is None:
            # Run for at least 5 decay time constants and 10 periods
            decay_time = 5 / info["alpha"]
            oscillation_time = 10 * info["period"] if info["period"] else decay_time
            tstop = max(decay_time, oscillation_time)

        # Backward Euler has numerical damping for oscillatory systems.
        # Need more points per period for accurate oscillation capture.
        dt = info["period"] / 500 if info["period"] else 1e-6  # 500 points per period

        return CircuitDefinition(
            name="rlc_underdamped",
            description=f"RLC underdamped (ζ={info['zeta']:.3f}, f₀={info['omega_0']/(2*np.pi):.1f}Hz)",
            level=ValidationLevel.LINEAR,
            pulsim_builder=lambda: build_rlc_series_circuit(R, L, C, V0),
            spice_netlist=get_rlc_netlist(R, L, C, V0),
            tstart=0.0,
            tstop=tstop,
            dt=dt,
            compare_nodes={"V(out)": "out"},
            circuit_params={
                "V0": V0, "R": R, "L": L, "C": C,
                "V_C_initial": 0.0, "I_L_initial": 0.0,
            },
            analytical_solution=AnalyticalSolutions.rlc_series_step_response,
            pulsim_options={"use_ic": True, "dtmax": dt},  # Fixed timestep
            # Slightly relaxed tolerance for underdamped due to Backward Euler numerical damping
            max_error_tolerance=0.03,  # 3% max error (vs 0.5% default)
            rms_error_tolerance=0.015,  # 1.5% rms error (vs 0.05% default)
        )

    @staticmethod
    def rlc_critically_damped(
        L: float = 10e-3,
        C: float = 1e-6,
        V0: float = 10.0,
        tstop: float = None,
    ) -> CircuitDefinition:
        """
        Critically damped RLC step response (ζ = 1).

        R is calculated to give critical damping: R = 2*sqrt(L/C)
        """
        R = 2 * np.sqrt(L / C)
        info = RLCCircuitDefinitions.get_damping_info(R, L, C)

        if tstop is None:
            tstop = 10 / info["alpha"]  # 10 decay time constants

        dt = 1 / (info["omega_0"] * 200)  # 200 points per natural period

        return CircuitDefinition(
            name="rlc_critically_damped",
            description=f"RLC critically damped (R={R:.1f}Ω, ζ≈1)",
            level=ValidationLevel.LINEAR,
            pulsim_builder=lambda: build_rlc_series_circuit(R, L, C, V0),
            spice_netlist=get_rlc_netlist(R, L, C, V0),
            tstart=0.0,
            tstop=tstop,
            dt=dt,
            compare_nodes={"V(out)": "out"},
            circuit_params={
                "V0": V0, "R": R, "L": L, "C": C,
                "V_C_initial": 0.0, "I_L_initial": 0.0,
            },
            analytical_solution=AnalyticalSolutions.rlc_series_step_response,
            pulsim_options={"use_ic": True, "dtmax": dt},  # Fixed timestep
        )

    @staticmethod
    def rlc_overdamped(
        R: float = 1000.0,
        L: float = 10e-3,
        C: float = 1e-6,
        V0: float = 10.0,
        tstop: float = None,
    ) -> CircuitDefinition:
        """
        Overdamped RLC step response (no oscillation).

        Default values give:
            ζ = 5 (strongly overdamped)
        """
        info = RLCCircuitDefinitions.get_damping_info(R, L, C)

        if info["zeta"] <= 1.0:
            raise ValueError(f"Parameters give ζ={info['zeta']:.3f}, not overdamped")

        if tstop is None:
            tstop = 10 / info["alpha"]  # 10 decay time constants

        dt = tstop / 2000  # 2000 points total for better accuracy

        return CircuitDefinition(
            name="rlc_overdamped",
            description=f"RLC overdamped (ζ={info['zeta']:.2f})",
            level=ValidationLevel.LINEAR,
            pulsim_builder=lambda: build_rlc_series_circuit(R, L, C, V0),
            spice_netlist=get_rlc_netlist(R, L, C, V0),
            tstart=0.0,
            tstop=tstop,
            dt=dt,
            compare_nodes={"V(out)": "out"},
            circuit_params={
                "V0": V0, "R": R, "L": L, "C": C,
                "V_C_initial": 0.0, "I_L_initial": 0.0,
            },
            analytical_solution=AnalyticalSolutions.rlc_series_step_response,
            pulsim_options={"use_ic": True, "dtmax": dt},  # Fixed timestep
        )


# =============================================================================
# Pytest Test Classes
# =============================================================================

class TestRLCUnderdamped:
    """Validate underdamped RLC response against analytical solution."""

    @pytest.fixture
    def circuit_def(self):
        return RLCCircuitDefinitions.rlc_underdamped()

    def test_rlc_underdamped_analytical(self, circuit_def):
        """Test underdamped RLC against analytical solution."""
        test = ValidationTest(circuit_def)
        results = test.validate(use_analytical=True)

        assert len(results) == 1
        result = results[0]

        print(f"\n{result}")

        assert result.passed, (
            f"RLC underdamped failed:\n"
            f"  Max error: {result.max_error:.2e} (threshold: {result.max_error_threshold:.2e})\n"
            f"  RMS error: {result.rms_error:.2e} (threshold: {result.rms_error_threshold:.2e})"
        )

    def test_rlc_underdamped_oscillation_frequency(self, circuit_def):
        """Verify oscillation frequency matches ω_d."""
        circuit = circuit_def.pulsim_builder()
        opts = sl.SimulationOptions()
        opts.tstart = circuit_def.tstart
        opts.tstop = circuit_def.tstop
        opts.dt = circuit_def.dt
        opts.use_ic = True

        result = sl.Simulator(circuit, opts).run_transient()
        assert result.final_status == sl.SolverStatus.Success

        time = np.array(result.time)
        v_out = np.array(result.node_voltages["V(out)"])
        V0 = circuit_def.circuit_params["V0"]

        # Find zero crossings (relative to final value V0)
        v_relative = v_out - V0
        crossings = []
        for i in range(1, len(v_relative)):
            if v_relative[i-1] * v_relative[i] < 0:
                # Linear interpolation for more precise crossing time
                t_cross = time[i-1] + (time[i] - time[i-1]) * abs(v_relative[i-1]) / abs(v_relative[i] - v_relative[i-1])
                crossings.append(t_cross)

        if len(crossings) >= 3:
            # Period is twice the half-period (between successive crossings)
            periods = []
            for i in range(2, len(crossings)):
                periods.append(crossings[i] - crossings[i-2])

            measured_period = np.mean(periods)
            measured_freq = 1.0 / measured_period

            # Expected damped frequency
            info = RLCCircuitDefinitions.get_damping_info(
                circuit_def.circuit_params["R"],
                circuit_def.circuit_params["L"],
                circuit_def.circuit_params["C"],
            )
            expected_freq = info["omega_d"] / (2 * np.pi)

            relative_error = abs(measured_freq - expected_freq) / expected_freq
            print(f"\nMeasured freq: {measured_freq:.2f}Hz, Expected: {expected_freq:.2f}Hz")
            print(f"Error: {relative_error:.4%}")

            assert relative_error < 0.05, f"Frequency error: {relative_error:.2%}"

    def test_rlc_underdamped_overshoot(self, circuit_def):
        """Verify overshoot matches expected value for underdamped system."""
        circuit = circuit_def.pulsim_builder()
        opts = sl.SimulationOptions()
        opts.tstart = circuit_def.tstart
        opts.tstop = circuit_def.tstop
        opts.dt = circuit_def.dt
        opts.use_ic = True

        result = sl.Simulator(circuit, opts).run_transient()
        assert result.final_status == sl.SolverStatus.Success

        v_out = np.array(result.node_voltages["V(out)"])
        V0 = circuit_def.circuit_params["V0"]

        # Find first peak (overshoot)
        max_voltage = np.max(v_out)
        overshoot = (max_voltage - V0) / V0

        # Expected overshoot: exp(-π*ζ / sqrt(1-ζ²))
        info = RLCCircuitDefinitions.get_damping_info(
            circuit_def.circuit_params["R"],
            circuit_def.circuit_params["L"],
            circuit_def.circuit_params["C"],
        )
        zeta = info["zeta"]
        expected_overshoot = np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2))

        relative_error = abs(overshoot - expected_overshoot) / expected_overshoot
        print(f"\nMeasured overshoot: {overshoot:.4f} ({overshoot*100:.1f}%)")
        print(f"Expected overshoot: {expected_overshoot:.4f} ({expected_overshoot*100:.1f}%)")
        print(f"Error: {relative_error:.4%}")

        assert relative_error < 0.1, f"Overshoot error: {relative_error:.2%}"


class TestRLCCriticallyDamped:
    """Validate critically damped RLC response against analytical solution."""

    @pytest.fixture
    def circuit_def(self):
        return RLCCircuitDefinitions.rlc_critically_damped()

    def test_rlc_critically_damped_analytical(self, circuit_def):
        """Test critically damped RLC against analytical solution."""
        test = ValidationTest(circuit_def)
        results = test.validate(use_analytical=True)

        assert len(results) == 1
        result = results[0]

        print(f"\n{result}")

        assert result.passed, (
            f"RLC critically damped failed:\n"
            f"  Max error: {result.max_error:.2e} (threshold: {result.max_error_threshold:.2e})\n"
            f"  RMS error: {result.rms_error:.2e} (threshold: {result.rms_error_threshold:.2e})"
        )

    def test_rlc_critically_damped_no_overshoot(self, circuit_def):
        """Verify no overshoot for critically damped system."""
        circuit = circuit_def.pulsim_builder()
        opts = sl.SimulationOptions()
        opts.tstart = circuit_def.tstart
        opts.tstop = circuit_def.tstop
        opts.dt = circuit_def.dt
        opts.use_ic = True

        result = sl.Simulator(circuit, opts).run_transient()
        assert result.final_status == sl.SolverStatus.Success

        v_out = np.array(result.node_voltages["V(out)"])
        V0 = circuit_def.circuit_params["V0"]

        max_voltage = np.max(v_out)
        overshoot = (max_voltage - V0) / V0

        print(f"\nMax voltage: {max_voltage:.4f}V, V0: {V0:.4f}V")
        print(f"Overshoot: {overshoot*100:.2f}%")

        # Critically damped should have minimal overshoot (< 2% typically)
        assert overshoot < 0.02, f"Excessive overshoot for critically damped: {overshoot*100:.2f}%"


class TestRLCOverdamped:
    """Validate overdamped RLC response against analytical solution."""

    @pytest.fixture
    def circuit_def(self):
        return RLCCircuitDefinitions.rlc_overdamped()

    def test_rlc_overdamped_analytical(self, circuit_def):
        """Test overdamped RLC against analytical solution."""
        test = ValidationTest(circuit_def)
        results = test.validate(use_analytical=True)

        assert len(results) == 1
        result = results[0]

        print(f"\n{result}")

        assert result.passed, (
            f"RLC overdamped failed:\n"
            f"  Max error: {result.max_error:.2e} (threshold: {result.max_error_threshold:.2e})\n"
            f"  RMS error: {result.rms_error:.2e} (threshold: {result.rms_error_threshold:.2e})"
        )

    def test_rlc_overdamped_no_oscillation(self, circuit_def):
        """Verify no oscillation for overdamped system."""
        circuit = circuit_def.pulsim_builder()
        opts = sl.SimulationOptions()
        opts.tstart = circuit_def.tstart
        opts.tstop = circuit_def.tstop
        opts.dt = circuit_def.dt
        opts.use_ic = True

        result = sl.Simulator(circuit, opts).run_transient()
        assert result.final_status == sl.SolverStatus.Success

        v_out = np.array(result.node_voltages["V(out)"])
        V0 = circuit_def.circuit_params["V0"]

        # Check that voltage monotonically approaches V0 (no oscillation)
        # Allow small numerical noise
        increasing = True
        for i in range(1, len(v_out)):
            if v_out[i] < v_out[i-1] - V0 * 1e-4:  # Allow 0.01% noise
                increasing = False
                break

        print(f"\nVoltage monotonically increasing: {increasing}")
        print(f"Final voltage: {v_out[-1]:.4f}V (expected {V0:.4f}V)")

        assert increasing, "Overdamped response should not oscillate"


class TestRLCDampingTransitions:
    """Test behavior across damping regimes."""

    @pytest.mark.parametrize("zeta_target,L,C", [
        (0.1, 10e-3, 1e-6),    # Underdamped
        (0.5, 10e-3, 1e-6),    # Underdamped
        (1.0, 10e-3, 1e-6),    # Critical
        (2.0, 10e-3, 1e-6),    # Overdamped
        (5.0, 10e-3, 1e-6),    # Heavily overdamped
    ])
    def test_rlc_various_damping(self, zeta_target, L, C):
        """Test RLC response at various damping ratios."""
        # Calculate R for target damping ratio
        omega_0 = 1.0 / np.sqrt(L * C)
        R = 2 * zeta_target * L * omega_0

        info = RLCCircuitDefinitions.get_damping_info(R, L, C)
        print(f"\nTarget ζ={zeta_target}, Actual ζ={info['zeta']:.3f}, Type: {info['damping_type']}")

        # Build circuit
        circuit = build_rlc_series_circuit(R, L, C, V0=10.0)
        opts = sl.SimulationOptions()
        opts.tstart = 0.0
        opts.tstop = 10 / info["alpha"] if info["alpha"] > 0 else 1e-3
        opts.dt = opts.tstop / 1000
        opts.use_ic = True

        result = sl.Simulator(circuit, opts).run_transient()
        assert result.final_status == sl.SolverStatus.Success

        time = np.array(result.time)
        v_out = np.array(result.node_voltages["V(out)"])

        # Calculate analytical solution
        analytical = AnalyticalSolutions.rlc_series_step_response(
            time, "v_cap",
            {"V0": 10.0, "R": R, "L": L, "C": C, "V_C_initial": 0.0, "I_L_initial": 0.0}
        )

        # Compare
        comparator = ResultComparator()
        metrics = comparator.compare(time, v_out, time, analytical)

        print(f"Max error: {metrics['max_error']:.2e}")
        print(f"RMS error: {metrics['rms_error']:.2e}")
        print(f"Correlation: {metrics['correlation']:.6f}")

        # Use LINEAR level tolerances
        assert metrics["max_error"] < 1e-3, f"Max error too high: {metrics['max_error']:.2e}"
        assert metrics["rms_error"] < 1e-4, f"RMS error too high: {metrics['rms_error']:.2e}"


# =============================================================================
# Validation Report Generation
# =============================================================================

def run_rlc_validation_suite(verbose: bool = True):
    """
    Run complete RLC validation suite.

    Returns:
        List of ValidationResult objects
    """
    circuits = [
        RLCCircuitDefinitions.rlc_underdamped(R=10.0, L=10e-3, C=1e-6),
        RLCCircuitDefinitions.rlc_underdamped(R=50.0, L=10e-3, C=1e-6),
        RLCCircuitDefinitions.rlc_critically_damped(L=10e-3, C=1e-6),
        RLCCircuitDefinitions.rlc_critically_damped(L=1e-3, C=10e-6),
        RLCCircuitDefinitions.rlc_overdamped(R=500.0, L=10e-3, C=1e-6),
        RLCCircuitDefinitions.rlc_overdamped(R=1000.0, L=10e-3, C=1e-6),
    ]

    all_results = []
    for circuit_def in circuits:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing: {circuit_def.name}")
            print(f"Description: {circuit_def.description}")
            print('='*60)

        test = ValidationTest(circuit_def)
        try:
            results = test.validate(use_analytical=True)
            all_results.extend(results)

            if verbose:
                for r in results:
                    print(r)
        except Exception as e:
            print(f"ERROR: {e}")

    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        passed = sum(1 for r in all_results if r.passed)
        total = len(all_results)
        print(f"Passed: {passed}/{total}")

        if passed < total:
            print("\nFailed tests:")
            for r in all_results:
                if not r.passed:
                    print(f"  - {r.test_name}: max_err={r.max_error:.2e}, rms_err={r.rms_error:.2e}")

    return all_results


if __name__ == "__main__":
    run_rlc_validation_suite()
