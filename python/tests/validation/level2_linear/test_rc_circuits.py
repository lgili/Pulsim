"""
RC Circuit Validation Tests

Validates Pulsim RC circuit simulations against analytical solutions.
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


# =============================================================================
# Circuit Builders for Pulsim
# =============================================================================

def build_rc_step_circuit(R: float = 1000.0, C: float = 1e-6, V0: float = 5.0):
    """Build RC step response circuit for Pulsim."""
    circuit = sl.Circuit()
    circuit.add_voltage_source("V1", "in", "0", V0)
    circuit.add_resistor("R1", "in", "out", R)
    circuit.add_capacitor("C1", "out", "0", C, ic=0.0)
    return circuit


def build_rc_discharge_circuit(R: float = 1000.0, C: float = 1e-6, V0: float = 5.0):
    """Build RC discharge circuit for Pulsim (capacitor with initial voltage)."""
    circuit = sl.Circuit()
    # No voltage source - capacitor discharges through resistor
    circuit.add_resistor("R1", "out", "0", R)
    circuit.add_capacitor("C1", "out", "0", C, ic=V0)
    return circuit


# =============================================================================
# SPICE Netlists for NgSpice Reference
# =============================================================================

RC_STEP_NETLIST = """
* RC Step Response
V1 in 0 DC 5
R1 in out 1k
C1 out 0 1u IC=0
.ic V(out)=0
"""

RC_DISCHARGE_NETLIST = """
* RC Discharge
R1 out 0 1k
C1 out 0 1u IC=5
.ic V(out)=5
"""


# =============================================================================
# Circuit Definitions
# =============================================================================

class RCCircuitDefinitions:
    """Collection of RC circuit test definitions."""

    @staticmethod
    def rc_step_response(
        R: float = 1000.0,
        C: float = 1e-6,
        V0: float = 5.0,
        tstop: float = None,
    ) -> CircuitDefinition:
        """
        RC step response test.

        Default: R=1kΩ, C=1µF → τ = 1ms
        """
        tau = R * C
        if tstop is None:
            tstop = 5 * tau  # 5 time constants for ~99.3% settling

        dt = tau / 500  # 500 points per time constant for <0.1% error
        return CircuitDefinition(
            name="rc_step_response",
            description=f"RC step response (R={R}, C={C}, τ={tau:.2e}s)",
            level=ValidationLevel.LINEAR,
            pulsim_builder=lambda: build_rc_step_circuit(R, C, V0),
            spice_netlist=RC_STEP_NETLIST,
            tstart=0.0,
            tstop=tstop,
            dt=dt,
            compare_nodes={"V(out)": "out"},
            circuit_params={"V0": V0, "R": R, "C": C, "V_initial": 0.0},
            analytical_solution=AnalyticalSolutions.rc_step_response,
            pulsim_options={"use_ic": True, "dtmax": dt, "dtmin": dt / 10},  # Fixed timestep
        )

    @staticmethod
    def rc_discharge(
        R: float = 1000.0,
        C: float = 1e-6,
        V0: float = 5.0,
        tstop: float = None,
    ) -> CircuitDefinition:
        """
        RC discharge test.

        Default: R=1kΩ, C=1µF → τ = 1ms
        """
        tau = R * C
        if tstop is None:
            tstop = 5 * tau

        dt = tau / 500  # 500 points per time constant for <0.1% error
        return CircuitDefinition(
            name="rc_discharge",
            description=f"RC discharge (R={R}, C={C}, τ={tau:.2e}s)",
            level=ValidationLevel.LINEAR,
            pulsim_builder=lambda: build_rc_discharge_circuit(R, C, V0),
            spice_netlist=RC_DISCHARGE_NETLIST,
            tstart=0.0,
            tstop=tstop,
            dt=dt,
            compare_nodes={"V(out)": "out"},
            circuit_params={"V0": V0, "R": R, "C": C},
            analytical_solution=AnalyticalSolutions.rc_discharge,
            pulsim_options={"use_ic": True, "dtmax": dt, "dtmin": dt / 10},  # Fixed timestep
        )


# =============================================================================
# Pytest Test Classes
# =============================================================================

class TestRCStepResponse:
    """Validate RC step response against analytical solution."""

    @pytest.fixture
    def circuit_def(self):
        return RCCircuitDefinitions.rc_step_response()

    def test_rc_step_analytical(self, circuit_def):
        """Test RC step response against analytical solution."""
        test = ValidationTest(circuit_def)
        results = test.validate(use_analytical=True)

        assert len(results) == 1
        result = results[0]

        print(f"\n{result}")

        # Check thresholds
        assert result.passed, (
            f"RC step response failed:\n"
            f"  Max error: {result.max_error:.2e} (threshold: {result.max_error_threshold:.2e})\n"
            f"  RMS error: {result.rms_error:.2e} (threshold: {result.rms_error_threshold:.2e})"
        )

    def test_rc_step_time_constant_accuracy(self, circuit_def):
        """Verify the time constant τ = RC is correct."""
        # Run Pulsim simulation
        circuit = circuit_def.pulsim_builder()
        opts = sl.SimulationOptions()
        opts.tstart = circuit_def.tstart
        opts.tstop = circuit_def.tstop
        opts.dt = circuit_def.dt
        opts.use_ic = True

        result = sl.Simulator(circuit, opts).run_transient()
        assert result.final_status == sl.SolverStatus.Success

        time = np.array(result.time)
        signal_names = result.signal_names
        data_matrix = np.array(result.data)
        signal_data = {name: data_matrix[:, i] for i, name in enumerate(signal_names)}
        v_out = signal_data["V(out)"]

        # At t = τ, voltage should be V0 * (1 - 1/e) ≈ 0.632 * V0
        V0 = circuit_def.circuit_params["V0"]
        tau = circuit_def.circuit_params["R"] * circuit_def.circuit_params["C"]

        # Find voltage at t = τ
        idx_tau = np.argmin(np.abs(time - tau))
        v_at_tau = v_out[idx_tau]
        expected_v_at_tau = V0 * (1 - np.exp(-1))

        relative_error = abs(v_at_tau - expected_v_at_tau) / expected_v_at_tau
        print(f"\nAt t=τ: Pulsim={v_at_tau:.4f}V, Expected={expected_v_at_tau:.4f}V, Error={relative_error:.4%}")

        assert relative_error < 0.01, f"Time constant accuracy error: {relative_error:.2%}"

    @pytest.mark.parametrize("R,C", [
        (100.0, 1e-6),      # τ = 100µs
        (1000.0, 1e-6),     # τ = 1ms
        (10000.0, 1e-6),    # τ = 10ms
        (1000.0, 100e-9),   # τ = 100µs
        (1000.0, 10e-6),    # τ = 10ms
    ])
    def test_rc_step_various_parameters(self, R, C):
        """Test RC step response with various R and C values."""
        circuit_def = RCCircuitDefinitions.rc_step_response(R=R, C=C, V0=5.0)
        test = ValidationTest(circuit_def)
        results = test.validate(use_analytical=True)

        result = results[0]
        tau = R * C
        print(f"\nR={R}, C={C}, τ={tau:.2e}s: max_err={result.max_error:.2e}, rms_err={result.rms_error:.2e}")

        assert result.passed, f"Failed for R={R}, C={C}"


class TestRCDischarge:
    """Validate RC discharge against analytical solution."""

    @pytest.fixture
    def circuit_def(self):
        return RCCircuitDefinitions.rc_discharge()

    def test_rc_discharge_analytical(self, circuit_def):
        """Test RC discharge against analytical solution."""
        test = ValidationTest(circuit_def)
        results = test.validate(use_analytical=True)

        assert len(results) == 1
        result = results[0]

        print(f"\n{result}")

        assert result.passed, (
            f"RC discharge failed:\n"
            f"  Max error: {result.max_error:.2e} (threshold: {result.max_error_threshold:.2e})\n"
            f"  RMS error: {result.rms_error:.2e} (threshold: {result.rms_error_threshold:.2e})"
        )

    def test_rc_discharge_final_value(self, circuit_def):
        """Verify capacitor fully discharges to ~0V."""
        circuit = circuit_def.pulsim_builder()
        opts = sl.SimulationOptions()
        opts.tstart = circuit_def.tstart
        opts.tstop = circuit_def.tstop
        opts.dt = circuit_def.dt
        opts.use_ic = True

        result = sl.Simulator(circuit, opts).run_transient()
        assert result.final_status == sl.SolverStatus.Success

        signal_names = result.signal_names
        data_matrix = np.array(result.data)
        signal_data = {name: data_matrix[:, i] for i, name in enumerate(signal_names)}
        v_out = signal_data["V(out)"]
        V0 = circuit_def.circuit_params["V0"]

        # After 5τ, voltage should be < 1% of initial
        final_voltage = v_out[-1]
        expected_final = V0 * np.exp(-5)  # e^-5 ≈ 0.67%

        print(f"\nFinal voltage: {final_voltage:.4f}V (expected ~{expected_final:.4f}V)")

        assert abs(final_voltage) < V0 * 0.01, f"Capacitor did not discharge: {final_voltage}V"


class TestRCFrequencyResponse:
    """Validate RC filter frequency response."""

    def test_rc_lowpass_cutoff(self):
        """Test RC lowpass filter at cutoff frequency."""
        R = 1000.0  # 1kΩ
        C = 1e-6    # 1µF
        1.0 / (2 * np.pi * R * C)  # ~159 Hz

        # Test at cutoff frequency - should attenuate to 1/√2 = 0.707
        # TODO: Implement sinusoidal source test when source types are available

    @pytest.mark.parametrize("freq_ratio", [0.1, 0.5, 1.0, 2.0, 10.0])
    def test_rc_lowpass_attenuation(self, freq_ratio):
        """Test RC lowpass filter attenuation at various frequencies."""
        R = 1000.0
        C = 1e-6
        f_cutoff = 1.0 / (2 * np.pi * R * C)

        f_cutoff * freq_ratio

        # Expected attenuation: 1 / sqrt(1 + (f/fc)^2)
        expected_gain = 1.0 / np.sqrt(1 + freq_ratio**2)

        print(f"\nf/fc={freq_ratio}: expected gain = {expected_gain:.4f} ({20*np.log10(expected_gain):.2f} dB)")

        # TODO: Implement with sinusoidal source


# =============================================================================
# Validation Report Generation
# =============================================================================

def run_rc_validation_suite(verbose: bool = True):
    """
    Run complete RC validation suite.

    Returns:
        List of ValidationResult objects
    """
    circuits = [
        RCCircuitDefinitions.rc_step_response(R=1000.0, C=1e-6),
        RCCircuitDefinitions.rc_step_response(R=100.0, C=1e-6),
        RCCircuitDefinitions.rc_step_response(R=10000.0, C=100e-9),
        RCCircuitDefinitions.rc_discharge(R=1000.0, C=1e-6),
        RCCircuitDefinitions.rc_discharge(R=500.0, C=2e-6),
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
    run_rc_validation_suite()
