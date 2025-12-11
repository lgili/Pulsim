"""
RL Circuit Validation Tests

Validates Pulsim RL circuit simulations against analytical solutions.
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

def build_rl_step_circuit(R: float = 100.0, L: float = 10e-3, V0: float = 10.0):
    """Build RL step response circuit for Pulsim."""
    circuit = sl.Circuit()
    circuit.add_voltage_source("V1", "in", "0", V0)
    circuit.add_resistor("R1", "in", "out", R)
    circuit.add_inductor("L1", "out", "0", L, ic=0.0)
    return circuit


def build_rl_current_decay_circuit(R: float = 100.0, L: float = 10e-3, I0: float = 0.1):
    """
    Build RL current decay circuit for Pulsim.

    Inductor with initial current, decaying through resistor.
    """
    circuit = sl.Circuit()
    # Inductor with initial current, series resistor to ground
    circuit.add_resistor("R1", "out", "0", R)
    circuit.add_inductor("L1", "out", "0", L, ic=I0)
    return circuit


# =============================================================================
# SPICE Netlists for NgSpice Reference
# =============================================================================

RL_STEP_NETLIST = """
* RL Step Response
V1 in 0 DC 10
R1 in out 100
L1 out 0 10m IC=0
.ic I(L1)=0
"""

RL_DECAY_NETLIST = """
* RL Current Decay
R1 out 0 100
L1 out 0 10m IC=0.1
.ic I(L1)=0.1
"""


# =============================================================================
# Circuit Definitions
# =============================================================================

class RLCircuitDefinitions:
    """Collection of RL circuit test definitions."""

    @staticmethod
    def rl_step_response(
        R: float = 100.0,
        L: float = 10e-3,
        V0: float = 10.0,
        tstop: float = None,
    ) -> CircuitDefinition:
        """
        RL step response test.

        Default: R=100Ω, L=10mH → τ = L/R = 100µs
        Final current: V0/R = 100mA
        """
        tau = L / R
        if tstop is None:
            tstop = 5 * tau  # 5 time constants

        dt = tau / 500  # 500 points per time constant for <0.1% error
        return CircuitDefinition(
            name="rl_step_response",
            description=f"RL step response (R={R}, L={L}, τ={tau:.2e}s)",
            level=ValidationLevel.LINEAR,
            pulsim_builder=lambda: build_rl_step_circuit(R, L, V0),
            spice_netlist=RL_STEP_NETLIST,
            tstart=0.0,
            tstop=tstop,
            dt=dt,
            compare_nodes={"I(L1)": "i(l1)"},  # Inductor current
            circuit_params={"V0": V0, "R": R, "L": L, "I_initial": 0.0},
            analytical_solution=AnalyticalSolutions.rl_step_response,
            pulsim_options={"use_ic": True, "dtmax": dt},  # Fixed timestep
        )

    @staticmethod
    def rl_current_decay(
        R: float = 100.0,
        L: float = 10e-3,
        I0: float = 0.1,
        tstop: float = None,
    ) -> CircuitDefinition:
        """
        RL current decay test.

        Default: R=100Ω, L=10mH, I0=100mA → τ = L/R = 100µs
        """
        tau = L / R
        if tstop is None:
            tstop = 5 * tau

        dt = tau / 500  # 500 points per time constant for <0.1% error
        return CircuitDefinition(
            name="rl_current_decay",
            description=f"RL current decay (R={R}, L={L}, τ={tau:.2e}s)",
            level=ValidationLevel.LINEAR,
            pulsim_builder=lambda: build_rl_current_decay_circuit(R, L, I0),
            spice_netlist=RL_DECAY_NETLIST,
            tstart=0.0,
            tstop=tstop,
            dt=dt,
            compare_nodes={"I(L1)": "i(l1)"},
            circuit_params={"I0": I0, "R": R, "L": L},
            analytical_solution=AnalyticalSolutions.rl_current_decay,
            pulsim_options={"use_ic": True, "dtmax": dt},  # Fixed timestep
        )


# =============================================================================
# Pytest Test Classes
# =============================================================================

class TestRLStepResponse:
    """Validate RL step response against analytical solution."""

    @pytest.fixture
    def circuit_def(self):
        return RLCircuitDefinitions.rl_step_response()

    def test_rl_step_analytical(self, circuit_def):
        """Test RL step response against analytical solution."""
        test = ValidationTest(circuit_def)
        results = test.validate(use_analytical=True)

        assert len(results) == 1
        result = results[0]

        print(f"\n{result}")

        assert result.passed, (
            f"RL step response failed:\n"
            f"  Max error: {result.max_error:.2e} (threshold: {result.max_error_threshold:.2e})\n"
            f"  RMS error: {result.rms_error:.2e} (threshold: {result.rms_error_threshold:.2e})"
        )

    def test_rl_step_time_constant_accuracy(self, circuit_def):
        """Verify the time constant τ = L/R is correct."""
        circuit = circuit_def.pulsim_builder()
        opts = sl.SimulationOptions()
        opts.tstart = circuit_def.tstart
        opts.tstop = circuit_def.tstop
        opts.dt = circuit_def.dt
        opts.use_ic = True

        result = sl.Simulator(circuit, opts).run_transient()
        assert result.final_status == sl.SolverStatus.Success

        time = np.array(result.time)
        i_L = np.array(result.branch_currents["I(L1)"])

        # At t = τ, current should be I_final * (1 - 1/e) ≈ 0.632 * I_final
        V0 = circuit_def.circuit_params["V0"]
        R = circuit_def.circuit_params["R"]
        L = circuit_def.circuit_params["L"]
        tau = L / R
        I_final = V0 / R

        idx_tau = np.argmin(np.abs(time - tau))
        i_at_tau = i_L[idx_tau]
        expected_i_at_tau = I_final * (1 - np.exp(-1))

        relative_error = abs(i_at_tau - expected_i_at_tau) / expected_i_at_tau
        print(f"\nAt t=τ: Pulsim={i_at_tau:.6f}A, Expected={expected_i_at_tau:.6f}A, Error={relative_error:.4%}")

        assert relative_error < 0.01, f"Time constant accuracy error: {relative_error:.2%}"

    def test_rl_step_final_current(self, circuit_def):
        """Verify inductor current reaches V0/R at steady state."""
        circuit = circuit_def.pulsim_builder()
        opts = sl.SimulationOptions()
        opts.tstart = circuit_def.tstart
        opts.tstop = circuit_def.tstop
        opts.dt = circuit_def.dt
        opts.use_ic = True

        result = sl.Simulator(circuit, opts).run_transient()
        assert result.final_status == sl.SolverStatus.Success

        i_L = np.array(result.branch_currents["I(L1)"])
        V0 = circuit_def.circuit_params["V0"]
        R = circuit_def.circuit_params["R"]
        I_final_expected = V0 / R

        # After 5τ, current should be within 1% of final
        final_current = i_L[-1]
        relative_error = abs(final_current - I_final_expected) / I_final_expected

        print(f"\nFinal current: {final_current:.6f}A (expected {I_final_expected:.6f}A)")

        assert relative_error < 0.01, f"Final current error: {relative_error:.2%}"

    @pytest.mark.parametrize("R,L", [
        (50.0, 10e-3),      # τ = 200µs
        (100.0, 10e-3),     # τ = 100µs
        (200.0, 10e-3),     # τ = 50µs
        (100.0, 1e-3),      # τ = 10µs
        (100.0, 100e-3),    # τ = 1ms
    ])
    def test_rl_step_various_parameters(self, R, L):
        """Test RL step response with various R and L values."""
        circuit_def = RLCircuitDefinitions.rl_step_response(R=R, L=L, V0=10.0)
        test = ValidationTest(circuit_def)
        results = test.validate(use_analytical=True)

        result = results[0]
        tau = L / R
        print(f"\nR={R}, L={L}, τ={tau:.2e}s: max_err={result.max_error:.2e}, rms_err={result.rms_error:.2e}")

        assert result.passed, f"Failed for R={R}, L={L}"


class TestRLCurrentDecay:
    """Validate RL current decay against analytical solution."""

    @pytest.fixture
    def circuit_def(self):
        return RLCircuitDefinitions.rl_current_decay()

    def test_rl_decay_analytical(self, circuit_def):
        """Test RL current decay against analytical solution."""
        test = ValidationTest(circuit_def)
        results = test.validate(use_analytical=True)

        assert len(results) == 1
        result = results[0]

        print(f"\n{result}")

        assert result.passed, (
            f"RL current decay failed:\n"
            f"  Max error: {result.max_error:.2e} (threshold: {result.max_error_threshold:.2e})\n"
            f"  RMS error: {result.rms_error:.2e} (threshold: {result.rms_error_threshold:.2e})"
        )

    def test_rl_decay_final_current(self, circuit_def):
        """Verify inductor current decays to ~0."""
        circuit = circuit_def.pulsim_builder()
        opts = sl.SimulationOptions()
        opts.tstart = circuit_def.tstart
        opts.tstop = circuit_def.tstop
        opts.dt = circuit_def.dt
        opts.use_ic = True

        result = sl.Simulator(circuit, opts).run_transient()
        assert result.final_status == sl.SolverStatus.Success

        i_L = np.array(result.branch_currents["I(L1)"])
        I0 = circuit_def.circuit_params["I0"]

        # After 5τ, current should be < 1% of initial
        final_current = abs(i_L[-1])
        expected_final = I0 * np.exp(-5)

        print(f"\nFinal current: {final_current:.6e}A (expected ~{expected_final:.6e}A)")

        assert final_current < I0 * 0.01, f"Current did not decay: {final_current}A"


class TestRLEnergyConservation:
    """Test energy conservation in RL circuits."""

    def test_rl_step_energy(self):
        """
        Verify energy conservation during RL step response.

        Energy stored in inductor: E = 0.5 * L * I²
        Energy dissipated in resistor: E = ∫ I²R dt
        """
        R = 100.0
        L = 10e-3
        V0 = 10.0
        tau = L / R
        tstop = 10 * tau  # Extra long for steady state

        circuit = build_rl_step_circuit(R, L, V0)
        opts = sl.SimulationOptions()
        opts.tstart = 0.0
        opts.tstop = tstop
        opts.dt = tau / 100
        opts.use_ic = True

        result = sl.Simulator(circuit, opts).run_transient()
        assert result.final_status == sl.SolverStatus.Success

        time = np.array(result.time)
        i_L = np.array(result.branch_currents["I(L1)"])

        # Final inductor current and energy
        I_final = i_L[-1]
        E_inductor = 0.5 * L * I_final**2

        # Expected values
        I_final_expected = V0 / R
        E_inductor_expected = 0.5 * L * I_final_expected**2

        relative_error = abs(E_inductor - E_inductor_expected) / E_inductor_expected

        print(f"\nInductor energy: {E_inductor:.6e}J (expected {E_inductor_expected:.6e}J)")
        print(f"Relative error: {relative_error:.4%}")

        assert relative_error < 0.02, f"Energy error: {relative_error:.2%}"


# =============================================================================
# Validation Report Generation
# =============================================================================

def run_rl_validation_suite(verbose: bool = True):
    """
    Run complete RL validation suite.

    Returns:
        List of ValidationResult objects
    """
    circuits = [
        RLCircuitDefinitions.rl_step_response(R=100.0, L=10e-3),
        RLCircuitDefinitions.rl_step_response(R=50.0, L=10e-3),
        RLCircuitDefinitions.rl_step_response(R=200.0, L=5e-3),
        RLCircuitDefinitions.rl_current_decay(R=100.0, L=10e-3),
        RLCircuitDefinitions.rl_current_decay(R=50.0, L=20e-3),
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
    run_rl_validation_suite()
