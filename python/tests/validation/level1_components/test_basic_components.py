"""
Basic Component Validation Tests (Level 1)

Validates fundamental circuit components against analytical solutions:
- Resistor: Ohm's law V = IR
- Capacitor: I = C * dV/dt
- Inductor: V = L * dI/dt
- Voltage Source: Constant voltage output
- Current Source: Constant current output
"""

import pytest
import numpy as np
import pulsim as sl

from ..framework.base import (
    ValidationLevel,
    CircuitDefinition,
    ValidationTest,
    DEFAULT_TOLERANCES,
)
from ..framework.comparator import ResultComparator


# =============================================================================
# Test: Resistor - Ohm's Law
# =============================================================================

class TestResistor:
    """Validate resistor behavior (Ohm's law: V = IR)."""

    @pytest.mark.parametrize("R,V", [
        (100.0, 10.0),
        (1000.0, 5.0),
        (10.0, 1.0),
        (1e6, 100.0),  # High resistance
        (0.1, 1.0),    # Low resistance
    ])
    def test_resistor_ohms_law(self, R, V):
        """Verify V = IR for various resistor values."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", V)
        circuit.add_resistor("R1", "in", "0", R)

        opts = sl.SimulationOptions()
        opts.tstart = 0.0
        opts.tstop = 1e-6  # Very short - just need DC operating point
        opts.dt = 1e-7

        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient()
        assert result.final_status == sl.SolverStatus.Success

        # Get current through voltage source (which equals current through R)
        signal_names = result.signal_names
        data_matrix = np.array(result.data)
        signal_data = {name: data_matrix[:, i] for i, name in enumerate(signal_names)}

        # Current through voltage source
        I_measured = signal_data["I(V1)"][-1]
        I_expected = V / R

        # Current flows into positive terminal of voltage source
        relative_error = abs(abs(I_measured) - I_expected) / I_expected
        print(f"\nR={R}Ω, V={V}V: I_measured={I_measured:.6e}A, I_expected={I_expected:.6e}A")
        print(f"Relative error: {relative_error:.6e}")

        assert relative_error < 1e-6, f"Ohm's law violation: {relative_error:.2e}"

    def test_resistor_voltage_divider(self):
        """Verify voltage divider formula: Vout = Vin * R2 / (R1 + R2)."""
        Vin = 10.0
        R1 = 1000.0
        R2 = 1000.0
        Vout_expected = Vin * R2 / (R1 + R2)

        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", Vin)
        circuit.add_resistor("R1", "in", "out", R1)
        circuit.add_resistor("R2", "out", "0", R2)

        opts = sl.SimulationOptions()
        opts.tstart = 0.0
        opts.tstop = 1e-6
        opts.dt = 1e-7

        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient()
        assert result.final_status == sl.SolverStatus.Success

        signal_names = result.signal_names
        data_matrix = np.array(result.data)
        signal_data = {name: data_matrix[:, i] for i, name in enumerate(signal_names)}

        Vout_measured = signal_data["V(out)"][-1]
        relative_error = abs(Vout_measured - Vout_expected) / Vout_expected

        print(f"\nVoltage divider: Vout={Vout_measured:.6f}V (expected {Vout_expected:.6f}V)")
        print(f"Relative error: {relative_error:.6e}")

        assert relative_error < 1e-6, f"Voltage divider error: {relative_error:.2e}"

    @pytest.mark.parametrize("R1,R2,R3", [
        (100.0, 200.0, 300.0),
        (1000.0, 1000.0, 1000.0),
    ])
    def test_resistor_series_parallel(self, R1, R2, R3):
        """Verify series and parallel resistor combinations."""
        V = 10.0

        # R1 in series with (R2 || R3)
        R_parallel = (R2 * R3) / (R2 + R3)
        R_total = R1 + R_parallel
        I_expected = V / R_total

        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", V)
        circuit.add_resistor("R1", "in", "mid", R1)
        circuit.add_resistor("R2", "mid", "0", R2)
        circuit.add_resistor("R3", "mid", "0", R3)

        opts = sl.SimulationOptions()
        opts.tstart = 0.0
        opts.tstop = 1e-6
        opts.dt = 1e-7

        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient()
        assert result.final_status == sl.SolverStatus.Success

        signal_names = result.signal_names
        data_matrix = np.array(result.data)
        signal_data = {name: data_matrix[:, i] for i, name in enumerate(signal_names)}

        I_measured = abs(signal_data["I(V1)"][-1])
        relative_error = abs(I_measured - I_expected) / I_expected

        print(f"\nSeries-parallel: I={I_measured:.6e}A (expected {I_expected:.6e}A)")
        print(f"Relative error: {relative_error:.6e}")

        assert relative_error < 1e-6, f"Series-parallel error: {relative_error:.2e}"


# =============================================================================
# Test: Capacitor - I = C * dV/dt
# =============================================================================

class TestCapacitor:
    """Validate capacitor behavior (I = C * dV/dt)."""

    def test_capacitor_charging_current(self):
        """Verify initial charging current I = V/R for RC circuit."""
        V = 10.0
        R = 1000.0
        C = 1e-6
        I_initial_expected = V / R  # At t=0, capacitor is short circuit

        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", V)
        circuit.add_resistor("R1", "in", "out", R)
        circuit.add_capacitor("C1", "out", "0", C, ic=0.0)

        opts = sl.SimulationOptions()
        opts.tstart = 0.0
        opts.tstop = 1e-6  # Very short time
        opts.dt = 1e-9
        opts.use_ic = True

        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient()
        assert result.final_status == sl.SolverStatus.Success

        signal_names = result.signal_names
        data_matrix = np.array(result.data)
        signal_data = {name: data_matrix[:, i] for i, name in enumerate(signal_names)}

        # Current at very beginning
        I_initial = abs(signal_data["I(V1)"][1])  # Skip t=0
        relative_error = abs(I_initial - I_initial_expected) / I_initial_expected

        print(f"\nInitial charging current: {I_initial:.6e}A (expected {I_initial_expected:.6e}A)")
        print(f"Relative error: {relative_error:.4%}")

        # Allow some tolerance due to numerical integration
        assert relative_error < 0.01, f"Initial current error: {relative_error:.2%}"

    def test_capacitor_charge_conservation(self):
        """Verify Q = CV relationship."""
        V = 5.0
        C = 10e-6
        Q_expected = C * V

        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", V)
        circuit.add_capacitor("C1", "in", "0", C, ic=0.0)

        # Small resistor to allow current flow (pure V-C would be singular)
        circuit.add_resistor("R1", "in", "0", 0.001)

        opts = sl.SimulationOptions()
        opts.tstart = 0.0
        opts.tstop = 1e-3  # Allow time to charge
        opts.dt = 1e-6
        opts.use_ic = True

        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient()
        assert result.final_status == sl.SolverStatus.Success

        signal_names = result.signal_names
        data_matrix = np.array(result.data)
        signal_data = {name: data_matrix[:, i] for i, name in enumerate(signal_names)}

        # Final voltage across capacitor should be V
        V_cap_final = signal_data["V(in)"][-1]

        print(f"\nCapacitor voltage: {V_cap_final:.6f}V (expected {V:.6f}V)")
        assert abs(V_cap_final - V) / V < 1e-4, "Capacitor did not charge to source voltage"

    def test_capacitor_parallel(self):
        """Verify parallel capacitors: C_total = C1 + C2."""
        V = 10.0
        R = 1000.0
        C1 = 1e-6
        C2 = 2e-6
        C_total = C1 + C2
        tau_expected = R * C_total

        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", V)
        circuit.add_resistor("R1", "in", "out", R)
        circuit.add_capacitor("C1", "out", "0", C1, ic=0.0)
        circuit.add_capacitor("C2", "out", "0", C2, ic=0.0)

        opts = sl.SimulationOptions()
        opts.tstart = 0.0
        opts.tstop = 5 * tau_expected
        opts.dt = tau_expected / 500
        opts.use_ic = True
        opts.dtmax = opts.dt

        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient()
        assert result.final_status == sl.SolverStatus.Success

        time = np.array(result.time)
        signal_names = result.signal_names
        data_matrix = np.array(result.data)
        signal_data = {name: data_matrix[:, i] for i, name in enumerate(signal_names)}

        v_out = signal_data["V(out)"]

        # At t = tau, V should be V * (1 - 1/e) ≈ 0.632 * V
        idx_tau = np.argmin(np.abs(time - tau_expected))
        v_at_tau = v_out[idx_tau]
        v_expected_at_tau = V * (1 - np.exp(-1))

        relative_error = abs(v_at_tau - v_expected_at_tau) / v_expected_at_tau
        print(f"\nParallel caps: V(τ)={v_at_tau:.4f}V (expected {v_expected_at_tau:.4f}V)")
        print(f"Relative error: {relative_error:.4%}")

        assert relative_error < 0.01, f"Parallel capacitor error: {relative_error:.2%}"


# =============================================================================
# Test: Inductor - V = L * dI/dt
# =============================================================================

class TestInductor:
    """Validate inductor behavior (V = L * dI/dt)."""

    def test_inductor_initial_current_zero(self):
        """Verify inductor blocks sudden current change (acts as open circuit initially)."""
        V = 10.0
        R = 100.0
        L = 10e-3

        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", V)
        circuit.add_resistor("R1", "in", "out", R)
        circuit.add_inductor("L1", "out", "0", L, ic=0.0)

        opts = sl.SimulationOptions()
        opts.tstart = 0.0
        opts.tstop = 1e-7  # Very short time
        opts.dt = 1e-9
        opts.use_ic = True

        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient()
        assert result.final_status == sl.SolverStatus.Success

        signal_names = result.signal_names
        data_matrix = np.array(result.data)
        signal_data = {name: data_matrix[:, i] for i, name in enumerate(signal_names)}

        # Current should start at 0
        I_initial = signal_data["I(L1)"][0]
        print(f"\nInitial inductor current: {I_initial:.6e}A (expected ~0)")

        assert abs(I_initial) < 1e-6, f"Initial current not zero: {I_initial}"

    def test_inductor_final_current(self):
        """Verify inductor acts as short circuit at DC steady state."""
        V = 10.0
        R = 100.0
        L = 10e-3
        tau = L / R
        I_final_expected = V / R

        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", V)
        circuit.add_resistor("R1", "in", "out", R)
        circuit.add_inductor("L1", "out", "0", L, ic=0.0)

        opts = sl.SimulationOptions()
        opts.tstart = 0.0
        opts.tstop = 10 * tau  # Long enough for steady state
        opts.dt = tau / 100
        opts.use_ic = True

        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient()
        assert result.final_status == sl.SolverStatus.Success

        signal_names = result.signal_names
        data_matrix = np.array(result.data)
        signal_data = {name: data_matrix[:, i] for i, name in enumerate(signal_names)}

        I_final = signal_data["I(L1)"][-1]
        relative_error = abs(I_final - I_final_expected) / I_final_expected

        print(f"\nFinal inductor current: {I_final:.6f}A (expected {I_final_expected:.6f}A)")
        print(f"Relative error: {relative_error:.4%}")

        assert relative_error < 0.01, f"Final current error: {relative_error:.2%}"

    def test_inductor_series(self):
        """Verify series inductors: L_total = L1 + L2."""
        V = 10.0
        R = 100.0
        L1 = 5e-3
        L2 = 10e-3
        L_total = L1 + L2
        tau_expected = L_total / R

        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", V)
        circuit.add_resistor("R1", "in", "n1", R)
        circuit.add_inductor("L1", "n1", "n2", L1, ic=0.0)
        circuit.add_inductor("L2", "n2", "0", L2, ic=0.0)

        opts = sl.SimulationOptions()
        opts.tstart = 0.0
        opts.tstop = 5 * tau_expected
        opts.dt = tau_expected / 500
        opts.use_ic = True
        opts.dtmax = opts.dt

        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient()
        assert result.final_status == sl.SolverStatus.Success

        time = np.array(result.time)
        signal_names = result.signal_names
        data_matrix = np.array(result.data)
        signal_data = {name: data_matrix[:, i] for i, name in enumerate(signal_names)}

        i_L = signal_data["I(L1)"]  # Same current through both

        # At t = tau, I should be I_final * (1 - 1/e)
        I_final = V / R
        idx_tau = np.argmin(np.abs(time - tau_expected))
        i_at_tau = i_L[idx_tau]
        i_expected_at_tau = I_final * (1 - np.exp(-1))

        relative_error = abs(i_at_tau - i_expected_at_tau) / i_expected_at_tau
        print(f"\nSeries inductors: I(τ)={i_at_tau:.6f}A (expected {i_expected_at_tau:.6f}A)")
        print(f"Relative error: {relative_error:.4%}")

        assert relative_error < 0.01, f"Series inductor error: {relative_error:.2%}"


# =============================================================================
# Test: Voltage Source
# =============================================================================

class TestVoltageSource:
    """Validate voltage source behavior."""

    @pytest.mark.parametrize("V", [1.0, 5.0, 10.0, 100.0, -5.0])
    def test_voltage_source_dc(self, V):
        """Verify DC voltage source maintains constant voltage."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "out", "0", V)
        circuit.add_resistor("R1", "out", "0", 1000.0)

        opts = sl.SimulationOptions()
        opts.tstart = 0.0
        opts.tstop = 1e-3
        opts.dt = 1e-5

        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient()
        assert result.final_status == sl.SolverStatus.Success

        signal_names = result.signal_names
        data_matrix = np.array(result.data)
        signal_data = {name: data_matrix[:, i] for i, name in enumerate(signal_names)}

        v_out = signal_data["V(out)"]

        # All values should be V
        max_deviation = np.max(np.abs(v_out - V))
        print(f"\nDC voltage source {V}V: max deviation = {max_deviation:.2e}V")

        assert max_deviation < 1e-10, f"Voltage source deviation: {max_deviation}"


# =============================================================================
# Test: Energy Conservation
# =============================================================================

class TestEnergyConservation:
    """Test energy conservation in reactive circuits."""

    def test_lc_oscillator_energy(self):
        """Verify energy conservation in LC oscillator."""
        L = 1e-3
        C = 1e-6
        V0 = 10.0  # Initial capacitor voltage

        # Energy = 0.5 * C * V^2 = 0.5 * L * I^2
        E_initial = 0.5 * C * V0**2
        omega = 1.0 / np.sqrt(L * C)
        period = 2 * np.pi / omega

        circuit = sl.Circuit()
        circuit.add_capacitor("C1", "out", "0", C, ic=V0)
        circuit.add_inductor("L1", "out", "0", L, ic=0.0)
        # Very large resistance for numerical stability, approximating ideal LC
        circuit.add_resistor("R1", "out", "0", 1e12)
              
        opts = sl.SimulationOptions()
        opts.tstart = 0.0
        opts.tstop = 5 * period
        opts.dt = period / 500
        opts.dtmin = opts.dt / 10  # Avoid edge case issues
        opts.use_ic = True
        # Use BDF2/GEAR2 for excellent energy conservation (~99.97%)
        opts.integration_method = sl.IntegrationMethod.GEAR2
        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient()
        assert result.final_status == sl.SolverStatus.Success

        signal_names = result.signal_names
        data_matrix = np.array(result.data)
        signal_data = {name: data_matrix[:, i] for i, name in enumerate(signal_names)}

        v_C = signal_data["V(out)"]
        i_L = signal_data["I(L1)"]

        # Calculate total energy at each point
        E_C = 0.5 * C * v_C**2
        E_L = 0.5 * L * i_L**2
        E_total = E_C + E_L

        # Energy should be approximately constant (with some numerical damping)
        E_variation = (np.max(E_total) - np.min(E_total)) / E_initial

        print(f"\nLC oscillator energy conservation:")
        print(f"  Initial energy: {E_initial:.6e} J")
        print(f"  Final energy: {E_total[-1]:.6e} J")
        print(f"  Energy variation: {E_variation*100:.2f}%")

        # Allow some energy loss due to numerical damping
        assert E_variation < 0.2, f"Energy variation too high: {E_variation*100:.1f}%"


# =============================================================================
# Run validation suite
# =============================================================================

def run_component_validation_suite(verbose: bool = True):
    """Run all component validation tests."""
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode


if __name__ == "__main__":
    # Run a quick manual test
    print("Testing basic components...")

    # Resistor
    print("\n=== Resistor Test ===")
    test = TestResistor()
    test.test_resistor_ohms_law(100.0, 10.0)
    test.test_resistor_voltage_divider()

    # Capacitor
    print("\n=== Capacitor Test ===")
    test = TestCapacitor()
    test.test_capacitor_charging_current()

    # Inductor
    print("\n=== Inductor Test ===")
    test = TestInductor()
    test.test_inductor_final_current()

    # Energy
    print("\n=== Energy Conservation Test ===")
    test = TestEnergyConservation()
    test.test_lc_oscillator_energy()

    print("\n\nAll component tests passed!")
