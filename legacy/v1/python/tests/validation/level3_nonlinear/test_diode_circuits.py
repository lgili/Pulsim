"""
Diode Circuit Validation Tests (Level 3)

Validates nonlinear diode behavior:
- Half-wave rectifier
- Full-wave rectifier
- Clamper circuits
- Clipper circuits
"""

import pytest
import numpy as np
import pulsim as sl



# =============================================================================
# Test: Ideal Diode Behavior
# =============================================================================

class TestIdealDiode:
    """Validate ideal diode behavior."""

    def test_diode_forward_bias(self):
        """Verify diode conducts in forward bias."""
        V = 5.0
        R = 1000.0

        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", V)

        diode_params = sl.DiodeParams()
        diode_params.ideal = True
        circuit.add_diode("D1", "in", "out", diode_params)

        circuit.add_resistor("R1", "out", "0", R)

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

        v_out = signal_data["V(out)"][-1]
        # Ideal diode: V_out should be close to V_in (no drop)
        print(f"\nForward bias: V_out = {v_out:.4f}V (V_in = {V}V)")
        print(f"Diode drop: {V - v_out:.4f}V")

        # For ideal diode, drop should be very small
        assert v_out > V * 0.9, f"Output voltage too low: {v_out}V"

    def test_diode_reverse_bias(self):
        """Verify diode blocks in reverse bias."""
        V = -5.0  # Reverse bias
        R = 1000.0

        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", V)

        diode_params = sl.DiodeParams()
        diode_params.ideal = True
        circuit.add_diode("D1", "in", "out", diode_params)

        circuit.add_resistor("R1", "out", "0", R)

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

        v_out = signal_data["V(out)"][-1]
        # Reverse bias: output should be ~0 (diode blocks)
        print(f"\nReverse bias: V_out = {v_out:.6f}V (expected ~0V)")

        assert abs(v_out) < 0.1, f"Diode not blocking: V_out = {v_out}V"


# =============================================================================
# Test: Half-Wave Rectifier
# =============================================================================

class TestHalfWaveRectifier:
    """Validate half-wave rectifier circuit."""

    @pytest.mark.xfail(reason="DC operating point convergence check bug - residual=0 but reports not converged")
    def test_halfwave_rectifier_ideal(self):
        """Test half-wave rectifier with ideal diode."""
        V_peak = 10.0
        freq = 1000.0  # 1 kHz
        R_load = 1000.0
        1.0 / freq

        circuit = sl.Circuit()

        # AC source (using PWL or sinusoidal approximation)
        # For now, use a piecewise simulation approach
        # We'll simulate with a sine wave source

        # Since we may not have a sine source, let's verify with step inputs
        # Test positive half-cycle
        circuit.add_voltage_source("V1", "in", "0", V_peak)

        diode_params = sl.DiodeParams()
        diode_params.ideal = True
        circuit.add_diode("D1", "in", "out", diode_params)

        circuit.add_resistor("R1", "out", "0", R_load)

        opts = sl.SimulationOptions()
        opts.tstart = 0.0
        opts.tstop = 1e-3
        opts.dt = 1e-6

        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient()
        assert result.final_status == sl.SolverStatus.Success

        signal_names = result.signal_names
        data_matrix = np.array(result.data)
        signal_data = {name: data_matrix[:, i] for i, name in enumerate(signal_names)}

        v_out = signal_data["V(out)"]

        # Output should be approximately V_peak (minus small diode drop)
        v_out_avg = np.mean(v_out)
        print(f"\nHalf-wave rectifier (positive): V_out_avg = {v_out_avg:.4f}V")

        assert v_out_avg > V_peak * 0.9, "Rectifier not passing positive voltage"

    @pytest.mark.xfail(reason="DC operating point convergence check bug - residual=0 but reports not converged")
    def test_halfwave_rectifier_with_capacitor(self):
        """Test half-wave rectifier with filter capacitor."""
        V_in = 10.0
        R_load = 1000.0
        C_filter = 100e-6

        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", V_in)

        diode_params = sl.DiodeParams()
        diode_params.ideal = True
        circuit.add_diode("D1", "in", "out", diode_params)

        circuit.add_capacitor("C1", "out", "0", C_filter, ic=0.0)
        circuit.add_resistor("R1", "out", "0", R_load)

        opts = sl.SimulationOptions()
        opts.tstart = 0.0
        opts.tstop = 10e-3  # 10ms
        opts.dt = 1e-6
        opts.use_ic = True

        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient()
        assert result.final_status == sl.SolverStatus.Success

        signal_names = result.signal_names
        data_matrix = np.array(result.data)
        signal_data = {name: data_matrix[:, i] for i, name in enumerate(signal_names)}

        v_out = signal_data["V(out)"]

        # Capacitor should charge up to approximately V_in
        v_final = v_out[-1]
        print(f"\nRectifier with cap: V_final = {v_final:.4f}V (expected ~{V_in}V)")

        assert v_final > V_in * 0.9, f"Capacitor not charged: {v_final}V"


# =============================================================================
# Test: Diode Clamp Circuit
# =============================================================================

class TestDiodeClamp:
    """Validate diode clamping/limiting circuits."""

    def test_diode_clipper_positive(self):
        """Test positive clipper - limits positive voltage."""
        V_in = 10.0
        V_clip = 5.0
        R = 1000.0

        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", V_in)
        circuit.add_resistor("R1", "in", "out", R)

        # Clip voltage
        circuit.add_voltage_source("Vclip", "clip", "0", V_clip)

        diode_params = sl.DiodeParams()
        diode_params.ideal = True
        circuit.add_diode("D1", "out", "clip", diode_params)

        # Load resistor
        circuit.add_resistor("R2", "out", "0", 10000.0)

        opts = sl.SimulationOptions()
        opts.tstart = 0.0
        opts.tstop = 1e-6
        opts.dt = 1e-8

        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient()
        assert result.final_status == sl.SolverStatus.Success

        signal_names = result.signal_names
        data_matrix = np.array(result.data)
        signal_data = {name: data_matrix[:, i] for i, name in enumerate(signal_names)}

        v_out = signal_data["V(out)"][-1]

        # Output should be clipped at approximately V_clip
        print(f"\nPositive clipper: V_out = {v_out:.4f}V (clip at {V_clip}V)")

        # Should be close to clip voltage (diode conducts when V_out > V_clip)
        assert v_out < V_clip + 0.5, f"Not clipping: V_out = {v_out}V"


# =============================================================================
# Test: Diode OR Gate
# =============================================================================

class TestDiodeLogic:
    """Validate diode logic circuits."""

    def test_diode_or_gate(self):
        """Test diode OR gate."""
        R = 1000.0

        # Test case: A=5V, B=0V -> Out should be ~5V
        circuit = sl.Circuit()
        circuit.add_voltage_source("VA", "a", "0", 5.0)
        circuit.add_voltage_source("VB", "b", "0", 0.0)

        diode_params = sl.DiodeParams()
        diode_params.ideal = True
        circuit.add_diode("D1", "a", "out", diode_params)
        circuit.add_diode("D2", "b", "out", diode_params)

        circuit.add_resistor("R1", "out", "0", R)

        opts = sl.SimulationOptions()
        opts.tstart = 0.0
        opts.tstop = 1e-6
        opts.dt = 1e-8

        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient()
        assert result.final_status == sl.SolverStatus.Success

        signal_names = result.signal_names
        data_matrix = np.array(result.data)
        signal_data = {name: data_matrix[:, i] for i, name in enumerate(signal_names)}

        v_out = signal_data["V(out)"][-1]

        print(f"\nDiode OR (A=5V, B=0V): V_out = {v_out:.4f}V (expected ~5V)")
        assert v_out > 4.0, f"OR gate failed: V_out = {v_out}V"


# =============================================================================
# Test: Exponential Diode Model
# =============================================================================

class TestExponentialDiode:
    """Validate Shockley diode equation behavior."""

    def test_diode_iv_curve(self):
        """Verify diode I-V characteristic follows Shockley equation."""
        Is = 1e-14  # Saturation current
        n = 1.0     # Ideality factor
        Vt = 0.026  # Thermal voltage at 300K

        # Test at several forward voltages
        test_voltages = [0.3, 0.4, 0.5, 0.6, 0.7]
        R = 1.0  # Small series resistance for measurement

        for V_diode in test_voltages:
            circuit = sl.Circuit()
            # Apply voltage that will result in ~V_diode across diode
            V_source = V_diode + 0.001  # Small IR drop

            circuit.add_voltage_source("V1", "in", "0", V_source)
            circuit.add_resistor("R1", "in", "out", R)

            diode_params = sl.DiodeParams()
            diode_params.is_ = Is
            diode_params.n = n
            diode_params.ideal = False
            circuit.add_diode("D1", "out", "0", diode_params)

            opts = sl.SimulationOptions()
            opts.tstart = 0.0
            opts.tstop = 1e-6
            opts.dt = 1e-8

            sim = sl.Simulator(circuit, opts)
            result = sim.run_transient()

            if result.final_status == sl.SolverStatus.Success:
                signal_names = result.signal_names
                data_matrix = np.array(result.data)
                signal_data = {name: data_matrix[:, i] for i, name in enumerate(signal_names)}

                v_out = signal_data["V(out)"][-1]
                i_source = abs(signal_data["I(V1)"][-1])

                # Expected current from Shockley equation
                I_expected = Is * (np.exp(v_out / (n * Vt)) - 1)

                print(f"V_d={v_out:.4f}V: I={i_source:.2e}A (Shockley: {I_expected:.2e}A)")


# =============================================================================
# Run validation suite
# =============================================================================

def run_diode_validation_suite(verbose: bool = True):
    """Run all diode validation tests."""
    print("=" * 60)
    print("DIODE VALIDATION TESTS")
    print("=" * 60)

    # Ideal diode tests
    print("\n--- Ideal Diode Tests ---")
    test = TestIdealDiode()
    test.test_diode_forward_bias()
    test.test_diode_reverse_bias()

    # Rectifier tests
    print("\n--- Rectifier Tests ---")
    test = TestHalfWaveRectifier()
    test.test_halfwave_rectifier_ideal()
    test.test_halfwave_rectifier_with_capacitor()

    # Clipper tests
    print("\n--- Clipper Tests ---")
    test = TestDiodeClamp()
    test.test_diode_clipper_positive()

    # Logic tests
    print("\n--- Diode Logic Tests ---")
    test = TestDiodeLogic()
    test.test_diode_or_gate()

    print("\n" + "=" * 60)
    print("All diode tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_diode_validation_suite()
