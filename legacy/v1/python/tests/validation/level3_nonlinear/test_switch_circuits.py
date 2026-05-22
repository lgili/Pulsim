"""
Switch Circuit Validation Tests (Level 3)

Validates controlled switch behavior:
- Voltage-controlled switch
- On/off resistance
- Switching transients
"""

import pytest
import numpy as np
import pulsim as sl



# =============================================================================
# Test: Basic Switch Behavior
# =============================================================================

class TestBasicSwitch:
    """Validate basic switch on/off behavior."""

    def test_switch_on_state(self):
        """Verify switch conducts when control voltage > threshold."""
        V_in = 10.0
        R_load = 1000.0
        V_ctrl = 5.0  # Above threshold
        R_on = 0.01

        circuit = sl.Circuit()
        circuit.add_voltage_source("Vdc", "in", "0", V_in)
        circuit.add_voltage_source("Vctrl", "ctrl", "0", V_ctrl)

        sw_params = sl.SwitchParams()
        sw_params.ron = R_on
        sw_params.roff = 1e9
        sw_params.vth = 2.5
        circuit.add_switch("S1", "in", "out", "ctrl", "0", sw_params)

        circuit.add_resistor("R1", "out", "0", R_load)

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

        # With switch ON, V_out ≈ V_in * R_load / (R_on + R_load)
        V_expected = V_in * R_load / (R_on + R_load)
        relative_error = abs(v_out - V_expected) / V_expected

        print(f"\nSwitch ON: V_out = {v_out:.4f}V (expected {V_expected:.4f}V)")
        print(f"Relative error: {relative_error:.4%}")

        assert relative_error < 0.01, f"Switch ON error: {relative_error:.2%}"

    def test_switch_off_state(self):
        """Verify switch blocks when control voltage < threshold."""
        V_in = 10.0
        R_load = 1000.0
        V_ctrl = 0.0  # Below threshold
        R_off = 1e9

        circuit = sl.Circuit()
        circuit.add_voltage_source("Vdc", "in", "0", V_in)
        circuit.add_voltage_source("Vctrl", "ctrl", "0", V_ctrl)

        sw_params = sl.SwitchParams()
        sw_params.ron = 0.01
        sw_params.roff = R_off
        sw_params.vth = 2.5
        circuit.add_switch("S1", "in", "out", "ctrl", "0", sw_params)

        circuit.add_resistor("R1", "out", "0", R_load)

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

        # With switch OFF, V_out ≈ V_in * R_load / (R_off + R_load) ≈ 0
        print(f"\nSwitch OFF: V_out = {v_out:.6f}V (expected ~0V)")

        assert v_out < 0.01, f"Switch not blocking: V_out = {v_out}V"

    def test_switch_threshold(self):
        """Verify switch transitions at threshold voltage."""
        V_in = 10.0
        R_load = 1000.0
        V_th = 2.5

        # Test just below threshold
        circuit = sl.Circuit()
        circuit.add_voltage_source("Vdc", "in", "0", V_in)
        circuit.add_voltage_source("Vctrl", "ctrl", "0", V_th - 0.1)

        sw_params = sl.SwitchParams()
        sw_params.ron = 0.01
        sw_params.roff = 1e9
        sw_params.vth = V_th
        circuit.add_switch("S1", "in", "out", "ctrl", "0", sw_params)
        circuit.add_resistor("R1", "out", "0", R_load)

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

        v_out_below = signal_data["V(out)"][-1]

        # Test just above threshold
        circuit2 = sl.Circuit()
        circuit2.add_voltage_source("Vdc", "in", "0", V_in)
        circuit2.add_voltage_source("Vctrl", "ctrl", "0", V_th + 0.1)
        circuit2.add_switch("S1", "in", "out", "ctrl", "0", sw_params)
        circuit2.add_resistor("R1", "out", "0", R_load)

        sim2 = sl.Simulator(circuit2, opts)
        result2 = sim2.run_transient()
        assert result2.final_status == sl.SolverStatus.Success

        signal_names2 = result2.signal_names
        data_matrix2 = np.array(result2.data)
        signal_data2 = {name: data_matrix2[:, i] for i, name in enumerate(signal_names2)}

        v_out_above = signal_data2["V(out)"][-1]

        print("\nThreshold test:")
        print(f"  V_ctrl = {V_th - 0.1}V (below): V_out = {v_out_below:.4f}V")
        print(f"  V_ctrl = {V_th + 0.1}V (above): V_out = {v_out_above:.4f}V")

        # Below threshold should be low, above should be high
        assert v_out_below < 1.0, f"Below threshold not off: {v_out_below}V"
        assert v_out_above > V_in * 0.9, f"Above threshold not on: {v_out_above}V"


# =============================================================================
# Test: Switch with Inductive Load
# =============================================================================

class TestSwitchInductiveLoad:
    """Test switch behavior with inductive loads."""

    def test_switch_rl_circuit(self):
        """Test switch driving RL load."""
        V_in = 10.0
        R = 100.0
        L = 1e-3
        tau = L / R

        circuit = sl.Circuit()
        circuit.add_voltage_source("Vdc", "in", "0", V_in)
        circuit.add_voltage_source("Vctrl", "ctrl", "0", 5.0)  # ON

        sw_params = sl.SwitchParams()
        sw_params.ron = 0.01
        sw_params.roff = 1e9
        sw_params.vth = 2.5
        circuit.add_switch("S1", "in", "sw", "ctrl", "0", sw_params)

        circuit.add_resistor("R1", "sw", "out", R)
        circuit.add_inductor("L1", "out", "0", L, ic=0.0)

        opts = sl.SimulationOptions()
        opts.tstart = 0.0
        opts.tstop = 5 * tau
        opts.dt = tau / 100
        opts.use_ic = True

        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient()
        assert result.final_status == sl.SolverStatus.Success

        signal_names = result.signal_names
        data_matrix = np.array(result.data)
        signal_data = {name: data_matrix[:, i] for i, name in enumerate(signal_names)}

        np.array(result.time)
        i_L = signal_data["I(L1)"]

        # Final current should be V_in / R
        I_final_expected = V_in / R
        I_final = i_L[-1]
        relative_error = abs(I_final - I_final_expected) / I_final_expected

        print("\nSwitch + RL:")
        print(f"  I_final = {I_final:.4f}A (expected {I_final_expected:.4f}A)")
        print(f"  Relative error: {relative_error:.4%}")

        assert relative_error < 0.02, f"Final current error: {relative_error:.2%}"


# =============================================================================
# Test: Half-Bridge Configuration
# =============================================================================

class TestHalfBridge:
    """Test half-bridge switch configuration."""

    def test_halfbridge_high_side_on(self):
        """Test half-bridge with high-side switch on."""
        V_dc = 48.0
        R_load = 10.0

        circuit = sl.Circuit()
        circuit.add_voltage_source("Vdc", "vcc", "0", V_dc)

        # High-side control ON, low-side control OFF
        circuit.add_voltage_source("Vhi", "ctrl_hi", "0", 5.0)
        circuit.add_voltage_source("Vlo", "ctrl_lo", "0", 0.0)

        sw_params = sl.SwitchParams()
        sw_params.ron = 0.01
        sw_params.roff = 1e9
        sw_params.vth = 2.5

        # High-side switch
        circuit.add_switch("S_hi", "vcc", "out", "ctrl_hi", "0", sw_params)

        # Low-side switch
        circuit.add_switch("S_lo", "out", "0", "ctrl_lo", "0", sw_params)

        # Load
        circuit.add_resistor("R_load", "out", "0", R_load)

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

        # High-side ON: V_out ≈ V_dc
        print(f"\nHalf-bridge (high ON, low OFF): V_out = {v_out:.2f}V (expected ~{V_dc}V)")

        assert v_out > V_dc * 0.95, f"High-side not conducting: {v_out}V"

    def test_halfbridge_low_side_on(self):
        """Test half-bridge with low-side switch on."""
        V_dc = 48.0
        R_load = 10.0

        circuit = sl.Circuit()
        circuit.add_voltage_source("Vdc", "vcc", "0", V_dc)

        # High-side control OFF, low-side control ON
        circuit.add_voltage_source("Vhi", "ctrl_hi", "0", 0.0)
        circuit.add_voltage_source("Vlo", "ctrl_lo", "0", 5.0)

        sw_params = sl.SwitchParams()
        sw_params.ron = 0.01
        sw_params.roff = 1e9
        sw_params.vth = 2.5

        circuit.add_switch("S_hi", "vcc", "out", "ctrl_hi", "0", sw_params)
        circuit.add_switch("S_lo", "out", "0", "ctrl_lo", "0", sw_params)
        circuit.add_resistor("R_load", "out", "0", R_load)

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

        # Low-side ON: V_out ≈ 0
        print(f"\nHalf-bridge (high OFF, low ON): V_out = {v_out:.4f}V (expected ~0V)")

        assert abs(v_out) < 0.1, f"Low-side not conducting to ground: {v_out}V"


# =============================================================================
# Test: Switch Resistance Values
# =============================================================================

class TestSwitchResistance:
    """Test switch on/off resistance values."""

    @pytest.mark.parametrize("R_on", [0.001, 0.01, 0.1, 1.0])
    def test_switch_on_resistance(self, R_on):
        """Verify switch on-resistance affects output."""
        V_in = 10.0
        R_load = 100.0

        circuit = sl.Circuit()
        circuit.add_voltage_source("Vdc", "in", "0", V_in)
        circuit.add_voltage_source("Vctrl", "ctrl", "0", 5.0)

        sw_params = sl.SwitchParams()
        sw_params.ron = R_on
        sw_params.roff = 1e9
        sw_params.vth = 2.5
        circuit.add_switch("S1", "in", "out", "ctrl", "0", sw_params)

        circuit.add_resistor("R1", "out", "0", R_load)

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
        V_expected = V_in * R_load / (R_on + R_load)
        relative_error = abs(v_out - V_expected) / V_expected

        print(f"\nR_on = {R_on}Ω: V_out = {v_out:.4f}V (expected {V_expected:.4f}V)")

        assert relative_error < 0.01, f"On-resistance error: {relative_error:.2%}"


# =============================================================================
# Run validation suite
# =============================================================================

def run_switch_validation_suite(verbose: bool = True):
    """Run all switch validation tests."""
    print("=" * 60)
    print("SWITCH VALIDATION TESTS")
    print("=" * 60)

    # Basic tests
    print("\n--- Basic Switch Tests ---")
    test = TestBasicSwitch()
    test.test_switch_on_state()
    test.test_switch_off_state()
    test.test_switch_threshold()

    # Inductive load tests
    print("\n--- Switch + Inductive Load Tests ---")
    test = TestSwitchInductiveLoad()
    test.test_switch_rl_circuit()

    # Half-bridge tests
    print("\n--- Half-Bridge Tests ---")
    test = TestHalfBridge()
    test.test_halfbridge_high_side_on()
    test.test_halfbridge_low_side_on()

    # Resistance tests
    print("\n--- Switch Resistance Tests ---")
    test = TestSwitchResistance()
    for R_on in [0.001, 0.01, 0.1]:
        test.test_switch_on_resistance(R_on)

    print("\n" + "=" * 60)
    print("All switch tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_switch_validation_suite()
