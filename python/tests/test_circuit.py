"""Tests for circuit building and validation."""

import pytest
import pulsim as sl


class TestCircuitConstruction:
    """Test circuit construction with various components."""

    def test_empty_circuit(self):
        """Empty circuit should have no nodes."""
        circuit = sl.Circuit()
        assert circuit.node_count() == 0
        assert circuit.total_variables() == 0

    def test_add_resistor(self):
        """Adding a resistor creates two nodes."""
        circuit = sl.Circuit()
        circuit.add_resistor("R1", "n1", "n2", 1000.0)
        assert circuit.node_count() == 2
        assert "n1" in circuit.node_names()
        assert "n2" in circuit.node_names()

    def test_add_capacitor(self):
        """Adding a capacitor with initial condition."""
        circuit = sl.Circuit()
        circuit.add_capacitor("C1", "in", "out", 1e-6, ic=5.0)
        assert circuit.node_count() == 2

    def test_add_inductor(self):
        """Adding an inductor creates a branch current variable."""
        circuit = sl.Circuit()
        circuit.add_inductor("L1", "in", "out", 1e-3)
        assert circuit.node_count() == 2
        assert circuit.branch_count() == 1
        assert circuit.total_variables() == 3

    def test_add_voltage_source(self):
        """Adding a voltage source creates a branch current."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "vcc", "0", 12.0)
        # Ground node "0" is not counted (reference node)
        assert circuit.node_count() == 1
        assert circuit.branch_count() == 1

    def test_add_current_source(self):
        """Current source doesn't add branch currents."""
        circuit = sl.Circuit()
        circuit.add_current_source("I1", "n1", "0", 1e-3)
        # Ground node "0" is not counted (reference node)
        assert circuit.node_count() == 1
        assert circuit.branch_count() == 0

    def test_add_diode(self):
        """Adding a diode."""
        circuit = sl.Circuit()
        params = sl.DiodeParams()
        params.ideal = True
        circuit.add_diode("D1", "anode", "cathode", params)
        assert circuit.node_count() == 2

    def test_add_switch(self):
        """Adding a switch with control nodes."""
        circuit = sl.Circuit()
        params = sl.SwitchParams()
        params.ron = 0.01
        params.roff = 1e9
        params.vth = 2.5
        circuit.add_switch("S1", "in", "out", "ctrl", "0", params)
        # Ground node "0" is not counted (reference node)
        assert circuit.node_count() == 3  # in, out, ctrl (not 0)

    def test_add_mosfet(self):
        """Adding a MOSFET."""
        circuit = sl.Circuit()
        params = sl.MOSFETParams()
        params.type = sl.MOSFETType.NMOS
        params.vth = 2.0
        params.rds_on = 0.05
        circuit.add_mosfet("M1", "drain", "gate", "source", params)
        assert circuit.node_count() == 3

    def test_add_transformer(self):
        """Adding a transformer creates two branch currents."""
        circuit = sl.Circuit()
        params = sl.TransformerParams()
        params.turns_ratio = 10.0
        circuit.add_transformer("T1", "p1", "p2", "s1", "s2", params)
        assert circuit.node_count() == 4
        assert circuit.branch_count() == 2

    def test_circuit_validation(self):
        """Valid circuit should pass validation."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", 5.0)
        circuit.add_resistor("R1", "in", "out", 1000.0)
        circuit.add_capacitor("C1", "out", "0", 1e-6)

        valid, error = circuit.validate()
        assert valid
        assert error == ""


class TestWaveforms:
    """Test waveform types."""

    def test_dc_waveform(self):
        """DC waveform construction."""
        dc = sl.DCWaveform(5.0)
        assert dc.value == 5.0

    def test_pulse_waveform(self):
        """Pulse waveform construction."""
        pulse = sl.PulseWaveform()
        pulse.v1 = 0.0
        pulse.v2 = 5.0
        pulse.td = 0.0
        pulse.tr = 1e-9
        pulse.tf = 1e-9
        pulse.pw = 50e-6
        pulse.period = 100e-6

        assert pulse.v1 == 0.0
        assert pulse.v2 == 5.0
        assert pulse.period == 100e-6

    def test_sine_waveform(self):
        """Sine waveform construction."""
        sine = sl.SineWaveform()
        sine.offset = 0.0
        sine.amplitude = 1.0
        sine.frequency = 1000.0

        assert sine.frequency == 1000.0

    def test_pwl_waveform(self):
        """PWL waveform construction."""
        pwl = sl.PWLWaveform()
        pwl.points = [(0.0, 0.0), (1e-3, 5.0), (2e-3, 5.0), (3e-3, 0.0)]

        assert len(pwl.points) == 4
        assert pwl.points[1] == (1e-3, 5.0)

    def test_pwm_waveform(self):
        """PWM waveform with dead-time."""
        pwm = sl.PWMWaveform()
        pwm.v_off = 0.0
        pwm.v_on = 15.0
        pwm.frequency = 20e3
        pwm.duty = 0.5
        pwm.dead_time = 1e-6
        pwm.phase = 0.0
        pwm.complementary = False

        assert pwm.frequency == 20e3
        assert pwm.dead_time == 1e-6
        assert pwm.period() == pytest.approx(50e-6)
        assert pwm.t_on() == pytest.approx(25e-6)


class TestComponentParams:
    """Test component parameter classes."""

    def test_diode_params(self):
        """Diode parameters."""
        params = sl.DiodeParams()
        params.ideal = False
        params.n = 1.5
        assert params.n == 1.5

    def test_switch_params(self):
        """Switch parameters."""
        params = sl.SwitchParams()
        params.ron = 0.001
        params.roff = 1e12
        params.vth = 3.0
        params.initial_state = True

        assert params.ron == 0.001
        assert params.initial_state is True

    def test_mosfet_params(self):
        """MOSFET parameters."""
        params = sl.MOSFETParams()
        params.type = sl.MOSFETType.PMOS
        params.vth = 3.0
        params.kp = 100e-6
        params.w = 10e-6
        params.l = 1e-6

        assert params.type == sl.MOSFETType.PMOS
        # kp_effective = kp * w / l = 100e-6 * 10e-6 / 1e-6 = 1e-3
        assert params.kp_effective() == pytest.approx(1e-3)

    def test_transformer_params(self):
        """Transformer parameters."""
        params = sl.TransformerParams()
        params.turns_ratio = 5.0
        params.lm = 1e-3
        params.ll1 = 1e-6
        params.ll2 = 1e-6

        assert params.turns_ratio == 5.0
