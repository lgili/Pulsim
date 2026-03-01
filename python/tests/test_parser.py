"""Tests for netlist parsing."""

import pytest
import tempfile
import os
import pulsim as sl


class TestNetlistParsing:
    """Test JSON netlist parsing."""

    def test_parse_simple_rc(self):
        """Parse a simple RC circuit."""
        netlist = '''
        {
            "name": "RC Circuit",
            "components": [
                {"type": "voltage_source", "name": "V1", "npos": "in", "nneg": "0",
                 "waveform": {"type": "dc", "value": 5.0}},
                {"type": "resistor", "name": "R1", "n1": "in", "n2": "out", "value": "1k"},
                {"type": "capacitor", "name": "C1", "n1": "out", "n2": "0", "value": "1u"}
            ]
        }
        '''
        circuit = sl.parse_netlist_string(netlist)

        # Ground node "0" is not counted (reference node)
        assert circuit.node_count() == 2  # in, out (not 0)
        valid, error = circuit.validate()
        assert valid

    def test_parse_with_pulse(self):
        """Parse circuit with pulse waveform."""
        netlist = '''
        {
            "name": "Pulse Test",
            "components": [
                {"type": "voltage_source", "name": "V1", "npos": "in", "nneg": "0",
                 "waveform": {"type": "pulse", "v1": 0, "v2": 5, "td": 0,
                              "tr": 1e-9, "tf": 1e-9, "pw": 50e-6, "period": 100e-6}},
                {"type": "resistor", "name": "R1", "n1": "in", "n2": "0", "value": 1000}
            ]
        }
        '''
        circuit = sl.parse_netlist_string(netlist)
        valid, _ = circuit.validate()
        assert valid

    def test_parse_with_pwm(self):
        """Parse circuit with PWM waveform including dead-time."""
        netlist = '''
        {
            "name": "PWM Test",
            "components": [
                {"type": "voltage_source", "name": "Vpwm", "npos": "ctrl", "nneg": "0",
                 "waveform": {"type": "pwm", "v_off": 0, "v_on": 15,
                              "frequency": 20000, "duty": 0.5,
                              "dead_time": 1e-6, "complementary": false}},
                {"type": "resistor", "name": "R1", "n1": "ctrl", "n2": "0", "value": 1000}
            ]
        }
        '''
        circuit = sl.parse_netlist_string(netlist)
        valid, _ = circuit.validate()
        assert valid

    def test_parse_value_suffixes(self):
        """Parse values with engineering suffixes."""
        netlist = '''
        {
            "components": [
                {"type": "voltage_source", "name": "V1", "npos": "in", "nneg": "0",
                 "waveform": {"type": "dc", "value": 5.0}},
                {"type": "resistor", "name": "R1", "n1": "in", "n2": "n1", "value": "1k"},
                {"type": "resistor", "name": "R2", "n1": "n1", "n2": "n2", "value": "10meg"},
                {"type": "capacitor", "name": "C1", "n1": "n2", "n2": "0", "value": "100n"},
                {"type": "inductor", "name": "L1", "n1": "n1", "n2": "0", "value": "10u"}
            ]
        }
        '''
        circuit = sl.parse_netlist_string(netlist)
        valid, _ = circuit.validate()
        assert valid

    def test_parse_file(self):
        """Parse circuit from file."""
        netlist = '''
        {
            "components": [
                {"type": "voltage_source", "name": "V1", "npos": "in", "nneg": "0",
                 "waveform": 5.0},
                {"type": "resistor", "name": "R1", "n1": "in", "n2": "0", "value": 1000}
            ]
        }
        '''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(netlist)
            path = f.name

        try:
            circuit = sl.parse_netlist_file(path)
            valid, _ = circuit.validate()
            assert valid
        finally:
            os.unlink(path)

    def test_parse_mosfet(self):
        """Parse MOSFET component."""
        netlist = '''
        {
            "components": [
                {"type": "voltage_source", "name": "Vds", "npos": "drain", "nneg": "0", "waveform": 5.0},
                {"type": "voltage_source", "name": "Vgs", "npos": "gate", "nneg": "0", "waveform": 10.0},
                {"type": "mosfet", "name": "M1", "drain": "drain", "gate": "gate", "source": "0",
                 "vth": 2.0, "rds_on": 0.05}
            ]
        }
        '''
        circuit = sl.parse_netlist_string(netlist)
        valid, _ = circuit.validate()
        assert valid

    def test_parse_transformer(self):
        """Parse transformer component."""
        netlist = '''
        {
            "components": [
                {"type": "voltage_source", "name": "Vpri", "npos": "p1", "nneg": "0", "waveform": 120.0},
                {"type": "transformer", "name": "T1", "p1": "p1", "p2": "0", "s1": "s1", "s2": "0",
                 "turns_ratio": 10.0, "lm": 1e-3},
                {"type": "resistor", "name": "Rload", "n1": "s1", "n2": "0", "value": 100}
            ]
        }
        '''
        circuit = sl.parse_netlist_string(netlist)
        valid, _ = circuit.validate()
        assert valid

    def test_parse_invalid_json(self):
        """Invalid JSON should raise error."""
        with pytest.raises(RuntimeError):
            sl.parse_netlist_string("not valid json")

    def test_parse_missing_components(self):
        """Missing components array should raise error."""
        with pytest.raises(RuntimeError):
            sl.parse_netlist_string('{"name": "Empty"}')
