"""Tests for SPICE netlist parser."""

import numpy as np
import pytest

import pulsim as ps
from pulsim.netlist import (
    parse_netlist,
    parse_netlist_verbose,
    parse_value,
    NetlistParseError,
)


class TestParseValue:
    """Tests for parse_value function."""

    def test_plain_integers(self):
        assert parse_value('100') == 100.0
        assert parse_value('0') == 0.0
        assert parse_value('-5') == -5.0

    def test_plain_floats(self):
        assert parse_value('1.5') == 1.5
        assert parse_value('0.001') == 0.001
        assert parse_value('-3.14') == -3.14

    def test_scientific_notation(self):
        assert parse_value('1e-6') == 1e-6
        assert parse_value('1E-6') == 1e-6
        assert parse_value('1.5e3') == 1500.0
        assert parse_value('2.5E+3') == 2500.0

    def test_engineering_suffixes(self):
        # Femto, pico, nano, micro, milli
        assert parse_value('1f') == pytest.approx(1e-15)
        assert parse_value('10p') == pytest.approx(10e-12)
        assert parse_value('100n') == pytest.approx(100e-9)
        assert parse_value('1u') == pytest.approx(1e-6)
        assert parse_value('10m') == pytest.approx(10e-3)

        # Kilo, mega, giga, tera
        assert parse_value('10k') == pytest.approx(10e3)
        assert parse_value('1.5meg') == pytest.approx(1.5e6)
        assert parse_value('1g') == pytest.approx(1e9)
        assert parse_value('1t') == pytest.approx(1e12)

    def test_case_insensitive(self):
        assert parse_value('10K') == pytest.approx(10e3)
        assert parse_value('1MEG') == pytest.approx(1e6)
        assert parse_value('100U') == pytest.approx(100e-6)
        assert parse_value('1N') == pytest.approx(1e-9)

    def test_combined_suffix(self):
        assert parse_value('4.7k') == pytest.approx(4700.0)
        assert parse_value('100n') == pytest.approx(100e-9)
        assert parse_value('2.2u') == pytest.approx(2.2e-6)

    def test_invalid_value(self):
        with pytest.raises(NetlistParseError):
            parse_value('abc')
        with pytest.raises(NetlistParseError):
            parse_value('10x')
        with pytest.raises(NetlistParseError):
            parse_value('')


class TestBasicParsing:
    """Tests for basic netlist parsing."""

    def test_simple_resistor(self):
        netlist = "R1 in out 1k"
        ckt = parse_netlist(netlist)
        assert ckt.num_devices() == 1
        assert ckt.num_nodes() == 2

    def test_simple_rc_circuit(self):
        netlist = """
        * Simple RC Circuit
        V1 in 0 5
        R1 in out 1k
        C1 out 0 1u
        .end
        """
        ckt = parse_netlist(netlist)
        assert ckt.num_devices() == 3
        assert ckt.num_nodes() == 2  # 'in' and 'out'

    def test_voltage_divider(self):
        netlist = """
        V1 in 0 10
        R1 in out 1k
        R2 out 0 1k
        """
        ckt = parse_netlist(netlist)
        assert ckt.num_devices() == 3

        # DC analysis should work
        opts = ps.NewtonOptions()
        x0 = np.zeros(ckt.system_size())
        dc = ps.solve_dc(ckt, x0, opts)
        assert dc.success()

        # Check voltages (v_in=10V, v_out=5V)
        assert abs(dc.solution[0] - 10.0) < 0.01
        assert abs(dc.solution[1] - 5.0) < 0.01

    def test_title_extraction(self):
        netlist = """
        * My Circuit Title
        R1 a b 1k
        """
        result = parse_netlist_verbose(netlist)
        assert result.title == "My Circuit Title"

    def test_node_map(self):
        netlist = """
        R1 node_a node_b 1k
        R2 node_b node_c 2k
        """
        result = parse_netlist_verbose(netlist)
        assert 'node_a' in result.node_map
        assert 'node_b' in result.node_map
        assert 'node_c' in result.node_map


class TestGroundAliases:
    """Tests for ground node aliases."""

    def test_zero_ground(self):
        netlist = "R1 a 0 1k"
        ckt = parse_netlist(netlist)
        assert ckt.num_nodes() == 1  # Only 'a', '0' is ground

    def test_gnd_lowercase(self):
        netlist = "R1 a gnd 1k"
        ckt = parse_netlist(netlist)
        assert ckt.num_nodes() == 1

    def test_gnd_uppercase(self):
        netlist = "R1 a GND 1k"
        ckt = parse_netlist(netlist)
        assert ckt.num_nodes() == 1

    def test_ground_full(self):
        netlist = "R1 a ground 1k"
        ckt = parse_netlist(netlist)
        assert ckt.num_nodes() == 1


class TestCapacitorInductor:
    """Tests for capacitor and inductor parsing."""

    def test_capacitor_basic(self):
        netlist = "C1 a b 100n"
        ckt = parse_netlist(netlist)
        assert ckt.num_devices() == 1

    def test_capacitor_with_ic(self):
        netlist = "C1 a 0 1u IC=5"
        ckt = parse_netlist(netlist)
        assert ckt.num_devices() == 1

    def test_inductor_basic(self):
        netlist = "L1 a b 100u"
        ckt = parse_netlist(netlist)
        assert ckt.num_devices() == 1

    def test_inductor_with_ic(self):
        netlist = "L1 a 0 1m IC=0.5"
        ckt = parse_netlist(netlist)
        assert ckt.num_devices() == 1


class TestSources:
    """Tests for voltage and current source parsing."""

    def test_voltage_source(self):
        netlist = "V1 pos neg 12"
        ckt = parse_netlist(netlist)
        assert ckt.num_devices() == 1

    def test_voltage_source_dc_keyword(self):
        netlist = "V1 pos neg DC 12"
        ckt = parse_netlist(netlist)
        assert ckt.num_devices() == 1

    def test_current_source(self):
        netlist = "I1 pos neg 1m"
        ckt = parse_netlist(netlist)
        assert ckt.num_devices() == 1

    def test_current_source_dc_keyword(self):
        netlist = "I1 pos neg DC 500u"
        ckt = parse_netlist(netlist)
        assert ckt.num_devices() == 1


class TestNonlinearDevices:
    """Tests for nonlinear device parsing."""

    def test_diode(self):
        netlist = "D1 anode cathode"
        ckt = parse_netlist(netlist)
        assert ckt.num_devices() == 1

    def test_switch_on(self):
        netlist = "S1 a b ON"
        ckt = parse_netlist(netlist)
        assert ckt.num_devices() == 1

    def test_switch_off(self):
        netlist = "S1 a b OFF"
        ckt = parse_netlist(netlist)
        assert ckt.num_devices() == 1

    def test_mosfet(self):
        netlist = "M1 drain gate source"
        ckt = parse_netlist(netlist)
        assert ckt.num_devices() == 1
        assert ckt.num_nodes() == 3

    def test_igbt(self):
        netlist = "Q1 collector gate emitter"
        ckt = parse_netlist(netlist)
        assert ckt.num_devices() == 1
        assert ckt.num_nodes() == 3


class TestModelDirective:
    """Tests for .MODEL directive parsing."""

    def test_nmos_model(self):
        netlist = """
        M1 d g s MYMOS
        .MODEL MYMOS NMOS (VTH=2 KP=0.5)
        """
        result = parse_netlist_verbose(netlist)
        assert 'MYMOS' in result.models
        model = result.models['MYMOS']
        assert model.vth == 2.0
        assert model.kp == 0.5
        assert model.is_nmos

    def test_pmos_model(self):
        netlist = """
        M1 d g s PMODEL
        .MODEL PMODEL PMOS (VTH=-1.5)
        """
        result = parse_netlist_verbose(netlist)
        model = result.models['PMODEL']
        assert not model.is_nmos


class TestComments:
    """Tests for comment handling."""

    def test_star_comment(self):
        netlist = """
        * This is a comment
        R1 a b 1k
        * Another comment
        """
        ckt = parse_netlist(netlist)
        assert ckt.num_devices() == 1

    def test_semicolon_inline_comment(self):
        netlist = "R1 a b 1k ; inline comment"
        ckt = parse_netlist(netlist)
        assert ckt.num_devices() == 1


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_resistor_value(self):
        with pytest.raises(NetlistParseError) as exc:
            parse_netlist("R1 a b")
        assert "Line 1" in str(exc.value)
        assert "Resistor requires" in str(exc.value)

    def test_invalid_value(self):
        with pytest.raises(NetlistParseError) as exc:
            parse_netlist("R1 a b xyz")
        assert "Cannot parse value" in str(exc.value)

    def test_negative_resistance(self):
        with pytest.raises(NetlistParseError) as exc:
            parse_netlist("R1 a b -1k")
        assert "positive" in str(exc.value).lower()

    def test_missing_mosfet_nodes(self):
        with pytest.raises(NetlistParseError):
            parse_netlist("M1 d g")  # Missing source

    def test_unknown_directive_strict(self):
        with pytest.raises(NetlistParseError):
            parse_netlist(".UNKNOWN", strict=True)

    def test_unknown_directive_lenient(self):
        result = parse_netlist_verbose(".OPTIONS RELTOL=1e-3", strict=False)
        assert len(result.warnings) == 1
        assert "Unsupported directive" in str(result.warnings[0])


class TestIntegration:
    """Integration tests with full simulation."""

    def test_rc_transient(self):
        """Test RC circuit transient simulation."""
        netlist = """
        * RC Step Response
        V1 in 0 5
        R1 in out 1k
        C1 out 0 1u
        """
        ckt = parse_netlist(netlist)

        # Run transient (5 time constants = 5ms)
        x0 = np.zeros(ckt.system_size())
        times, states, success, msg = ps.run_transient(ckt, 0.0, 5e-3, 1e-6, x0)

        assert success
        states_arr = np.array(states)

        # Final voltage should be close to 5V
        v_final = states_arr[-1, 1]
        assert abs(v_final - 5.0) < 0.1  # Within 2% of final value

    def test_rl_circuit(self):
        """Test RL circuit DC analysis."""
        netlist = """
        V1 in 0 10
        R1 in out 100
        L1 out 0 1m
        """
        ckt = parse_netlist(netlist)

        opts = ps.NewtonOptions()
        x0 = np.zeros(ckt.system_size())
        dc = ps.solve_dc(ckt, x0, opts)

        assert dc.success()
        # In DC, inductor is short circuit, so v_out = 0
        # Current through R = 10V / 100Ω = 0.1A
