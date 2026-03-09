"""Tests for SPICE netlist parser."""

from pathlib import Path

import numpy as np
import pytest
import yaml

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
        assert parse_value("100") == 100.0
        assert parse_value("0") == 0.0
        assert parse_value("-5") == -5.0

    def test_plain_floats(self):
        assert parse_value("1.5") == 1.5
        assert parse_value("0.001") == 0.001
        assert parse_value("-3.14") == -3.14

    def test_scientific_notation(self):
        assert parse_value("1e-6") == 1e-6
        assert parse_value("1E-6") == 1e-6
        assert parse_value("1.5e3") == 1500.0
        assert parse_value("2.5E+3") == 2500.0

    def test_engineering_suffixes(self):
        # Femto, pico, nano, micro, milli
        assert parse_value("1f") == pytest.approx(1e-15)
        assert parse_value("10p") == pytest.approx(10e-12)
        assert parse_value("100n") == pytest.approx(100e-9)
        assert parse_value("1u") == pytest.approx(1e-6)
        assert parse_value("10m") == pytest.approx(10e-3)

        # Kilo, mega, giga, tera
        assert parse_value("10k") == pytest.approx(10e3)
        assert parse_value("1.5meg") == pytest.approx(1.5e6)
        assert parse_value("1g") == pytest.approx(1e9)
        assert parse_value("1t") == pytest.approx(1e12)

    def test_case_insensitive(self):
        assert parse_value("10K") == pytest.approx(10e3)
        assert parse_value("1MEG") == pytest.approx(1e6)
        assert parse_value("100U") == pytest.approx(100e-6)
        assert parse_value("1N") == pytest.approx(1e-9)

    def test_combined_suffix(self):
        assert parse_value("4.7k") == pytest.approx(4700.0)
        assert parse_value("100n") == pytest.approx(100e-9)
        assert parse_value("2.2u") == pytest.approx(2.2e-6)

    def test_invalid_value(self):
        with pytest.raises(NetlistParseError):
            parse_value("abc")
        with pytest.raises(NetlistParseError):
            parse_value("10x")
        with pytest.raises(NetlistParseError):
            parse_value("")


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
        assert "node_a" in result.node_map
        assert "node_b" in result.node_map
        assert "node_c" in result.node_map


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
        assert "MYMOS" in result.models
        model = result.models["MYMOS"]
        assert model.vth == 2.0
        assert model.kp == 0.5
        assert model.is_nmos

    def test_pmos_model(self):
        netlist = """
        M1 d g s PMODEL
        .MODEL PMODEL PMOS (VTH=-1.5)
        """
        result = parse_netlist_verbose(netlist)
        model = result.models["PMODEL"]
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


# ---------------------------------------------------------------------------
# 6.3  YAML netlist parser — C_BLOCK component tests
# ---------------------------------------------------------------------------


class TestYamlCBlockParser:
    """YAML netlist round-trip tests for c_block components (task 6.3)."""

    _HEADER = (
        "schema: pulsim-v1\n"
        "version: 1\n"
        "simulation:\n"
        "  tstop: 1e-3\n"
        "  dt: 1e-6\n"
        "components:\n"
    )

    def _parse(self, component_yaml: str) -> ps.YamlParser:
        p = ps.YamlParser()
        p.load_string(self._HEADER + component_yaml)
        return p

    def _has_error_code(self, errors: list, code: str) -> bool:
        return any(code in e for e in errors)

    # 6.3.1 — valid c_block with lib_path produces no ERROR diagnostics
    def test_yaml_cblock_valid_lib_path(self, tmp_path) -> None:
        """Parse valid c_block with lib_path; no ERROR-level diagnostics."""
        lib = tmp_path / "gain.so"
        lib.write_bytes(b"")  # create empty file so existence check passes
        p = self._parse(
            f"  - {{name: CB1, type: c_block, n_inputs: 1, n_outputs: 1,"
            f" lib_path: {lib}, nodes: [a, b]}}\n"
        )
        # No error-level diagnostics
        assert all("PARAM_INVALID" not in e for e in p.errors), p.errors

    # 6.3.2 — valid c_block with source path (existing file)
    def test_yaml_cblock_valid_source(self, tmp_path) -> None:
        """Parse valid c_block with source path; no error-level diagnostics."""
        src = tmp_path / "gain.c"
        src.write_text("// dummy")
        p = self._parse(
            f"  - {{name: CB2, type: c_block, n_inputs: 1, n_outputs: 1,"
            f" source: {src}, nodes: [a, b]}}\n"
        )
        assert all("PARAM_INVALID" not in e for e in p.errors), p.errors

    def test_yaml_cblock_control_only_allows_empty_nodes(self, tmp_path) -> None:
        """Control-domain C_BLOCK supports nodes: [] without being dropped by parser."""
        lib = tmp_path / "gain.so"
        lib.write_bytes(b"")
        component_yaml = (
            f"  - {{name: CB0, type: c_block, n_inputs: 1, n_outputs: 1,"
            f" lib_path: {lib}, nodes: [], inputs: [ERR]}}\n"
        )

        p = self._parse(component_yaml)
        assert all("PARAM_INVALID" not in e for e in p.errors), p.errors

        p_roundtrip = ps.YamlParser()
        circuit, _ = p_roundtrip.load_string(self._HEADER + component_yaml)
        assert p_roundtrip.errors == [], p_roundtrip.errors
        assert any(v.name == "CB0" for v in circuit.virtual_components())

    def test_yaml_pi_controller_accepts_component_sample_time(self) -> None:
        """Control blocks accept per-component sample_time >= 0."""
        p = self._parse(
            "  - {name: PI_TS, type: pi_controller, nodes: [e, fb, out],"
            " kp: 0.1, ki: 10.0, sample_time: 5e-6}\n"
        )
        assert all("PARAM_INVALID" not in e for e in p.errors), p.errors

    def test_yaml_pi_controller_negative_sample_time_rejected(self) -> None:
        """Control block sample_time must be finite and >= 0."""
        p = self._parse(
            "  - {name: PI_BAD_TS, type: pi_controller, nodes: [e, fb, out],"
            " kp: 0.1, ki: 10.0, sample_time: -1e-6}\n"
        )
        assert self._has_error_code(p.errors, "PARAM_INVALID"), p.errors

    def test_yaml_probe_rejects_component_sample_time(self) -> None:
        """Scopes/probes are not schedulable control blocks and must reject sample_time."""
        p = self._parse(
            "  - {name: VP_TS, type: voltage_probe, nodes: [a, b], sample_time: 1e-6}\n"
        )
        assert self._has_error_code(p.errors, "PARAM_INVALID"), p.errors

    def test_yaml_saturable_inductor_accepts_magnetic_core_block(self) -> None:
        """saturable_inductor accepts canonical magnetic_core settings."""
        p = self._parse(
            "  - name: Lsat\n"
            "    type: saturable_inductor\n"
            "    nodes: [a, 0]\n"
            "    inductance: 1m\n"
            "    magnetic_core:\n"
            "      enabled: true\n"
            "      model: saturation\n"
            "      saturation_current: 2\n"
            "      saturation_inductance: 200u\n"
            "      saturation_exponent: 2.5\n"
            "      core_loss_k: 0.01\n"
            "      core_loss_alpha: 1.8\n"
        )
        assert all("MAGNETIC_CONFIG_INVALID" not in e for e in p.errors), p.errors
        assert all("PARAM_INVALID" not in e for e in p.errors), p.errors

    def test_yaml_rejects_magnetic_core_on_unsupported_component(self) -> None:
        """magnetic_core is rejected for components outside supported magnetic families."""
        p = self._parse(
            "  - name: Rmag\n"
            "    type: resistor\n"
            "    nodes: [a, 0]\n"
            "    value: 1\n"
            "    magnetic_core:\n"
            "      enabled: true\n"
            "      model: saturation\n"
        )
        assert self._has_error_code(p.errors, "MAGNETIC_CONFIG_INVALID"), p.errors

    def test_yaml_transformer_accepts_magnetic_core_loss_block(self) -> None:
        """transformer accepts magnetic_core loss parameters in MVP."""
        p = self._parse(
            "  - name: Tmag\n"
            "    type: transformer\n"
            "    nodes: [p1, p2, s1, s2]\n"
            "    turns_ratio: 2.0\n"
            "    magnetic_core:\n"
            "      enabled: true\n"
            "      model: saturation\n"
            "      core_loss_k: 0.03\n"
            "      core_loss_alpha: 1.9\n"
        )
        assert all("MAGNETIC_CONFIG_INVALID" not in e for e in p.errors), p.errors

    def test_yaml_rejects_unsupported_magnetic_core_model(self) -> None:
        """Only saturation model is supported in the initial magnetic_core release."""
        p = self._parse(
            "  - name: Lsat_bad_model\n"
            "    type: saturable_inductor\n"
            "    nodes: [a, 0]\n"
            "    inductance: 1m\n"
            "    magnetic_core:\n"
            "      enabled: true\n"
            "      model: hysteresis\n"
        )
        assert self._has_error_code(p.errors, "MAGNETIC_CONFIG_INVALID"), p.errors

    def test_yaml_coupled_inductor_accepts_magnetic_core_loss_block(self) -> None:
        """coupled_inductor accepts magnetic_core loss parameters in MVP."""
        p = self._parse(
            "  - name: Kmag\n"
            "    type: coupled_inductor\n"
            "    nodes: [p1, p2, s1, s2]\n"
            "    l1: 1m\n"
            "    l2: 1m\n"
            "    coupling: 0.9\n"
            "    magnetic_core:\n"
            "      enabled: true\n"
            "      model: saturation\n"
            "      core_loss_k: 0.05\n"
            "      core_loss_alpha: 2.0\n"
        )
        assert all("MAGNETIC_CONFIG_INVALID" not in e for e in p.errors), p.errors

    # 6.3.3 — both lib_path and source → error
    def test_yaml_cblock_both_specified_error(self, tmp_path) -> None:
        """Specifying both lib_path and source yields a PARAM_INVALID error."""
        p = self._parse(
            "  - {name: CB3, type: c_block, n_inputs: 1, n_outputs: 1,"
            " lib_path: /tmp/x.so, source: /tmp/x.c, nodes: [a, b]}\n"
        )
        assert self._has_error_code(p.errors, "PARAM_INVALID"), p.errors

    # 6.3.4 — missing n_inputs → error
    def test_yaml_cblock_missing_n_inputs_error(self) -> None:
        """Missing n_inputs yields a PARAM_INVALID error."""
        p = self._parse(
            "  - {name: CB4, type: c_block, n_outputs: 1,"
            " lib_path: /tmp/x.so, nodes: [a, b]}\n"
        )
        assert self._has_error_code(p.errors, "PARAM_INVALID"), p.errors

    # 6.3.5 — neither lib_path nor source → no error (Python-only usage)
    def test_yaml_cblock_neither_path_nor_source_info(self) -> None:
        """Neither lib_path nor source is valid for Python-only usage (no ERROR)."""
        p = self._parse(
            "  - {name: CB5, type: c_block, n_inputs: 1, n_outputs: 1, nodes: [a, b]}\n"
        )
        # No error-level (PARAM_INVALID) diagnostics — only warnings
        assert all("PARAM_INVALID" not in e for e in p.errors), p.errors
        # A warning must be emitted (Python-only hint)
        assert any("VIRTUAL" in w or "neither" in w.lower() for w in p.warnings), (
            p.warnings
        )

    # 4.2 / 6.3 — extra_cflags must be list[str]
    def test_yaml_cblock_extra_cflags_must_be_list_of_strings(self) -> None:
        """extra_cflags must be a sequence where each element is a string."""
        p_not_list = self._parse(
            "  - {name: CB6, type: c_block, n_inputs: 1, n_outputs: 1,"
            " source: /tmp/x.c, extra_cflags: -O3, nodes: [a, b]}\n"
        )
        assert self._has_error_code(p_not_list.errors, "TYPE_MISMATCH"), (
            p_not_list.errors
        )

        p_bad_item = self._parse(
            "  - {name: CB7, type: c_block, n_inputs: 1, n_outputs: 1,"
            " source: /tmp/x.c, extra_cflags: ['-O3', [123]], nodes: [a, b]}\n"
        )
        assert self._has_error_code(p_bad_item.errors, "TYPE_MISMATCH"), (
            p_bad_item.errors
        )

    # 4.2 — n_inputs/n_outputs must be integer >= 1
    def test_yaml_cblock_n_inputs_n_outputs_must_be_integer(self) -> None:
        """Fractional n_inputs/n_outputs must be rejected."""
        p = self._parse(
            "  - {name: CB8, type: c_block, n_inputs: 1.5, n_outputs: 1,"
            " source: /tmp/x.c, nodes: [a, b]}\n"
        )
        assert self._has_error_code(p.errors, "TYPE_MISMATCH"), p.errors

    # 4.4 — parse -> re-serialize -> parse contract
    def test_yaml_cblock_round_trip_parse_serialize_parse(self, tmp_path) -> None:
        """Round-trip C_BLOCK fields by re-serializing parsed component metadata."""
        source_path = tmp_path / "gain.c"
        source_path.write_text("/* test */", encoding="utf-8")
        source_yaml_path = source_path.as_posix()

        component_yaml = (
            "  - name: CB_RT\n"
            "    type: c_block\n"
            "    n_inputs: 2\n"
            "    n_outputs: 3\n"
            f'    source: "{source_yaml_path}"\n'
            "    extra_cflags: ['-O3', '-DTEST=1']\n"
            "    nodes: [a, b]\n"
        )
        p = self._parse(component_yaml)
        assert p.errors == [], p.errors

        p_roundtrip = ps.YamlParser()
        circuit, _ = p_roundtrip.load_string(self._HEADER + component_yaml)
        assert p_roundtrip.errors == [], p_roundtrip.errors
        vc = next(v for v in circuit.virtual_components() if v.name == "CB_RT")
        flags = yaml.safe_load(vc.metadata["extra_cflags"])
        assert flags == ["-O3", "-DTEST=1"]

        roundtrip_yaml = (
            self._HEADER
            + "  - name: CB_RT\n"
            + "    type: c_block\n"
            + f"    n_inputs: {int(vc.numeric_params['n_inputs'])}\n"
            + f"    n_outputs: {int(vc.numeric_params['n_outputs'])}\n"
            + f'    source: "{vc.metadata["source"]}"\n'
            + f"    extra_cflags: {yaml.safe_dump(flags, default_flow_style=True).strip()}\n"
            + "    nodes: [a, b]\n"
        )

        p2 = ps.YamlParser()
        circuit2, _ = p2.load_string(roundtrip_yaml)
        assert p2.errors == [], p2.errors
        vc2 = next(v for v in circuit2.virtual_components() if v.name == "CB_RT")
        assert int(vc2.numeric_params["n_inputs"]) == 2
        assert int(vc2.numeric_params["n_outputs"]) == 3
        assert Path(vc2.metadata["source"]).resolve() == source_path.resolve()
        assert yaml.safe_load(vc2.metadata["extra_cflags"]) == flags
