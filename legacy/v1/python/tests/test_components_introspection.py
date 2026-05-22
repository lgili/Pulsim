"""Tests for the Phase 1 component-introspection bindings.

Covers `Circuit.components()`, `Circuit.num_components()`, and
`Circuit.node_position_hint()` — the data layer consumed by the upcoming
`pulsim.schematic` module (see openspec/changes/add-schematic-rendering).
"""

from __future__ import annotations

from pathlib import Path

import pulsim as ps


REPO_ROOT = Path(__file__).resolve().parents[2]
BUCK_YAML = REPO_ROOT / "examples" / "buck_converter.yaml"


def _build_rc_circuit() -> ps.Circuit:
    ckt = ps.Circuit()
    n_in = ckt.add_node("in")
    n_out = ckt.add_node("out")
    gnd = ckt.ground()
    ckt.add_voltage_source("V1", n_in, gnd, 12.0)
    ckt.add_resistor("R1", n_in, n_out, 1_000.0)
    ckt.add_capacitor("C1", n_out, gnd, 1e-6, 0.0)
    return ckt


def test_components_empty_circuit_returns_empty_list() -> None:
    ckt = ps.Circuit()
    assert ckt.num_components() == 0
    assert ckt.components() == []


def test_components_code_built_circuit_insertion_order() -> None:
    ckt = _build_rc_circuit()
    comps = ckt.components()

    assert ckt.num_components() == 3
    assert len(comps) == 3

    assert [c.name for c in comps] == ["V1", "R1", "C1"]
    assert [c.kind for c in comps] == ["voltage_source", "resistor", "capacitor"]


def test_components_params_populated_for_primary_types() -> None:
    ckt = _build_rc_circuit()
    by_name = {c.name: c for c in ckt.components()}

    assert by_name["V1"].params["V"] == 12.0
    assert by_name["R1"].params["R"] == 1000.0
    assert by_name["C1"].params["C"] == 1e-6


def test_components_inductor_current_source_params() -> None:
    ckt = ps.Circuit()
    n1 = ckt.add_node("n1")
    n2 = ckt.add_node("n2")
    gnd = ckt.ground()
    ckt.add_inductor("L1", n1, n2, 100e-6, 0.0)
    ckt.add_current_source("I1", n1, gnd, 0.5)

    comps = ckt.components()
    assert comps[0].kind == "inductor"
    assert comps[0].params["L"] == 100e-6
    assert comps[1].kind == "current_source"
    assert comps[1].params["I"] == 0.5


def test_components_pin_order_preserved() -> None:
    ckt = ps.Circuit()
    a = ckt.add_node("a")
    b = ckt.add_node("b")
    c = ckt.add_node("c")
    gnd = ckt.ground()
    ckt.add_vcswitch("S1", c, a, b)

    comps = ckt.components()
    assert comps[0].kind == "vcswitch"
    # 3-terminal: [ctrl, t1, t2]
    assert comps[0].nodes == [c, a, b]
    assert gnd == ckt.ground()


def test_components_switching_devices_canonical_kinds() -> None:
    ckt = ps.Circuit()
    a = ckt.add_node("a")
    b = ckt.add_node("b")
    ctrl = ckt.add_node("ctrl")

    ckt.add_diode("D1", a, b)
    ckt.add_switch("S1", a, b, False)
    ckt.add_vcswitch("S2", ctrl, a, b)

    kinds = [c.kind for c in ckt.components()]
    assert kinds == ["diode", "switch", "vcswitch"]


def test_components_determinism_same_circuit_twice() -> None:
    a = _build_rc_circuit().components()
    b = _build_rc_circuit().components()

    assert [c.name for c in a] == [c.name for c in b]
    assert [c.kind for c in a] == [c.kind for c in b]
    assert [c.nodes for c in a] == [c.nodes for c in b]
    assert [dict(c.params) for c in a] == [dict(c.params) for c in b]


def test_components_yaml_loaded_circuit_mirrors_yaml_order() -> None:
    parser = ps.YamlParser(ps.YamlParserOptions())
    ckt, _opts = parser.load(str(BUCK_YAML))

    comps = ckt.components()
    # buck_converter.yaml lists 7 atomic components: Vdc, Vpwm, S1, D1, L1, C1, Rload.
    names = [c.name for c in comps]
    assert names == ["Vdc", "Vpwm", "S1", "D1", "L1", "C1", "Rload"]

    kinds = [c.kind for c in comps]
    assert kinds[0] == "voltage_source"
    assert kinds[2] == "vcswitch"
    assert kinds[3] == "diode"
    assert kinds[4] == "inductor"
    assert kinds[5] == "capacitor"
    assert kinds[6] == "resistor"


def test_node_position_hint_ground() -> None:
    ckt = _build_rc_circuit()
    assert ckt.node_position_hint(ckt.ground()) == "ground"


def test_node_position_hint_voltage_source_positive_is_source_pos() -> None:
    ckt = _build_rc_circuit()
    n_in = ckt.get_node("in")
    assert ckt.node_position_hint(n_in) == "source_pos"


def test_node_position_hint_resistor_to_ground_is_load() -> None:
    ckt = ps.Circuit()
    vin = ckt.add_node("vin")
    vout = ckt.add_node("vout")
    ckt.add_voltage_source("V1", vin, ckt.ground(), 12.0)
    ckt.add_resistor("R1", vin, vout, 100.0)
    ckt.add_resistor("Rload", vout, ckt.ground(), 1_000.0)

    assert ckt.node_position_hint(vout) == "load"


def test_node_position_hint_out_of_range_returns_none() -> None:
    ckt = _build_rc_circuit()
    assert ckt.node_position_hint(99) is None


def test_descriptor_repr_smoke() -> None:
    ckt = _build_rc_circuit()
    rep = repr(ckt.components()[0])
    assert "ComponentDescriptor" in rep
    assert "voltage_source" in rep
