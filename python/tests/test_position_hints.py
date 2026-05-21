"""Phase 2 tests for position-hint storage on Circuit (Python side).

Covers the contract in
`openspec/changes/add-python-schematic-renderer/specs/python-bindings/spec.md`
("Position Hint Binding") plus the YAML round-trip path documented in
`specs/netlist-yaml/spec.md`.

Hints do NOT affect simulation; they're consumed by the renderer in
Phase 3. These tests only verify storage + round-trip behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import pulsim as ps


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_circuit_starts_with_no_hints() -> None:
    ckt = ps.Circuit()
    assert ckt.num_position_hints() == 0
    assert ckt.position_hints() == {}


def test_set_position_layer_slot_round_trips() -> None:
    ckt = ps.Circuit()
    ckt.set_position("Q1", layer=2, slot=1)
    hint = ckt.position_hint("Q1")
    assert hint is not None
    assert hint.layer == 2
    assert hint.slot == 1
    assert hint.x is None
    assert hint.y is None


def test_set_position_xy_round_trips() -> None:
    ckt = ps.Circuit()
    ckt.set_position("Cout", x=200.0, y=80.0)
    hint = ckt.position_hint("Cout")
    assert hint is not None
    assert hint.x == 200.0
    assert hint.y == 80.0
    assert hint.layer is None
    assert hint.slot is None


def test_set_position_both_forms_coexist() -> None:
    ckt = ps.Circuit()
    ckt.set_position("M1", layer=3, slot=4, x=120.0, y=60.0)
    hint = ckt.position_hint("M1")
    assert hint is not None
    assert hint.layer == 3
    assert hint.slot == 4
    assert hint.x == 120.0
    assert hint.y == 60.0


def test_position_hint_missing_returns_none() -> None:
    ckt = ps.Circuit()
    ckt.set_position("R1", layer=0, slot=0)
    assert ckt.position_hint("never_set") is None


def test_position_hints_snapshot_is_detached() -> None:
    ckt = ps.Circuit()
    ckt.set_position("R1", layer=0, slot=0)
    snap = ckt.position_hints()
    assert "R1" in snap
    assert snap["R1"].layer == 0

    ckt.set_position("R1", layer=5, slot=5)

    # The snapshot dict is a copy — mutating the circuit doesn't update it.
    assert snap["R1"].layer == 0
    # But a fresh query reflects the new hint.
    assert ckt.position_hint("R1") is not None
    assert ckt.position_hint("R1").layer == 5  # type: ignore[union-attr]


def test_position_hint_readonly() -> None:
    """`PositionHint` exposes its fields read-only — assignment raises."""
    ckt = ps.Circuit()
    ckt.set_position("Q1", layer=2, slot=1)
    hint = ckt.position_hint("Q1")
    assert hint is not None
    with pytest.raises(AttributeError):
        hint.layer = 99  # type: ignore[misc]


def test_set_position_rejects_empty_hint() -> None:
    """At least one of (layer, slot) / (x, y) MUST be set."""
    ckt = ps.Circuit()
    # pybind11 maps invalid_argument → ValueError by default.
    with pytest.raises(ValueError):
        ckt.set_position("R1")


def test_position_hint_repr_smoke() -> None:
    ckt = ps.Circuit()
    ckt.set_position("Q1", layer=2, slot=1)
    hint = ckt.position_hint("Q1")
    rep = repr(hint)
    assert "PositionHint" in rep
    assert "layer" in rep
    assert "slot" in rep


def test_resetting_overwrites_previous_hint() -> None:
    ckt = ps.Circuit()
    ckt.set_position("R1", layer=0, slot=0)
    ckt.set_position("R1", x=50.0, y=25.0)
    hint = ckt.position_hint("R1")
    assert hint is not None
    # Previous (layer, slot) replaced wholesale — NOT merged.
    assert hint.layer is None
    assert hint.slot is None
    assert hint.x == 50.0
    assert hint.y == 25.0


def test_position_hints_dont_affect_components_list() -> None:
    """Setting a hint never registers a new device."""
    ckt = ps.Circuit()
    a = ckt.add_node("a")
    ckt.add_resistor("R1", a, ckt.ground(), 100.0)
    before = ckt.num_components()
    ckt.set_position("R1", layer=0, slot=0)
    assert ckt.num_components() == before
    # The descriptor list is unchanged.
    descs = ckt.components()
    assert len(descs) == 1
    assert descs[0].name == "R1"


def test_hints_can_be_set_before_devices_are_added() -> None:
    """YAML parser pattern: it reads `position:` and `nodes:` in order
    so the hint may be set before the corresponding `add_*` call."""
    ckt = ps.Circuit()
    ckt.set_position("R1", layer=0, slot=0)
    a = ckt.add_node("a")
    ckt.add_resistor("R1", a, ckt.ground(), 100.0)
    hint = ckt.position_hint("R1")
    assert hint is not None
    assert hint.layer == 0


# ----------------------------------------------------------------------
# YAML round-trip tests (specs/netlist-yaml)
# ----------------------------------------------------------------------


def test_yaml_position_layer_slot_round_trips(tmp_path) -> None:
    """A YAML netlist with `position: {layer, slot}` lands in the
    Circuit's position_hints with the same values."""
    yaml_path = tmp_path / "rc_with_positions.yaml"
    yaml_path.write_text(
        """
schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-3
  dt: 1e-6
components:
  - type: voltage_source
    name: V1
    nodes: [vin, 0]
    waveform: { type: dc, value: 5.0 }
    position: { layer: 0, slot: 0 }
  - type: resistor
    name: R1
    nodes: [vin, vout]
    value: 1k
    position: { layer: 1, slot: 0 }
  - type: capacitor
    name: C1
    nodes: [vout, 0]
    value: 1u
    position: { layer: 2, slot: 0 }
"""
    )
    parser = ps.YamlParser(ps.YamlParserOptions())
    ckt, _ = parser.load(str(yaml_path))

    hints = ckt.position_hints()
    assert set(hints.keys()) == {"V1", "R1", "C1"}
    assert hints["V1"].layer == 0 and hints["V1"].slot == 0
    assert hints["R1"].layer == 1 and hints["R1"].slot == 0
    assert hints["C1"].layer == 2 and hints["C1"].slot == 0


def test_yaml_position_xy_round_trips(tmp_path) -> None:
    yaml_path = tmp_path / "rc_with_xy.yaml"
    yaml_path.write_text(
        """
schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-3
  dt: 1e-6
components:
  - type: voltage_source
    name: V1
    nodes: [vin, 0]
    waveform: { type: dc, value: 5.0 }
    position: { x: 10.0, y: 20.0 }
  - type: resistor
    name: R1
    nodes: [vin, 0]
    value: 100
    position: { x: 150.5, y: 50.5 }
"""
    )
    parser = ps.YamlParser(ps.YamlParserOptions())
    ckt, _ = parser.load(str(yaml_path))
    hints = ckt.position_hints()
    assert hints["V1"].x == 10.0 and hints["V1"].y == 20.0
    assert hints["R1"].x == 150.5 and hints["R1"].y == 50.5


def test_yaml_missing_position_is_not_an_error(tmp_path) -> None:
    """Existing demos (rc / buck / half-bridge) have no `position:`
    field — they MUST still parse cleanly and yield an empty hint set."""
    parser = ps.YamlParser(ps.YamlParserOptions())
    ckt, _ = parser.load(str(REPO_ROOT / "examples" / "buck_converter.yaml"))
    assert ckt.num_position_hints() == 0
    assert ckt.position_hints() == {}


def test_yaml_position_empty_map_warns_and_ignores(tmp_path) -> None:
    """A `position:` map with no recognized keys is a warning, not an
    error. The component is still created without a hint."""
    yaml_path = tmp_path / "rc_bad_position.yaml"
    yaml_path.write_text(
        """
schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-3
  dt: 1e-6
components:
  - type: resistor
    name: R1
    nodes: [a, 0]
    value: 100
    position: { foo: 1, bar: 2 }
"""
    )
    parser = ps.YamlParser(ps.YamlParserOptions())
    ckt, _ = parser.load(str(yaml_path))
    # R1 still created.
    descs = {d.name for d in ckt.components()}
    assert "R1" in descs
    # But no hint set.
    assert ckt.num_position_hints() == 0


def test_yaml_position_mixed_xy_and_layer_slot(tmp_path) -> None:
    """A YAML hint that sets both (layer, slot) and (x, y) on the same
    component keeps all four values."""
    yaml_path = tmp_path / "rc_mixed.yaml"
    yaml_path.write_text(
        """
schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-3
  dt: 1e-6
components:
  - type: resistor
    name: R1
    nodes: [a, 0]
    value: 100
    position: { layer: 3, slot: 4, x: 200.0, y: 80.0 }
"""
    )
    parser = ps.YamlParser(ps.YamlParserOptions())
    ckt, _ = parser.load(str(yaml_path))
    hint = ckt.position_hint("R1")
    assert hint is not None
    assert hint.layer == 3
    assert hint.slot == 4
    assert hint.x == 200.0
    assert hint.y == 80.0
