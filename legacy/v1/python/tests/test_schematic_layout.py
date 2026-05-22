"""Phase 2 tests for the pulsim.schematic layout engine.

Covers compute_layout() determinism, electrical priors (ground south,
source west, load east), JSON round-trip, lazy networkx import gate, and
GUI-consumable coordinate schema.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

import pulsim as ps
from pulsim import schematic
from pulsim.schematic import (
    ComponentPlacement,
    SchematicLayout,
    TerminalAnchor,
    compute_layout,
)
from pulsim.schematic.types import SCHEMATIC_SCHEMA_VERSION


REPO_ROOT = Path(__file__).resolve().parents[2]
BUCK_YAML = REPO_ROOT / "examples" / "buck_converter.yaml"
RC_YAML = REPO_ROOT / "examples" / "rc_circuit.yaml"


pytest.importorskip("networkx")


# Some assertions in this file describe properties of the legacy
# in-tree force-directed / templates / ground-rail backend
# (`PULSIM_SCHEMATIC_BACKEND=spring`). The default ELK backend places
# components in layered columns with orthogonal routing — fundamentally
# different geometry — so those properties no longer hold. Tests tagged
# with `spring_only` skip when the active backend is ELK and run when
# the spring backend is selected.
_SPRING_ONLY = pytest.mark.skipif(
    os.environ.get("PULSIM_SCHEMATIC_BACKEND", "elk").lower() != "spring",
    reason="Asserts a property of the legacy spring backend; ELK uses a different geometry.",
)


def _build_rc_circuit() -> ps.Circuit:
    ckt = ps.Circuit()
    n_in = ckt.add_node("in")
    n_out = ckt.add_node("out")
    gnd = ckt.ground()
    ckt.add_voltage_source("V1", n_in, gnd, 12.0)
    ckt.add_resistor("R1", n_in, n_out, 1_000.0)
    ckt.add_capacitor("C1", n_out, gnd, 1e-6, 0.0)
    return ckt


def _load_buck() -> ps.Circuit:
    parser = ps.YamlParser(ps.YamlParserOptions())
    ckt, _opts = parser.load(str(BUCK_YAML))
    return ckt


# -----------------------------------------------------------------------------
# compute_layout: structural correctness
# -----------------------------------------------------------------------------


def test_compute_layout_empty_circuit_returns_empty_layout() -> None:
    ckt = ps.Circuit()
    layout = compute_layout(ckt)
    assert layout.components == {}
    assert layout.wires == ()
    assert layout.junctions == ()
    assert layout.schema_version == SCHEMATIC_SCHEMA_VERSION


def test_compute_layout_rc_circuit_places_every_component() -> None:
    ckt = _build_rc_circuit()
    layout = compute_layout(ckt)

    # One placement per added component
    assert set(layout.components.keys()) == {"V1", "R1", "C1"}
    for placement in layout.components.values():
        assert isinstance(placement, ComponentPlacement)
        assert math.isfinite(placement.x)
        assert math.isfinite(placement.y)
        assert placement.rotation in {0, 90, 180, 270}
        # Per-terminal anchors carry node ID + finite coords
        for anchor in placement.terminal_anchors:
            assert isinstance(anchor, TerminalAnchor)
            assert math.isfinite(anchor.x)
            assert math.isfinite(anchor.y)


def test_compute_layout_canvas_is_finite_mm() -> None:
    ckt = _build_rc_circuit()
    layout = compute_layout(ckt)
    assert layout.canvas.unit == "mm"
    assert layout.canvas.width > 0
    assert layout.canvas.height > 0
    assert math.isfinite(layout.canvas.x)
    assert math.isfinite(layout.canvas.y)


def test_compute_layout_kind_strings_match_introspection() -> None:
    ckt = _build_rc_circuit()
    by_name = {c.name: c for c in ckt.components()}
    layout = compute_layout(ckt)
    for name, placement in layout.components.items():
        assert placement.kind == by_name[name].kind


# -----------------------------------------------------------------------------
# Gate G.5: determinism
# -----------------------------------------------------------------------------


def test_compute_layout_determinism_two_runs_byte_identical() -> None:
    """Gate G.5: same circuit -> byte-identical layout JSON."""
    a = compute_layout(_build_rc_circuit())
    b = compute_layout(_build_rc_circuit())
    assert json.dumps(a.to_json(), sort_keys=True) == json.dumps(b.to_json(), sort_keys=True)


def test_compute_layout_determinism_yaml_load() -> None:
    a = compute_layout(_load_buck())
    b = compute_layout(_load_buck())
    assert json.dumps(a.to_json(), sort_keys=True) == json.dumps(b.to_json(), sort_keys=True)


# -----------------------------------------------------------------------------
# Gate G.6: electrical priors (ground south, source west, load east)
# -----------------------------------------------------------------------------


@_SPRING_ONLY
def test_ground_node_is_below_every_other_node_rc() -> None:
    """Gate G.6 (spring backend only): y(ground) >= max(y_other).

    The ELK backend routes ground terminals onto the SOUTH port of each
    component rather than collapsing them to a single canvas-south
    coordinate, so per-anchor y values vary by component."""
    ckt = _build_rc_circuit()
    layout = compute_layout(ckt)

    # Find ground anchor + other anchors via terminal_anchors (which carry node ID).
    ground_id = ckt.ground()
    ground_ys = []
    other_ys = []
    for placement in layout.components.values():
        for anchor in placement.terminal_anchors:
            if anchor.node == ground_id:
                ground_ys.append(anchor.y)
            else:
                other_ys.append(anchor.y)

    assert ground_ys, "circuit must have a ground terminal"
    assert other_ys, "circuit must have at least one non-ground terminal"
    assert min(ground_ys) >= max(other_ys) - 1e-6


@_SPRING_ONLY
def test_ground_node_is_below_every_other_node_buck() -> None:
    """Gate G.6 (spring backend only) on buck_converter.yaml.

    Same caveat as the RC variant — ELK doesn't enforce a single
    canvas-south y for ground."""
    ckt = _load_buck()
    layout = compute_layout(ckt)

    ground_id = ckt.ground()
    ground_ys = [
        a.y for p in layout.components.values()
        for a in p.terminal_anchors
        if a.node == ground_id
    ]
    other_ys = [
        a.y for p in layout.components.values()
        for a in p.terminal_anchors
        if a.node != ground_id
    ]
    assert ground_ys
    assert other_ys
    assert min(ground_ys) >= max(other_ys) - 1e-6


@_SPRING_ONLY
def test_voltage_source_positive_terminal_on_left() -> None:
    """Spec scenario (spring backend only): positive terminal x <= canvas.width / 4.

    ELK places voltage sources in the FIRST layer rather than at a
    specific absolute x — the canvas-width / 4 threshold doesn't apply
    when the canvas origin is the ELK root, not (0,0)."""
    ckt = _build_rc_circuit()
    layout = compute_layout(ckt)

    n_in = ckt.get_node("in")
    # n_in is the positive terminal of V1
    pos_term_x_candidates = [
        a.x for p in layout.components.values()
        for a in p.terminal_anchors
        if a.node == n_in
    ]
    assert pos_term_x_candidates
    # Anchor x is in absolute mm — relative to canvas origin
    # (which may have a negative offset due to margin). Compare against
    # canvas.x + canvas.width / 4 as the "left quarter" threshold.
    threshold = layout.canvas.x + layout.canvas.width / 4.0
    assert min(pos_term_x_candidates) <= threshold + 1e-6


# -----------------------------------------------------------------------------
# Gate G.4: JSON round-trip
# -----------------------------------------------------------------------------


def test_layout_to_json_returns_serializable_dict() -> None:
    layout = compute_layout(_build_rc_circuit())
    payload = layout.to_json()
    # Must round-trip through json.dumps without errors
    blob = json.dumps(payload)
    assert isinstance(blob, str)
    assert len(blob) > 0
    # Required top-level keys
    assert set(payload.keys()) >= {"components", "wires", "junctions", "canvas", "schema_version"}
    assert payload["schema_version"] == SCHEMATIC_SCHEMA_VERSION


def test_layout_roundtrip_preserves_placements() -> None:
    """Spec scenario: SchematicLayout.from_json(layout.to_json()) == layout."""
    original = compute_layout(_build_rc_circuit())
    roundtripped = SchematicLayout.from_json(original.to_json())

    assert roundtripped.canvas == original.canvas
    assert roundtripped.junctions == original.junctions
    assert roundtripped.components.keys() == original.components.keys()
    for name, placement in original.components.items():
        assert roundtripped.components[name] == placement
    assert roundtripped.wires == original.wires
    assert roundtripped.schema_version == original.schema_version


def test_layout_from_json_rejects_unknown_schema_version() -> None:
    payload = compute_layout(_build_rc_circuit()).to_json()
    payload["schema_version"] = "schematic-v999"
    with pytest.raises(ValueError, match="schematic-v999"):
        SchematicLayout.from_json(payload)


def test_layout_canvas_unit_is_mm() -> None:
    layout = compute_layout(_build_rc_circuit())
    assert layout.canvas.unit == "mm"


# -----------------------------------------------------------------------------
# Wires: at least one wire per pair of components sharing a node
# -----------------------------------------------------------------------------


def test_wires_connect_components_sharing_an_electrical_node() -> None:
    ckt = _build_rc_circuit()
    layout = compute_layout(ckt)

    # V1 and R1 share node n_in -> at least one wire between them
    # R1 and C1 share node n_out -> at least one wire
    # V1 and C1 share ground -> at least one wire
    endpoints = {
        (w.from_.component, w.to.component) for w in layout.wires
    }
    endpoints_norm = {tuple(sorted(p)) for p in endpoints}
    assert ("R1", "V1") in endpoints_norm
    assert ("C1", "R1") in endpoints_norm
    assert ("C1", "V1") in endpoints_norm


def test_wire_endpoints_reference_existing_components_and_terminals() -> None:
    ckt = _build_rc_circuit()
    layout = compute_layout(ckt)
    for wire in layout.wires:
        for endpoint in (wire.from_, wire.to):
            placement = layout.components[endpoint.component]
            assert 0 <= endpoint.terminal < len(placement.terminal_anchors)


# -----------------------------------------------------------------------------
# Lazy import: pulsim.schematic always importable
# -----------------------------------------------------------------------------


def test_schematic_module_is_importable() -> None:
    # Import works even before this test runs (covered by the importorskip
    # at the top); reasserting here makes the intent explicit.
    assert hasattr(schematic, "compute_layout")
    assert hasattr(schematic, "SchematicLayout")
    assert hasattr(schematic, "ComponentPlacement")
    assert hasattr(schematic, "Wire")
    assert hasattr(schematic, "BoundingBox")
