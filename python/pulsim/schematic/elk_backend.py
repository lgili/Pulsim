"""ELK (Eclipse Layout Kernel) backend for ``pulsim.schematic``.

Replaces the in-tree force-directed + templates + parallel-pair + ground-rail
pipeline with a single call to ``elkjs`` running ``layered`` Sugiyama with
orthogonal edge routing. ELK's layered algorithm gives us:

- Power-flow direction (left → right) honored by layer assignment.
- Per-port attachment sides (`WEST` / `EAST` / `NORTH` / `SOUTH`) so a
  MOSFET's drain is on top, source on the bottom, gate on the side.
- Manhattan-routed edges with bend points — the visible payoff vs our
  straight-line force-directed wires.
- Crossing minimization (barycenter heuristic) — no more bridge diodes
  forming a tangled "X over a rail".

The transport between Python and ELK is a subprocess to ``node`` running
the sibling :file:`elk_bridge.js` script. The graph is passed as JSON on
stdin; the laid-out result returns on stdout. ELK has no Python port that
implements the layered algorithm at this fidelity (as of 2026); this
indirection is the price of using the engine that production tools like
``netlistsvg`` rely on.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .types import (
    BoundingBox,
    ComponentPlacement,
    SchematicLayout,
    TerminalAnchor,
    Wire,
    WireEndpoint,
)


_BRIDGE_JS = Path(__file__).resolve().parent / "elk_bridge.js"

# Component box dimensions in ELK units (mm-equivalent). Picked to roughly
# match schemdraw's natural symbol size at SCALE=0.1.
_BOX_2TERM = (40.0, 20.0)
_BOX_3TERM = (40.0, 40.0)
_BOX_4TERM = (60.0, 50.0)

_LAYOUT_OPTIONS = {
    "elk.algorithm": "layered",
    "elk.direction": "RIGHT",
    "elk.edgeRouting": "ORTHOGONAL",
    "elk.layered.spacing.nodeNodeBetweenLayers": "40",
    "elk.spacing.nodeNode": "30",
    "elk.spacing.edgeNode": "15",
    "elk.spacing.edgeEdge": "10",
    "elk.layered.crossingMinimization.semiInteractive": "true",
}


def _require_node() -> str:
    """Locate the ``node`` binary; raise with an install hint if absent."""
    node = shutil.which("node")
    if node is None:
        raise ImportError(
            "pulsim.schematic ELK backend requires Node.js (and elkjs) on PATH. "
            "Install Node from https://nodejs.org or via your package manager, "
            "then run `npm install elkjs` in the pulsim repo root. "
            "(Or set PULSIM_SCHEMATIC_BACKEND=spring to fall back to the legacy layout.)"
        )
    return node


def _component_box(kind: str, num_terminals: int) -> tuple[float, float]:
    """Pick a sensible (width, height) ELK box for a component kind."""
    if num_terminals >= 4:
        return _BOX_4TERM
    if num_terminals >= 3:
        return _BOX_3TERM
    return _BOX_2TERM


def _port_sides_for(kind: str, nodes: list[int], ground_id: int) -> list[str]:
    """Return the ELK port side (``"WEST"``/``"EAST"``/``"NORTH"``/``"SOUTH"``)
    for each terminal of a component.

    Heuristics:
      - 2-terminal: terminal at ground → SOUTH; the other → NORTH. If
        neither is ground, WEST/EAST (in pin order) so the device sits
        horizontally in the flow.
      - 3-terminal (MOSFET/IGBT/vcswitch, pin order [ctrl, drain, source]):
        if source is ground → ctrl=WEST, drain=NORTH, source=SOUTH
        (vertical body with the gate driver coming in from the left).
        Otherwise → ctrl=WEST, drain=NORTH, source=EAST (T-shape).
      - 4-terminal (transformer, pin order [p1, p2, s1, s2]): p1 & p2
        WEST, s1 & s2 EAST.
    """
    n = len(nodes)
    if n == 2:
        a, b = int(nodes[0]), int(nodes[1])
        if a == ground_id:
            return ["SOUTH", "NORTH"]
        if b == ground_id:
            return ["NORTH", "SOUTH"]
        return ["WEST", "EAST"]
    if n == 3:
        # [gate/ctrl, drain/t1, source/t2]
        if int(nodes[2]) == ground_id:
            return ["WEST", "NORTH", "SOUTH"]
        return ["WEST", "NORTH", "EAST"]
    if n == 4:
        return ["WEST", "WEST", "EAST", "EAST"]
    return ["WEST"] * n


def _layer_constraint_for(comp, ground_id: int) -> str | None:
    """Map roles to ELK layer constraints to push the flow left-to-right.

    Voltage sources land in the FIRST (leftmost) layer; resistors whose
    other terminal is ground (loads) land in the LAST (rightmost) layer.
    """
    if comp.kind in ("voltage_source", "pwm_voltage_source",
                     "sine_voltage_source", "pulse_voltage_source"):
        return "FIRST"
    if comp.kind == "resistor" and len(comp.nodes) == 2:
        if int(comp.nodes[0]) == ground_id or int(comp.nodes[1]) == ground_id:
            return "LAST"
    return None


def _build_elk_graph(components: list, ground_id: int) -> dict[str, Any]:
    """Assemble the ELK JSON input for a circuit's component list."""
    children: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for comp in components:
        sides = _port_sides_for(comp.kind, list(comp.nodes), ground_id)
        width, height = _component_box(comp.kind, len(comp.nodes))
        ports = [
            {
                "id": f"{comp.name}::{idx}",
                "layoutOptions": {"port.side": sides[idx]},
            }
            for idx in range(len(comp.nodes))
        ]
        node_dict: dict[str, Any] = {
            "id": comp.name,
            "width": width,
            "height": height,
            "ports": ports,
            "properties": {"portConstraints": "FIXED_SIDE"},
        }
        layer = _layer_constraint_for(comp, ground_id)
        if layer is not None:
            node_dict["layoutOptions"] = {"elk.layered.layering.layerChoiceConstraint": "-1"}
            node_dict["properties"]["layerConstraint"] = layer
        children.append(node_dict)

    # Build edges: star pattern from the first component on each electrical
    # node (excludes nodes touched by only 1 terminal — those are open).
    node_to_terms: dict[int, list[tuple[str, int]]] = {}
    for comp in components:
        for idx, n in enumerate(comp.nodes):
            node_to_terms.setdefault(int(n), []).append((comp.name, idx))

    edge_counter = 0
    for node_id in sorted(node_to_terms.keys()):
        terms = node_to_terms[node_id]
        if len(terms) < 2:
            continue
        first_comp, first_term = terms[0]
        first_port = f"{first_comp}::{first_term}"
        for other_comp, other_term in terms[1:]:
            other_port = f"{other_comp}::{other_term}"
            edges.append({
                "id": f"e{edge_counter}",
                "sources": [first_port],
                "targets": [other_port],
            })
            edge_counter += 1

    return {
        "id": "root",
        "layoutOptions": _LAYOUT_OPTIONS,
        "children": children,
        "edges": edges,
    }


def _call_elk(graph: dict[str, Any]) -> dict[str, Any]:
    """Run :file:`elk_bridge.js` via Node and parse the JSON response."""
    node = _require_node()
    if not _BRIDGE_JS.exists():
        raise FileNotFoundError(
            f"elk_bridge.js missing at {_BRIDGE_JS} — was pulsim/schematic/ "
            "copied into the build tree?"
        )
    result = subprocess.run(
        [node, str(_BRIDGE_JS)],
        input=json.dumps(graph).encode("utf-8"),
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"elk_bridge.js failed (exit {result.returncode}): {stderr}"
        )
    if not result.stdout.strip():
        raise RuntimeError("elk_bridge.js produced no output")
    return json.loads(result.stdout)


def _parse_elk_output(
    result: dict[str, Any], components: list, ground_id: int
) -> SchematicLayout:
    """Convert ELK's laid-out JSON into our :class:`SchematicLayout`."""
    nodes_by_id = {child["id"]: child for child in result.get("children", [])}

    placements: dict[str, ComponentPlacement] = {}
    for comp in components:
        node = nodes_by_id.get(comp.name)
        if node is None:
            continue
        node_x = float(node.get("x", 0.0))
        node_y = float(node.get("y", 0.0))
        width = float(node.get("width", 40.0))
        height = float(node.get("height", 20.0))
        cx = node_x + width / 2.0
        cy = node_y + height / 2.0

        sides = _port_sides_for(comp.kind, list(comp.nodes), ground_id)
        ports_by_id = {p["id"]: p for p in node.get("ports", [])}
        anchors: list[TerminalAnchor] = []
        for idx, n in enumerate(comp.nodes):
            port = ports_by_id.get(f"{comp.name}::{idx}")
            if port is None:
                anchors.append(TerminalAnchor(index=idx, x=cx, y=cy, node=int(n)))
                continue
            px = node_x + float(port.get("x", 0.0))
            py = node_y + float(port.get("y", 0.0))
            anchors.append(TerminalAnchor(index=idx, x=px, y=py, node=int(n)))

        # Rotation derived from port sides:
        #   2-terminal NORTH/SOUTH (vertical) → 90
        #   2-terminal WEST/EAST (horizontal) → 0
        #   3-terminal vertical body (source SOUTH) → 90
        rotation = 0
        if len(comp.nodes) == 2:
            if {"NORTH", "SOUTH"} == set(sides):
                rotation = 90
        elif len(comp.nodes) >= 3:
            if "SOUTH" in sides:
                rotation = 90

        placements[comp.name] = ComponentPlacement(
            name=comp.name,
            kind=comp.kind,
            x=cx,
            y=cy,
            rotation=rotation,
            terminal_anchors=tuple(anchors),
        )

    # Wires from ELK edges: each edge has `sections` with start/end + bend
    # points. We translate (component::terminal_index) → WireEndpoint and
    # preserve the bend points in Wire.path so the renderer can draw the
    # orthogonal segments.
    wires: list[Wire] = []
    edge_source_points: dict[str, tuple[float, float]] = {}
    for edge in result.get("edges", []):
        src_ref = edge.get("sources", [None])[0]
        tgt_ref = edge.get("targets", [None])[0]
        if not src_ref or not tgt_ref:
            continue
        src_comp, src_term = src_ref.rsplit("::", 1)
        tgt_comp, tgt_term = tgt_ref.rsplit("::", 1)

        path_points: list[tuple[float, float]] = []
        for section in edge.get("sections", []):
            start = section.get("startPoint")
            end = section.get("endPoint")
            bends = section.get("bendPoints", []) or []
            pts = [start] + list(bends) + [end]
            for p in pts:
                if p is None:
                    continue
                path_points.append((float(p["x"]), float(p["y"])))
        if not path_points and src_comp in placements and tgt_comp in placements:
            src_anchor = placements[src_comp].terminal_anchors[int(src_term)]
            tgt_anchor = placements[tgt_comp].terminal_anchors[int(tgt_term)]
            path_points = [(src_anchor.x, src_anchor.y), (tgt_anchor.x, tgt_anchor.y)]

        if path_points:
            edge_source_points[src_ref] = path_points[0]

        wires.append(Wire(
            from_=WireEndpoint(component=src_comp, terminal=int(src_term)),
            to=WireEndpoint(component=tgt_comp, terminal=int(tgt_term)),
            path=tuple(path_points),
        ))

    # Junctions: any electrical node with 3+ component terminals is a
    # real electrical junction (a "T" or "+" in the schematic). Its
    # visible position is the shared source port — i.e. the start of the
    # star-pattern edges we emitted from that node's first terminal. Bend
    # points are NOT junctions, just wire turns; we explicitly do not
    # emit dots for those.
    node_to_terms: dict[int, list[tuple[str, int]]] = {}
    for comp in components:
        for idx, n in enumerate(comp.nodes):
            node_to_terms.setdefault(int(n), []).append((comp.name, idx))

    junctions_list: list[tuple[float, float]] = []
    for node_id in sorted(node_to_terms.keys()):
        terms = node_to_terms[node_id]
        if len(terms) < 3:
            continue
        first_comp, first_term = terms[0]
        src_ref = f"{first_comp}::{first_term}"
        if src_ref in edge_source_points:
            jx, jy = edge_source_points[src_ref]
            junctions_list.append((float(jx), float(jy)))
    junctions = tuple(junctions_list)

    # Canvas from the root bbox plus margin.
    root_w = float(result.get("width", 0.0))
    root_h = float(result.get("height", 0.0))
    margin = 10.0
    canvas = BoundingBox(
        x=-margin,
        y=-margin,
        width=root_w + 2 * margin,
        height=root_h + 2 * margin,
    )

    return SchematicLayout(
        components=placements,
        wires=tuple(wires),
        junctions=junctions,
        canvas=canvas,
    )


def compute_layout_elk(circuit) -> SchematicLayout:
    """Run the ELK Sugiyama layered backend on a Circuit.

    Args:
        circuit: Pulsim Circuit (uses ``components()``, ``ground()``,
            ``node_position_hint()``).

    Returns:
        A :class:`SchematicLayout` with components laid out left-to-right
        in layers, edges routed orthogonally, ground terminals on the
        SOUTH side of each device by default.

    Raises:
        ImportError: If Node.js is not on PATH.
        RuntimeError: If the ELK subprocess fails.
    """
    components = list(circuit.components())
    if not components:
        return SchematicLayout(
            components={},
            wires=(),
            junctions=(),
            canvas=BoundingBox(x=0.0, y=0.0, width=60.0, height=60.0),
        )
    ground_id = int(circuit.ground())
    graph = _build_elk_graph(components, ground_id)
    result = _call_elk(graph)
    return _parse_elk_output(result, components, ground_id)
