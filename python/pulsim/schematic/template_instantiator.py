"""Template-based layout instantiator.

Bridges the recognizer (Tiers 1 + 2) with the renderer: given a
``RecognizedTopology`` and the user's circuit, load the matching YAML
template, map roles to actual components via ``role_map``, and emit a
:class:`SchematicLayout` whose component positions follow the
template's canvas-fraction slots.

Wires are emitted by walking the circuit graph: every shared node
becomes a star of straight lines between the component-terminals
landing on it.

For unrecognized components (components in the circuit whose name is
not in ``role_map``), the instantiator falls back to placing them
along the right margin so the rest of the layout stays template-
exact. A future pass may run force-directed within the unused canvas
region.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .topology_recognizer import RecognizedTopology, _CircuitView, _Comp
from .types import (
    BoundingBox,
    ComponentPlacement,
    SchematicLayout,
    TerminalAnchor,
    Wire,
    WireEndpoint,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Template loader
# ---------------------------------------------------------------------------

_TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass(frozen=True)
class Slot:
    """One slot in a template — where a role lands on the canvas."""
    role: str
    x_frac: float
    y_frac: float
    rotation: int = 0


@dataclass(frozen=True)
class Template:
    """A topology template loaded from disk."""
    name: str
    quality: str          # "polished" | "draft"
    canvas_width: float   # mm
    canvas_height: float
    slots: dict[str, Slot]   # role → Slot


def load_template(name: str) -> Optional[Template]:
    """Load the template for ``name`` from ``templates/<name>.yaml``.

    Returns ``None`` if the template file does not exist or PyYAML is
    not installed.
    """
    path = _TEMPLATE_DIR / f"{name}.yaml"
    if not path.exists():
        return None
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        logger.debug("PyYAML not installed; template loader disabled")
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("template %s could not be parsed: %s", name, exc)
        return None
    if not isinstance(data, dict):
        return None

    slots_raw = data.get("slots", {}) or {}
    slots: dict[str, Slot] = {}
    for role, spec in slots_raw.items():
        if not isinstance(spec, dict):
            continue
        slots[str(role)] = Slot(
            role=str(role),
            x_frac=float(spec.get("x", 0.5)),
            y_frac=float(spec.get("y", 0.5)),
            rotation=int(spec.get("rotation", 0)),
        )

    canvas = data.get("canvas", {}) or {}
    return Template(
        name=str(data.get("name", name)),
        quality=str(data.get("quality", "draft")),
        canvas_width=float(canvas.get("width", 200.0)),
        canvas_height=float(canvas.get("height", 120.0)),
        slots=slots,
    )


def list_available_templates() -> list[str]:
    """Names of every ``*.yaml`` in the templates directory (without
    the suffix)."""
    if not _TEMPLATE_DIR.exists():
        return []
    return sorted(p.stem for p in _TEMPLATE_DIR.glob("*.yaml"))


# ---------------------------------------------------------------------------
# Component geometry (very simple — assumes default symbol sizes)
# ---------------------------------------------------------------------------

# Default body dimensions in mm. Width/height before rotation —
# rotation rotates the box. Index by kind; falls back to (12, 18)
# for unknown kinds.
_BODY_SIZE: dict[str, tuple[float, float]] = {
    "resistor":            (16.0,  6.0),
    "capacitor":           ( 8.0,  6.0),
    "inductor":            (18.0,  8.0),
    "diode":               (12.0,  8.0),
    "nonlinear_diode":     (12.0,  8.0),
    "switch":              (14.0, 14.0),
    "mosfet_level1":       (14.0, 14.0),
    "igbt_level1":         (14.0, 14.0),
    "voltage_source":      (10.0, 16.0),
    "current_source":      (10.0, 16.0),
    "sine_voltage_source": (10.0, 16.0),
    "pwm_voltage_source":  (10.0, 16.0),
    "pulse_voltage_source": (10.0, 16.0),
    "vcvs":                (18.0, 14.0),
}
_DEFAULT_BODY_SIZE = (12.0, 18.0)


def _body_size(kind: str) -> tuple[float, float]:
    return _BODY_SIZE.get(kind, _DEFAULT_BODY_SIZE)


def _rotate_offset(dx: float, dy: float, rotation: int) -> tuple[float, float]:
    """Rotate (dx, dy) by `rotation` degrees CW (snap to {0,90,180,270})."""
    rot = rotation % 360
    if rot == 0:
        return (dx, dy)
    if rot == 90:
        return (-dy, dx)
    if rot == 180:
        return (-dx, -dy)
    if rot == 270:
        return (dy, -dx)
    # Non-orthogonal — fall back to no rotation.
    return (dx, dy)


# Side labels mapped under rotation (CW).
_ROTATE_SIDE: dict[int, dict[str, str]] = {
    0:   {"N": "N", "S": "S", "E": "E", "W": "W"},
    90:  {"N": "E", "E": "S", "S": "W", "W": "N"},
    180: {"N": "S", "S": "N", "E": "W", "W": "E"},
    270: {"N": "W", "W": "S", "S": "E", "E": "N"},
}


def _terminal_anchors_with_sides(
        comp: _Comp,
        center_x: float,
        center_y: float,
        rotation: int) -> tuple[list[TerminalAnchor], list[str]]:
    """Compute (anchor, side) pairs for each terminal of `comp`.

    Sides are one of "N", "S", "E", "W" — they tell the orthogonal
    wire router from which direction the wire should leave the body.
    Rotation rotates both the anchor coordinate and the side label.
    """
    w, h = _body_size(comp.kind)
    half_w = w / 2.0
    half_h = h / 2.0
    n = len(comp.nodes)
    raw: list[tuple[float, float, str]]
    if n == 2:
        # Default vertical: pin 0 north, pin 1 south.
        raw = [(0.0, -half_h, "N"), (0.0, half_h, "S")]
    elif n == 3:
        # [gate, drain, source]: gate west, drain north, source south.
        raw = [(-half_w, 0.0, "W"), (0.0, -half_h, "N"), (0.0, half_h, "S")]
    elif n == 4:
        # [p1, p2, s1, s2] transformer: west pair on left, east pair on right.
        raw = [
            (-half_w, -half_h / 1.5, "W"),
            (-half_w,  half_h / 1.5, "W"),
            ( half_w, -half_h / 1.5, "E"),
            ( half_w,  half_h / 1.5, "E"),
        ]
    else:
        raw = []
        for i in range(n):
            t = (i + 0.5) / n
            raw.append((-half_w + t * w, -half_h, "N"))

    rot_map = _ROTATE_SIDE.get(rotation % 360, _ROTATE_SIDE[0])
    anchors: list[TerminalAnchor] = []
    sides: list[str] = []
    for i, (dx, dy, side) in enumerate(raw):
        rx, ry = _rotate_offset(dx, dy, rotation)
        anchors.append(TerminalAnchor(
            index=i,
            x=center_x + rx,
            y=center_y + ry,
            node=int(comp.nodes[i]) if i < len(comp.nodes) else 0,
        ))
        sides.append(rot_map.get(side, side))
    return anchors, sides


# Back-compat wrapper for callers that only need positions.
def _terminal_anchors(comp: _Comp, cx: float, cy: float, rotation: int):
    return _terminal_anchors_with_sides(comp, cx, cy, rotation)[0]


# ---------------------------------------------------------------------------
# Orthogonal routing helpers
# ---------------------------------------------------------------------------

_SIDE_DIR: dict[str, tuple[int, int]] = {
    "N": ( 0, -1),
    "S": ( 0,  1),
    "E": ( 1,  0),
    "W": (-1,  0),
}
_STUB_LEN_MM: float = 3.0


def _orthogonal_l_path(
        ax: float, ay: float, a_side: str,
        bx: float, by: float, b_side: str) -> tuple[tuple[float, float], ...]:
    """L- or Z-shape polyline from terminal A to terminal B respecting
    each one's exit direction. Wire leaves each body perpendicular to
    its side, turns once or twice before arriving at the other end.
    """
    da = _SIDE_DIR.get(a_side, (0, -1))
    db = _SIDE_DIR.get(b_side, (0,  1))
    a_out = (ax + da[0] * _STUB_LEN_MM, ay + da[1] * _STUB_LEN_MM)
    b_out = (bx + db[0] * _STUB_LEN_MM, by + db[1] * _STUB_LEN_MM)
    a_axis = "v" if a_side in ("N", "S") else "h"
    b_axis = "v" if b_side in ("N", "S") else "h"

    if a_axis == "v" and b_axis == "v":
        mid_y = (a_out[1] + b_out[1]) / 2.0
        return (
            (ax, ay), a_out,
            (a_out[0], mid_y), (b_out[0], mid_y),
            b_out, (bx, by),
        )
    if a_axis == "h" and b_axis == "h":
        mid_x = (a_out[0] + b_out[0]) / 2.0
        return (
            (ax, ay), a_out,
            (mid_x, a_out[1]), (mid_x, b_out[1]),
            b_out, (bx, by),
        )
    if a_axis == "v":
        corner = (a_out[0], b_out[1])
    else:
        corner = (b_out[0], a_out[1])
    return ((ax, ay), a_out, corner, b_out, (bx, by))


def _vertical_stub(ax: float, ay: float, target_y: float
                   ) -> tuple[tuple[float, float], ...]:
    """Straight vertical line from (ax, ay) to (ax, target_y) — used
    to drop ground-touching terminals onto the bottom rail."""
    return ((ax, ay), (ax, target_y))


# ---------------------------------------------------------------------------
# Instantiator
# ---------------------------------------------------------------------------

def template_layout(
        circuit: Any,
        recognized: RecognizedTopology) -> Optional[SchematicLayout]:
    """Build a SchematicLayout by laying out ``circuit``'s components
    according to the template that matches ``recognized``.

    Returns ``None`` if:
      * No template exists for ``recognized.name``;
      * The template cannot be parsed (e.g. PyYAML missing);
      * The recognized ``role_map`` is empty (we need at least one
        anchor role to instantiate the layout sensibly).
    """
    template = load_template(recognized.name)
    if template is None:
        return None
    if not recognized.role_map:
        return None

    view = _CircuitView(circuit)
    comps_by_name = {c.name: c for c in view.components}

    # Component placements anchored by role.
    placements: dict[str, ComponentPlacement] = {}

    role_to_name: dict[str, str] = {}
    for comp_name, role in recognized.role_map.items():
        role_to_name[role] = comp_name

    # Per-terminal side annotation — used by the orthogonal router.
    # Map (component_name, terminal_index) → side ("N"/"S"/"E"/"W").
    sides_by_term: dict[tuple[str, int], str] = {}

    for role, comp_name in role_to_name.items():
        comp = comps_by_name.get(comp_name)
        if comp is None:
            continue
        slot = template.slots.get(role)
        if slot is None:
            continue
        cx = slot.x_frac * template.canvas_width
        cy = slot.y_frac * template.canvas_height
        anchors, sides = _terminal_anchors_with_sides(
            comp, cx, cy, slot.rotation)
        for a, side in zip(anchors, sides):
            sides_by_term[(comp.name, a.index)] = side
        placements[comp.name] = ComponentPlacement(
            name=comp.name,
            kind=comp.kind,
            x=cx,
            y=cy,
            rotation=slot.rotation,
            terminal_anchors=tuple(anchors),
        )

    # Unrecognized components — place along the right margin.
    placed_names = set(placements.keys())
    unrecognized = [c for c in view.components if c.name not in placed_names]
    if unrecognized:
        x_margin = template.canvas_width + 30.0
        spacing = template.canvas_height / (len(unrecognized) + 1)
        for i, comp in enumerate(unrecognized):
            cy = (i + 1) * spacing
            anchors, sides = _terminal_anchors_with_sides(
                comp, x_margin, cy, 0)
            for a, side in zip(anchors, sides):
                sides_by_term[(comp.name, a.index)] = side
            placements[comp.name] = ComponentPlacement(
                name=comp.name, kind=comp.kind,
                x=x_margin, y=cy, rotation=0,
                terminal_anchors=tuple(anchors),
            )

    # Collect (component, terminal, x, y, side, node) tuples grouped by node.
    node_terminals: dict[int, list[tuple[str, int, float, float, str]]] = {}
    for placement in placements.values():
        for anchor in placement.terminal_anchors:
            side = sides_by_term.get((placement.name, anchor.index), "S")
            node_terminals.setdefault(anchor.node, []).append(
                (placement.name, anchor.index,
                 anchor.x, anchor.y, side))

    ground_id = view.ground
    # Ground rail Y — fixed offset from canvas bottom.
    rail_y = template.canvas_height + 8.0

    wires: list[Wire] = []
    junctions: list[tuple[float, float]] = []

    # ---- Ground rail ---------------------------------------------------
    gnd_terms = node_terminals.get(ground_id, [])
    if gnd_terms:
        # Drop a vertical stub from each ground-touching anchor to the
        # rail; collect anchor x-coordinates for the horizontal rail.
        for name, idx, x, y, _ in gnd_terms:
            wires.append(Wire(
                from_=WireEndpoint(component=name, terminal=idx),
                to=WireEndpoint(component="<gnd-rail>", terminal=0),
                path=_vertical_stub(x, y, rail_y),
            ))
        if len(gnd_terms) >= 2:
            xs_sorted = sorted(t[2] for t in gnd_terms)
            wires.append(Wire(
                from_=WireEndpoint(component="<gnd-rail>", terminal=0),
                to=WireEndpoint(component="<gnd-rail>", terminal=1),
                path=((xs_sorted[0], rail_y), (xs_sorted[-1], rail_y)),
            ))

    # ---- Non-ground nets — orthogonal pair routing ---------------------
    for node, terms in node_terminals.items():
        if node == ground_id:
            continue
        if len(terms) < 2:
            continue
        # Deterministic order — sort by anchor coords.
        terms_sorted = sorted(terms, key=lambda t: (t[2], t[3]))
        for i in range(len(terms_sorted) - 1):
            a = terms_sorted[i]
            b = terms_sorted[i + 1]
            wires.append(Wire(
                from_=WireEndpoint(component=a[0], terminal=a[1]),
                to=WireEndpoint(component=b[0], terminal=b[1]),
                path=_orthogonal_l_path(a[2], a[3], a[4],
                                          b[2], b[3], b[4]),
            ))
        # Junction emission deferred to the post-pass that counts how
        # many wires actually share a geometric point.

    # Post-pass: emit junctions only at points where ≥ 3 distinct
    # wire endpoints land at the same geometric coordinate. Round to
    # 0.5 mm so floating-point jitter doesn't suppress real junctions.
    from collections import Counter
    point_count: Counter[tuple[float, float]] = Counter()
    for w in wires:
        for px, py in w.path:
            point_count[(round(px * 2) / 2.0, round(py * 2) / 2.0)] += 1
    for (px, py), count in point_count.items():
        if count >= 3:
            junctions.append((px, py))

    # Canvas — template's intrinsic size, plus extra room below for
    # the ground rail (8 mm + a margin), plus right-margin overflow
    # for any unrecognized components.
    canvas_w = template.canvas_width
    if unrecognized:
        canvas_w = max(canvas_w, template.canvas_width + 60.0)
    canvas_h = template.canvas_height + 15.0 if gnd_terms else template.canvas_height
    canvas = BoundingBox(
        x=0.0, y=0.0,
        width=canvas_w, height=canvas_h,
        unit="mm",
    )

    return SchematicLayout(
        components=placements,
        wires=tuple(wires),
        junctions=tuple(junctions),
        canvas=canvas,
    )
