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


def _terminal_anchors(comp: _Comp,
                      center_x: float,
                      center_y: float,
                      rotation: int) -> list[TerminalAnchor]:
    """Compute terminal anchors for a component placed at
    (center_x, center_y) with the given rotation. Defaults to a
    symmetric layout: for 2-terminal devices pins are on the long
    axis; for 3-terminal MOSFET-like devices pin 0 (gate) extends
    sideways."""
    w, h = _body_size(comp.kind)
    half_w = w / 2.0
    half_h = h / 2.0
    n = len(comp.nodes)
    raw: list[tuple[float, float]]
    if n == 2:
        # Pin 0 top, pin 1 bottom (default orientation).
        raw = [(0.0, -half_h), (0.0, half_h)]
    elif n == 3:
        # [gate, drain, source] — gate west, drain north, source south.
        raw = [(-half_w, 0.0), (0.0, -half_h), (0.0, half_h)]
    elif n == 4:
        # [p1, p2, s1, s2] for transformers — west pair, east pair.
        raw = [
            (-half_w, -half_h / 1.5),
            (-half_w,  half_h / 1.5),
            ( half_w, -half_h / 1.5),
            ( half_w,  half_h / 1.5),
        ]
    else:
        # Generic: spread along the top edge.
        raw = []
        for i in range(n):
            t = (i + 0.5) / n
            raw.append((-half_w + t * w, -half_h))

    anchors: list[TerminalAnchor] = []
    for i, (dx, dy) in enumerate(raw):
        rx, ry = _rotate_offset(dx, dy, rotation)
        anchors.append(TerminalAnchor(
            index=i,
            x=center_x + rx,
            y=center_y + ry,
            node=int(comp.nodes[i]) if i < len(comp.nodes) else 0,
        ))
    return anchors


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

    for role, comp_name in role_to_name.items():
        comp = comps_by_name.get(comp_name)
        if comp is None:
            continue
        slot = template.slots.get(role)
        if slot is None:
            # Role not in the template — defer to the unrecognized
            # fallback (right-margin column) below.
            continue
        cx = slot.x_frac * template.canvas_width
        cy = slot.y_frac * template.canvas_height
        anchors = _terminal_anchors(comp, cx, cy, slot.rotation)
        placements[comp.name] = ComponentPlacement(
            name=comp.name,
            kind=comp.kind,
            x=cx,
            y=cy,
            rotation=slot.rotation,
            terminal_anchors=tuple(anchors),
        )

    # Unrecognized components — place along the right margin so the
    # template area stays uncluttered. Vertical column at the right
    # edge of the canvas.
    placed_names = set(placements.keys())
    unrecognized = [c for c in view.components if c.name not in placed_names]
    if unrecognized:
        x_margin = template.canvas_width + 30.0
        spacing = template.canvas_height / (len(unrecognized) + 1)
        for i, comp in enumerate(unrecognized):
            cy = (i + 1) * spacing
            anchors = _terminal_anchors(comp, x_margin, cy, 0)
            placements[comp.name] = ComponentPlacement(
                name=comp.name, kind=comp.kind,
                x=x_margin, y=cy, rotation=0,
                terminal_anchors=tuple(anchors),
            )

    # Emit wires from shared-node connectivity. For each node touched
    # by ≥ 2 component terminals, draw straight lines between the
    # terminal anchors in pairs (a tree, not a complete graph).
    wires: list[Wire] = []
    node_terminals: dict[int, list[tuple[str, int, float, float]]] = {}
    for placement in placements.values():
        for anchor in placement.terminal_anchors:
            node_terminals.setdefault(anchor.node, []).append(
                (placement.name, anchor.index, anchor.x, anchor.y))
    for node, terms in node_terminals.items():
        if len(terms) < 2:
            continue
        # Chain each terminal to the next — simple, deterministic.
        terms_sorted = sorted(terms, key=lambda t: (t[2], t[3]))
        for i in range(len(terms_sorted) - 1):
            a = terms_sorted[i]
            b = terms_sorted[i + 1]
            wires.append(Wire(
                from_=WireEndpoint(component=a[0], terminal=a[1]),
                to=WireEndpoint(component=b[0], terminal=b[1]),
                path=((a[2], a[3]), (b[2], b[3])),
            ))

    # Junctions: any node with 3+ terminals lands one junction at the
    # centroid of its anchors.
    junctions: list[tuple[float, float]] = []
    for node, terms in node_terminals.items():
        if len(terms) < 3:
            continue
        cx = sum(t[2] for t in terms) / len(terms)
        cy = sum(t[3] for t in terms) / len(terms)
        junctions.append((cx, cy))

    # Canvas — template's intrinsic size, plus enough room for the
    # right-margin overflow if any unrecognized components were placed.
    canvas_w = template.canvas_width
    if unrecognized:
        canvas_w = max(canvas_w, template.canvas_width + 60.0)
    canvas = BoundingBox(
        x=0.0, y=0.0,
        width=canvas_w, height=template.canvas_height,
        unit="mm",
    )

    return SchematicLayout(
        components=placements,
        wires=tuple(wires),
        junctions=tuple(junctions),
        canvas=canvas,
    )
