"""Value types for the schematic-rendering capability.

Coordinate convention:
    Unit is millimeters. Origin is top-left of the canvas.
    +x is right, +y is down (matches SVG and most CAD tools).

All types are JSON-serializable via ``to_json()`` and round-trip via
``from_json()``. The top-level :class:`SchematicLayout` payload carries
a ``schema_version`` field (``"schematic-v1"``); :meth:`SchematicLayout.from_json`
rejects unknown versions with :class:`ValueError`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMATIC_SCHEMA_VERSION = "schematic-v1"


@dataclass(frozen=True)
class BoundingBox:
    """Canvas bounding box in millimeters.

    Attributes:
        x, y: Origin of the bounding box (top-left corner).
        width, height: Extent along +x and +y axes.
        unit: Always ``"mm"`` for schematic-v1.
    """

    x: float
    y: float
    width: float
    height: float
    unit: str = "mm"

    def to_json(self) -> dict[str, Any]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "width": float(self.width),
            "height": float(self.height),
            "unit": str(self.unit),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "BoundingBox":
        return cls(
            x=float(payload["x"]),
            y=float(payload["y"]),
            width=float(payload["width"]),
            height=float(payload["height"]),
            unit=str(payload.get("unit", "mm")),
        )


@dataclass(frozen=True)
class TerminalAnchor:
    """Where one terminal of a component connects to the canvas.

    Attributes:
        index: Terminal position in the component's pin order (0-based).
        x, y: Anchor coordinate in mm.
        node: Electrical node index this terminal belongs to.
    """

    index: int
    x: float
    y: float
    node: int

    def to_json(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "x": float(self.x),
            "y": float(self.y),
            "node": int(self.node),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "TerminalAnchor":
        return cls(
            index=int(payload["index"]),
            x=float(payload["x"]),
            y=float(payload["y"]),
            node=int(payload["node"]),
        )


@dataclass(frozen=True)
class ComponentPlacement:
    """Placement of one component on the canvas.

    Attributes:
        name: User-facing component name (matches ``Circuit.components()``).
        kind: Canonical kind string (e.g. ``"resistor"``, ``"mosfet"``).
        x, y: Center of the component graphic in mm.
        rotation: One of ``{0, 90, 180, 270}`` (degrees, CW from +x).
        terminal_anchors: Per-terminal anchor coordinates (pin order).
    """

    name: str
    kind: str
    x: float
    y: float
    rotation: int
    terminal_anchors: tuple[TerminalAnchor, ...] = field(default_factory=tuple)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "kind": str(self.kind),
            "x": float(self.x),
            "y": float(self.y),
            "rotation": int(self.rotation),
            "terminal_anchors": [a.to_json() for a in self.terminal_anchors],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "ComponentPlacement":
        anchors = tuple(
            TerminalAnchor.from_json(a) for a in payload.get("terminal_anchors", [])
        )
        return cls(
            name=str(payload["name"]),
            kind=str(payload["kind"]),
            x=float(payload["x"]),
            y=float(payload["y"]),
            rotation=int(payload["rotation"]),
            terminal_anchors=anchors,
        )


@dataclass(frozen=True)
class WireEndpoint:
    """One endpoint of a wire, identified by (component_name, terminal_index)."""

    component: str
    terminal: int

    def to_json(self) -> dict[str, Any]:
        return {"component": str(self.component), "terminal": int(self.terminal)}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "WireEndpoint":
        return cls(component=str(payload["component"]), terminal=int(payload["terminal"]))


@dataclass(frozen=True)
class Wire:
    """A wire connecting two component terminals along an explicit path.

    Phase 2 emits straight-line wires: ``path`` contains the two endpoint
    coordinates only. Manhattan routing is a follow-up change.

    Attributes:
        from_: Source endpoint. JSON key is ``"from"`` (Python's reserved
            ``from`` keyword prevents using it as a field name directly).
        to: Destination endpoint.
        path: Polyline as a list of ``[x, y]`` pairs in mm.
    """

    from_: WireEndpoint
    to: WireEndpoint
    path: tuple[tuple[float, float], ...] = field(default_factory=tuple)

    def to_json(self) -> dict[str, Any]:
        return {
            "from": self.from_.to_json(),
            "to": self.to.to_json(),
            "path": [[float(x), float(y)] for x, y in self.path],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "Wire":
        return cls(
            from_=WireEndpoint.from_json(payload["from"]),
            to=WireEndpoint.from_json(payload["to"]),
            path=tuple((float(x), float(y)) for x, y in payload.get("path", [])),
        )


@dataclass(frozen=True)
class SchematicLayout:
    """Top-level schematic layout payload.

    Carries every placed component, the wires connecting them, junction
    dots, and the canvas bounding box. Serializes to a single JSON
    document via :meth:`to_json`; round-trips via :meth:`from_json`.

    Attributes:
        components: Component name → :class:`ComponentPlacement`.
        wires: Tuple of wires drawn on the canvas.
        junctions: Coordinates where 3+ wire endpoints meet (mm).
        canvas: Bounding box covering all placements with margin.
        schema_version: Stable string ``"schematic-v1"``; bumped on
            breaking schema changes.
    """

    components: dict[str, ComponentPlacement]
    wires: tuple[Wire, ...]
    junctions: tuple[tuple[float, float], ...]
    canvas: BoundingBox
    schema_version: str = SCHEMATIC_SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "canvas": self.canvas.to_json(),
            "components": {
                name: placement.to_json() for name, placement in self.components.items()
            },
            "wires": [w.to_json() for w in self.wires],
            "junctions": [[float(x), float(y)] for x, y in self.junctions],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "SchematicLayout":
        version = str(payload.get("schema_version", ""))
        if version != SCHEMATIC_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schematic schema_version {version!r}; "
                f"expected {SCHEMATIC_SCHEMA_VERSION!r}"
            )
        components = {
            name: ComponentPlacement.from_json(p)
            for name, p in payload.get("components", {}).items()
        }
        wires = tuple(Wire.from_json(w) for w in payload.get("wires", []))
        junctions = tuple(
            (float(j[0]), float(j[1])) for j in payload.get("junctions", [])
        )
        canvas = BoundingBox.from_json(payload["canvas"])
        return cls(
            components=components,
            wires=wires,
            junctions=junctions,
            canvas=canvas,
            schema_version=version,
        )

    # -------------------------------------------------------------------
    # Jupyter inline display (Phase 5)
    # -------------------------------------------------------------------
    #
    # Returning a `SchematicLayout` from a Jupyter cell auto-renders
    # the schematic without a user-visible `.render()` call. The
    # implementation defers to `render.render_layout` so the inline
    # output matches the file-export output exactly.
    #
    # `_repr_svg_` is preferred (vector, zoomable, lightweight);
    # `_repr_html_` falls back to a `<pre>` describing the layout so
    # nothing breaks if `schemdraw` isn't installed.

    def _repr_svg_(self) -> str | None:
        try:
            from .render import render_layout
        except ImportError:
            return None
        import tempfile
        from pathlib import Path
        try:
            with tempfile.NamedTemporaryFile(
                    "w", suffix=".svg", delete=False) as fh:
                tmp = Path(fh.name)
            try:
                render_layout(self, tmp, format="svg")
                return tmp.read_text(encoding="utf-8")
            finally:
                try:
                    tmp.unlink()
                except OSError:
                    pass
        except Exception:
            return None

    def _repr_html_(self) -> str:
        svg = self._repr_svg_()
        if svg:
            return svg
        # Fallback: brief tabular summary so the Jupyter cell still
        # shows *something* when schemdraw isn't installed.
        rows = "".join(
            f"<tr><td>{name}</td><td>{p.kind}</td>"
            f"<td>{p.x:.1f}</td><td>{p.y:.1f}</td>"
            f"<td>{p.rotation}°</td></tr>"
            for name, p in sorted(self.components.items())
        )
        return (
            "<table style='border-collapse:collapse;font-family:monospace;'>"
            "<thead><tr style='border-bottom:1px solid #999;'>"
            "<th>name</th><th>kind</th><th>x (mm)</th>"
            "<th>y (mm)</th><th>rot</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
            "<p style='font-size:0.85em;color:#666;'>"
            "Install <code>pip install pulsim[schematic]</code> for "
            "inline SVG rendering.</p>"
        )
