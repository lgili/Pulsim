"""Schematic rendering and auto-layout for Pulsim circuits.

This is the Python-side surface of the ``add-schematic-rendering`` change.
Phase 2 (this module's first release) ships:

- :func:`compute_layout` — turn a ``Circuit`` into a canvas-agnostic
  :class:`SchematicLayout` (component placements, wires, junctions, canvas).
- Value types :class:`SchematicLayout`, :class:`ComponentPlacement`,
  :class:`Wire`, :class:`WireEndpoint`, :class:`TerminalAnchor`,
  :class:`BoundingBox` — JSON-serializable for GUI consumption.

The render layer (SVG/PNG via ``schemdraw``) lands in Phase 3.

Import contract:
  ``import pulsim.schematic`` succeeds even if optional deps (``networkx``,
  ``schemdraw``) are missing. The dependencies are imported lazily inside
  :func:`compute_layout` and render APIs, which then raise ``ImportError``
  with an explicit ``pip install pulsim[schematic]`` instruction.
"""

from __future__ import annotations

from .types import (
    SchematicLayout,
    ComponentPlacement,
    Wire,
    WireEndpoint,
    TerminalAnchor,
    BoundingBox,
    SCHEMATIC_SCHEMA_VERSION,
)
from .layout import compute_layout
from .netlistsvg_backend import render_netlistsvg
from .render import render, render_layout
from .templates import RecognizedTemplate, recognize_all

__all__ = [
    "compute_layout",
    "render",
    "render_layout",
    "render_netlistsvg",
    "recognize_all",
    "RecognizedTemplate",
    "SchematicLayout",
    "ComponentPlacement",
    "Wire",
    "WireEndpoint",
    "TerminalAnchor",
    "BoundingBox",
    "SCHEMATIC_SCHEMA_VERSION",
]
