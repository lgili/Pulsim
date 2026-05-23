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

import types as _stdlib_types

from .types import (
    SchematicLayout,
    ComponentPlacement,
    Wire,
    WireEndpoint,
    TerminalAnchor,
    BoundingBox,
    SCHEMATIC_SCHEMA_VERSION,
)
from .layout import compute_layout as _compute_layout_impl
from .netlistsvg_backend import render_netlistsvg as _render_netlistsvg_impl
from .render import render as _render_impl, render_layout
from .templates import RecognizedTemplate, recognize_all


# -----------------------------------------------------------------------------
# Adapter — the flat-namespace `CircuitBuilder` returns `components()`
# entries as `dict` objects. The layout / render code in this module was
# written when components had attribute-style access (`comp.kind`,
# `comp.nodes`). Wrap the builder so both shapes work transparently.
# -----------------------------------------------------------------------------

class _CircuitAdapter:
    """Thin wrapper exposing the old attribute-style component API on top
    of the new flat-namespace ``CircuitBuilder``.

    The original archived schematic code (commits ``8e489e9``..``87dcd1e``)
    expected ``circuit.components()`` to yield objects with ``.kind``,
    ``.name``, ``.nodes``, ``.params`` attributes. The 1.0 binding returns
    a ``list[dict]`` for forward compatibility. This adapter converts each
    dict to a ``SimpleNamespace`` lazily and forwards the other methods
    the schematic engine calls on ``circuit`` (``ground``,
    ``node_position_hint``, ``position_hints``).
    """

    __slots__ = ("_inner",)

    def __init__(self, inner):
        self._inner = inner

    def components(self):
        return [
            _stdlib_types.SimpleNamespace(**c)
            for c in self._inner.components()
        ]

    def ground(self):
        g = self._inner.ground
        return g() if callable(g) else g

    def node_position_hint(self, node_id):
        return self._inner.node_position_hint(node_id)

    def position_hints(self):
        return self._inner.position_hints()

    # Pass-through any other attribute the renderer might touch.
    def __getattr__(self, name):
        return getattr(self._inner, name)


def _maybe_adapt(circuit):
    """Wrap a flat-namespace builder so the schematic engine sees the
    component-attribute interface. If the object already looks like the
    expected shape (e.g. a test fixture), it is returned unchanged."""
    if isinstance(circuit, _CircuitAdapter):
        return circuit
    comps = getattr(circuit, "components", None)
    if comps is None:
        return circuit
    try:
        sample = comps()
        if sample and isinstance(sample[0], dict):
            return _CircuitAdapter(circuit)
    except TypeError:
        # `components` is not callable on a non-builder — leave it alone.
        return circuit
    return circuit


def compute_layout(circuit):
    """Compute a SchematicLayout for ``circuit``.

    Forwards to the layout engine after wrapping the circuit in a
    SimpleNamespace adapter when needed.
    """
    return _compute_layout_impl(_maybe_adapt(circuit))


def render(circuit, path, **kwargs):
    """Render ``circuit`` to ``path`` (SVG or PNG)."""
    return _render_impl(_maybe_adapt(circuit), path, **kwargs)


def render_netlistsvg(circuit, *args, **kwargs):
    """netlistsvg backend wrapper — auto-adapts the builder."""
    return _render_netlistsvg_impl(
        _maybe_adapt(circuit), *args, **kwargs)

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
