"""Parser for the Pulsim analog skin SVG (``pulsim_analog.svg``).

The skin file uses the netlistsvg vocabulary [#]_: each top-level
``<g s:type="...">`` element is a reusable symbol template, with port
anchor points declared via inner ``<g s:x s:y s:pid="..."/>`` children
and a bounding box from the symbol's ``s:width`` / ``s:height``
attributes. Symbols are arranged side-by-side in the file (for visual
preview) via ``transform="translate(X, Y)"``; we strip that translation
on parse so each template's local coordinates start at (0, 0).

This module is the Phase 1 building block for ``native_backend.py`` —
it gives the SVG composer a quick ``dict[kind, SymbolTemplate]`` lookup
without forcing every render call to re-parse the 400-line skin file.
The parsed cache is keyed by the skin path's *resolved* absolute path so
a custom skin (via ``PULSIM_SCHEMATIC_SKIN``) gets its own cache slot.

.. [#] https://github.com/nturley/netlistsvg/blob/master/doc/AnalogSkin.md
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


# netlistsvg's skin namespace. Attributes inside the skin SVG that carry
# layout/anchor metadata live in this namespace (e.g. ``s:type``,
# ``s:x``, ``s:y``, ``s:pid``, ``s:width``, ``s:height``, ``s:alias``).
_SKIN_NS = "https://github.com/nturley/netlistsvg"
_SVG_NS = "http://www.w3.org/2000/svg"


@dataclass(frozen=True)
class SymbolTemplate:
    """One reusable symbol from the analog skin.

    Attributes:
        kind: The canonical ``s:type`` from the skin (e.g. ``"resistor_v"``,
            ``"voltage_source"``, ``"mosfet_n"``). The skin's ``<s:alias>``
            indirections are flattened by :func:`parse_skin` so a lookup
            with the alias key returns the same template.
        inner: A list of XML sub-elements to instantiate when placing this
            symbol. Already de-translated (top-level translate stripped)
            and namespace-clean. The renderer deep-copies these into its
            output document under a ``<g transform="...">`` per placement.
        ports: Mapping from port id (matching ``s:pid``) to the port's
            local (x, y) anchor coordinates. The renderer translates these
            by the placement's (x, y) to get absolute wire endpoints.
        width: Symbol bounding-box width from the skin's ``s:width``.
        height: Symbol bounding-box height from the skin's ``s:height``.
    """

    kind: str
    inner: tuple[ET.Element, ...] = field(default_factory=tuple)
    ports: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    width: float = 0.0
    height: float = 0.0


def _qname(local: str, ns: str = _SKIN_NS) -> str:
    """Build an ``{namespace}localname`` ElementTree-style qualified name."""
    return f"{{{ns}}}{local}"


def _parse_top_level_symbol(node: ET.Element) -> SymbolTemplate | None:
    """Turn one ``<g s:type="...">`` element into a :class:`SymbolTemplate`.

    Returns ``None`` if the element isn't a symbol declaration (missing
    ``s:type``) so the caller can skip non-symbol top-level groups (style
    tags, decorative elements, etc.).
    """
    kind = node.get(_qname("type"))
    if not kind:
        return None
    width = float(node.get(_qname("width"), "0") or 0)
    height = float(node.get(_qname("height"), "0") or 0)

    # Collect port anchors by scanning every descendant carrying s:pid.
    ports: dict[str, tuple[float, float]] = {}
    for desc in node.iter():
        pid = desc.get(_qname("pid"))
        if pid is None:
            continue
        try:
            x = float(desc.get(_qname("x"), "0") or 0)
            y = float(desc.get(_qname("y"), "0") or 0)
        except ValueError:
            continue
        ports[pid] = (x, y)

    # Build the inner content the renderer will instantiate. We deep-copy
    # every child OTHER than port anchors (those are metadata, not visible
    # geometry) and the ``<s:alias>`` declarations.
    inner: list[ET.Element] = []
    for child in list(node):
        # Skip s:alias declarations (renderer doesn't paint them).
        if child.tag == _qname("alias"):
            continue
        # Skip pure port-anchor groups (g containing only s:pid). The
        # netlistsvg skin sometimes nests both ports and geometry inside
        # the same <g>; only skip if this child is literally a port stub.
        if (
            child.tag == f"{{{_SVG_NS}}}g"
            and child.get(_qname("pid")) is not None
            and len(list(child)) == 0
        ):
            continue
        inner.append(child)

    return SymbolTemplate(
        kind=kind,
        inner=tuple(inner),
        ports=dict(ports),
        width=width,
        height=height,
    )


def _collect_aliases(node: ET.Element) -> tuple[str, ...]:
    """Return every alias declared inside a symbol element.

    Aliases come from ``<s:alias val="..."/>`` children. They're how
    netlistsvg's skin lets one ``<g s:type="resistor_v">`` be referenced
    by the short cell-type ``"r_v"`` in the input JSON.
    """
    out: list[str] = []
    for child in node.iter(_qname("alias")):
        val = child.get("val")
        if val:
            out.append(val)
    return tuple(out)


# Module-level cache so repeat calls with the same skin path don't re-parse.
# Keyed by the *resolved* absolute path string so symlinks and relative
# paths converge to the same entry.
_SKIN_CACHE: dict[str, dict[str, SymbolTemplate]] = {}


def parse_skin(svg_path: Path | str) -> dict[str, SymbolTemplate]:
    """Parse a netlistsvg-format analog skin into ``{kind/alias -> template}``.

    The same template object is registered under its primary ``s:type``
    AND every ``s:alias`` declared inside it, so callers can look up by
    either the long name (``"resistor_v"``) or the short alias (``"r_v"``).

    Args:
        svg_path: Path to the skin SVG file. Either the shipped
            ``pulsim_analog.svg`` or a user-supplied skin.

    Returns:
        A dict mapping every symbol key (primary type + aliases) to the
        same :class:`SymbolTemplate`. Lookup order is irrelevant — the
        primary kind and its aliases all return identical templates.

    Raises:
        FileNotFoundError: If ``svg_path`` does not exist.
        ET.ParseError: If the SVG file is malformed.
    """
    resolved = Path(svg_path).resolve()
    cache_key = str(resolved)
    cached = _SKIN_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not resolved.exists():
        raise FileNotFoundError(f"skin SVG not found at {resolved}")

    tree = ET.parse(resolved)
    root = tree.getroot()

    out: dict[str, SymbolTemplate] = {}
    for child in list(root):
        # Top-level symbols are <g s:type="..."> children of the SVG root.
        if child.tag != f"{{{_SVG_NS}}}g":
            continue
        template = _parse_top_level_symbol(child)
        if template is None:
            continue
        # Register under the primary s:type ...
        out[template.kind] = template
        # ... and under every alias the skin declares for it.
        for alias in _collect_aliases(child):
            out[alias] = template

    _SKIN_CACHE[cache_key] = out
    return out


def clear_skin_cache() -> None:
    """Drop the module-level skin cache.

    Test-only helper — production callers should never need this because
    skin files are immutable for the lifetime of a process. Useful when
    a test wants to assert a fresh parse of a freshly-written file.
    """
    _SKIN_CACHE.clear()
