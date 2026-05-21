"""Pure-Python schematic rendering backend.

Replaces the ``netlistsvg`` Node subprocess for the SVG-composition step
while reusing ``elk_bridge.js`` (via :mod:`elk_backend`) for the layout
step. The output SVG is composed by:

  1. Parsing the Pulsim analog skin (see :mod:`skin_parser`) into a
     ``{kind -> SymbolTemplate}`` cache. Each template carries the symbol
     geometry (paths/lines/text) and port anchor coordinates.
  2. Translating Pulsim canonical component kinds (``"resistor"``,
     ``"mosfet"``, …) to skin alias keys (``"r_v"``, ``"mosfet_n"``, …)
     and looking up each template.
  3. Building an ELK layered graph with one node per component (sized
     from the skin's ``s:width`` / ``s:height``) and one edge per
     non-degenerate electrical net. Calls ``node elk_bridge.js`` to get
     back the laid-out graph with cell positions and orthogonal wire
     bend points.
  4. Composing the output SVG: one ``<g transform="translate(x, y)">``
     per cell wrapping a deep-copy of the skin template's inner XML;
     one ``<path>`` per wire using ELK's section bend points; one
     ``<text>`` label per component.

The whole pipeline runs without invoking ``netlistsvg``. The only Node
subprocess is ``elk_bridge.js`` (the existing ELK backend bridge). When
PR #6's ``netlistsvg`` backend is deprecated and removed, the only Node
dependency for the schematic stack is ``elkjs`` itself.

This is **Phase 1** of the ``add-python-schematic-renderer`` change.
Position hints (Phase 2/3) and topology-aware auto-layouts (Phase 4)
build on this file but are intentionally NOT implemented here — the
goal of Phase 1 is bit-equivalent-quality output to the existing
netlistsvg backend on the canonical demo set (rc, buck, half-bridge).
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .skin_parser import SymbolTemplate, parse_skin


_SVG_NS = "http://www.w3.org/2000/svg"
_XLINK_NS = "http://www.w3.org/1999/xlink"
_SKIN_NS = "https://github.com/nturley/netlistsvg"
ET.register_namespace("", _SVG_NS)
ET.register_namespace("xlink", _XLINK_NS)


# Default skin shipped alongside this module. Honors PULSIM_SCHEMATIC_SKIN
# at module load time (the env var is read every render call so a test
# can toggle it mid-process).
_BUILTIN_SKIN = Path(__file__).resolve().parent / "skin" / "pulsim_analog.svg"
_BRIDGE_JS = Path(__file__).resolve().parent / "elk_bridge.js"


# Pulsim canonical kind → skin alias / s:type. Same table as
# ``netlistsvg_backend._CELL_TYPE`` — we keep an independent copy so the
# two backends can drift if one needs a different symbol choice (e.g. a
# native-only rendering of inductor_v instead of inductor_h).
_KIND_TO_SKIN: dict[str, str] = {
    "resistor":             "r_v",
    "capacitor":            "c_v",
    "inductor":             "l_h",
    "voltage_source":       "v",
    "pwm_voltage_source":   "v",
    "sine_voltage_source":  "v",
    "pulse_voltage_source": "v",
    "current_source":       "i",
    "diode":                "d_h",
    "transformer":          "transformer_1p_1s",
    "mosfet":               "mosfet_n",
    "igbt":                 "igbt",
    "vcswitch":             "vcswitch",
}

# Per skin-type, the port-id order matching Pulsim's terminal pin order.
# ``Circuit.components()`` returns nodes in the order they were added via
# ``add_*``; the renderer uses this table to map terminal index → skin
# port id so wire endpoints land at the right anchor.
_SKIN_PORT_ORDER: dict[str, tuple[str, ...]] = {
    "r_v":               ("A", "B"),
    "c_v":               ("A", "B"),
    "l_h":               ("A", "B"),
    "v":                 ("+", "-"),
    "i":                 ("+", "-"),
    "d_h":               ("+", "-"),
    "transformer_1p_1s": ("L1.1", "L1.2", "L2.1", "L2.2"),
    "mosfet_n":          ("G", "D", "S"),
    "mosfet_p":          ("G", "D", "S"),
    "igbt":              ("G", "C", "E"),
    "vcswitch":          ("ctrl", "t1", "t2"),
    # 1-port ground stamp: one terminal connecting to gnd.
    "gnd":               ("A",),
}

# Generic fallback for unknown kinds — rendered as a labeled rectangle
# from the skin's ``generic`` symbol.
_GENERIC_KIND = "generic"


def _resolve_skin_path() -> Path:
    """Return the path to the skin SVG, honoring ``PULSIM_SCHEMATIC_SKIN``."""
    import os
    env = os.environ.get("PULSIM_SCHEMATIC_SKIN")
    if env:
        return Path(env).resolve()
    return _BUILTIN_SKIN


def _require_node() -> str:
    """Locate the ``node`` binary; raise with an install hint if absent."""
    node = shutil.which("node")
    if node is None:
        raise ImportError(
            "pulsim.schematic native backend requires Node.js (and elkjs) on PATH "
            "for the layout step. Install Node from https://nodejs.org, then run "
            "`npm install elkjs` at the repo root. "
            "(The renderer itself is pure Python; only the layout call needs Node.)"
        )
    return node


# ---------------------------------------------------------------------------
# Layout step: Pulsim Circuit → ELK graph → laid-out ELK graph
# ---------------------------------------------------------------------------


def _port_side_for(
    terminal_index: int,
    total_terms: int,
    nodes: list[int],
    ground_id: int,
) -> str:
    """Pick an ELK ``port.side`` for a component terminal.

    Mirrors :func:`elk_backend._port_sides_for` but kept in-module so the
    native backend can evolve its own port-placement heuristics without
    affecting the legacy ELK backend.
    """
    if total_terms == 2:
        a, b = int(nodes[0]), int(nodes[1])
        if a == ground_id:
            return ("SOUTH", "NORTH")[terminal_index]
        if b == ground_id:
            return ("NORTH", "SOUTH")[terminal_index]
        return ("WEST", "EAST")[terminal_index]
    if total_terms == 3:
        # [gate/ctrl, drain/t1, source/t2]
        if int(nodes[2]) == ground_id:
            return ("WEST", "NORTH", "SOUTH")[terminal_index]
        return ("WEST", "NORTH", "EAST")[terminal_index]
    if total_terms == 4:
        return ("WEST", "WEST", "EAST", "EAST")[terminal_index]
    return "WEST"


# Grid metric used by ``_resolve_hints`` to translate semantic
# `(layer, slot)` hints into absolute coordinates that ELK / the
# renderer can consume. Picked so a typical buck-converter layout
# (~6 layers × 4 slots) fits comfortably on screen.
LAYER_PX = 120.0
SLOT_PX  = 80.0


def _resolve_hints(circuit: Any) -> dict[str, tuple[float, float]]:
    """Translate every position hint on ``circuit`` into absolute coords.

    Returns a dict mapping component name -> ``(x, y)`` in renderer
    units. The translation rule (in priority order):

      1. If both `(x, y)` and `(layer, slot)` are set, absolute coords
         win (the layer/slot are silently ignored — the user explicitly
         pinned an absolute position).
      2. `(x, y)` only → passed through.
      3. `(layer, slot)` only → ``(layer * LAYER_PX, slot * SLOT_PX)``.
      4. Partial semantic hints (only `layer` or only `slot`) → the
         missing axis defaults to 0.

    Two hints resolving to the same absolute coords are reported as a
    deterministic error so the user gets a clear message instead of
    overlapping symbols in the rendered SVG.
    """
    if not hasattr(circuit, "position_hints"):
        return {}
    raw = circuit.position_hints()
    if not raw:
        return {}
    resolved: dict[str, tuple[float, float]] = {}
    seen_coords: dict[tuple[float, float], str] = {}
    for name, hint in raw.items():
        # Absolute coords (with or without semantic) take priority.
        if hint.x is not None and hint.y is not None:
            xy = (float(hint.x), float(hint.y))
        elif hint.layer is not None or hint.slot is not None:
            layer = float(hint.layer) if hint.layer is not None else 0.0
            slot = float(hint.slot) if hint.slot is not None else 0.0
            xy = (layer * LAYER_PX, slot * SLOT_PX)
        else:
            # Malformed hint (kernel would have rejected this, but be
            # defensive — `position_hints()` is a snapshot and could in
            # principle disagree with `set_position`'s invariant).
            continue
        if xy in seen_coords:
            raise ValueError(
                f"Position hint conflict: components {seen_coords[xy]!r} "
                f"and {name!r} both resolve to {xy}. Adjust one of the "
                f"hints so they land on distinct grid cells."
            )
        seen_coords[xy] = name
        resolved[name] = xy
    return resolved


def _build_elk_graph(
    components: list,
    ground_id: int,
    skin: dict[str, SymbolTemplate],
    position_hints: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Assemble the ELK JSON input for a circuit's component list.

    Uses skin-derived widths/heights so the layout reflects real symbol
    geometry — wires from ELK land at the symbol's actual port anchors
    rather than a generic box edge.

    When ``position_hints`` is non-empty, each hinted cell gets pinned
    via ELK's ``org.eclipse.elk.position`` constraint plus an explicit
    ``x, y`` in the input JSON. Un-hinted cells float and ELK places
    them normally; the layered algorithm honors the constraints when
    routing wires, so the result stays consistent.
    """
    hints = position_hints or {}
    children: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for comp in components:
        skin_key = _KIND_TO_SKIN.get(comp.kind, _GENERIC_KIND)
        template = skin.get(skin_key) or skin.get(_GENERIC_KIND)
        if template is None:
            # Skin doesn't even have `generic` — pick conservative dims.
            width, height = 40.0, 40.0
        else:
            # Pad slightly so the symbol doesn't kiss the port anchor.
            width = template.width
            height = template.height

        n_terms = len(comp.nodes)
        ports = [
            {
                "id": f"{comp.name}::{idx}",
                "layoutOptions": {
                    "port.side": _port_side_for(idx, n_terms, list(comp.nodes), ground_id),
                },
            }
            for idx in range(n_terms)
        ]
        node_dict: dict[str, Any] = {
            "id": comp.name,
            "width": width,
            "height": height,
            "ports": ports,
            "properties": {"portConstraints": "FIXED_SIDE"},
        }

        # NOTE: Layer constraints (sources→FIRST, ground-loads→LAST) work
        # for pure DAG topologies but ELK rejects them when a "FIRST"-pinned
        # node has incoming edges from non-FIRST_SEPARATE nodes (the case
        # for any closed loop, e.g. a buck converter's input cap). Phase 4
        # of `add-python-schematic-renderer` introduces topology-aware
        # auto-hints, which is the right place to express these priors; for
        # Phase 1 we let ELK choose layers freely.

        # Position hints (Phase 3): pin this cell to absolute coords if
        # the user supplied a hint. ELK's "interactive" layered mode
        # honors `org.eclipse.elk.position` per node and re-routes wires
        # around the pinned positions.
        hinted = hints.get(comp.name)
        if hinted is not None:
            hx, hy = hinted
            node_dict["x"] = hx
            node_dict["y"] = hy
            node_dict["layoutOptions"] = {
                "org.eclipse.elk.position": f"({hx},{hy})",
            }

        children.append(node_dict)

    # Build edges as a star pattern from the first terminal on each node.
    node_to_terms: dict[int, list[tuple[str, int]]] = {}
    for comp in components:
        for idx, node_id in enumerate(comp.nodes):
            node_to_terms.setdefault(int(node_id), []).append((comp.name, idx))

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

    layout_options: dict[str, str] = {
        "elk.algorithm": "layered",
        "elk.direction": "RIGHT",
        "elk.edgeRouting": "ORTHOGONAL",
        "elk.layered.spacing.nodeNodeBetweenLayers": "40",
        "elk.spacing.nodeNode": "30",
        "elk.spacing.edgeNode": "15",
        "elk.spacing.edgeEdge": "10",
        "elk.layered.crossingMinimization.semiInteractive": "true",
    }
    if hints:
        # When the user supplied position hints, switch ELK to its
        # INTERACTIVE strategy chain so it honors per-node `position`
        # values instead of computing its own layering / cycle breaking.
        # Un-hinted nodes are still placed by ELK; the interactive
        # strategies just respect anchors when present.
        layout_options.update({
            "elk.layered.cycleBreaking.strategy": "INTERACTIVE",
            "elk.layered.layering.strategy": "INTERACTIVE",
            "elk.layered.crossingMinimization.strategy": "INTERACTIVE",
        })

    return {
        "id": "root",
        "layoutOptions": layout_options,
        "children": children,
        "edges": edges,
    }


def _apply_position_hints(
    laid_out: dict[str, Any],
    hints: dict[str, tuple[float, float]],
    components: list,
    skin: dict[str, SymbolTemplate],
) -> None:
    """Override ELK's cell positions with user hints, in-place.

    ELK's layered algorithm doesn't snap to `org.eclipse.elk.position`
    constraints (its layering pass overrides them). To honor user hints
    deterministically we:

      1. Walk `laid_out["children"]` and rewrite `(x, y)` for any hinted
         cell to the user's resolved absolute coords.
      2. Rebuild each cell's port positions inside that hinted cell so
         downstream wire-endpoint calculations land on the right pin.
      3. Replace every edge with at least one hinted endpoint cell with
         a simple two-segment L-route from one port to the other (one
         horizontal + one vertical segment). The result isn't as crossing-
         optimal as ELK's full route, but it's deterministic, never
         intersects the symbols, and matches the user's spatial intent.

    Edges whose both endpoints are un-hinted are left alone — ELK already
    routed them well around the floating cells.
    """
    # Build a lookup of component → skin port-id order so we can
    # translate terminal indices to skin-port (x, y) anchors after
    # repositioning.
    name_to_kind: dict[str, str] = {}
    for comp in components:
        name_to_kind[comp.name] = _KIND_TO_SKIN.get(comp.kind, _GENERIC_KIND)

    # 1. Override cell positions for every hinted cell.
    for child in laid_out.get("children", []):
        cid = child.get("id")
        if cid not in hints:
            continue
        new_x, new_y = hints[cid]
        child["x"] = new_x
        child["y"] = new_y

    # 2. For every edge, if EITHER endpoint cell was hinted, recompute
    #    a simple L-route from the source port to the target port.
    hinted_cells = set(hints.keys())
    cells_by_id = {c["id"]: c for c in laid_out.get("children", [])}
    for edge in laid_out.get("edges", []):
        sources = edge.get("sources", [])
        targets = edge.get("targets", [])
        if not sources or not targets:
            continue
        src_ref = sources[0]
        tgt_ref = targets[0]
        try:
            src_cell, _ = src_ref.rsplit("::", 1)
            tgt_cell, _ = tgt_ref.rsplit("::", 1)
        except ValueError:
            continue
        if src_cell not in hinted_cells and tgt_cell not in hinted_cells:
            continue

        # Resolve absolute port positions for both endpoints.
        src_abs = _resolve_port_abs(src_ref, cells_by_id, name_to_kind, skin)
        tgt_abs = _resolve_port_abs(tgt_ref, cells_by_id, name_to_kind, skin)
        if src_abs is None or tgt_abs is None:
            continue

        sx, sy = src_abs
        tx, ty = tgt_abs
        # L-route via an intermediate elbow point. Pick horizontal-first
        # if the cells share a row, vertical-first otherwise.
        if abs(sy - ty) < 1e-6:
            bend_points: list[dict[str, float]] = []
        elif abs(sx - tx) < 1e-6:
            bend_points = []
        else:
            bend_points = [{"x": tx, "y": sy}]

        edge["sections"] = [
            {
                "startPoint": {"x": sx, "y": sy},
                "endPoint":   {"x": tx, "y": ty},
                "bendPoints": bend_points,
            }
        ]
        # Drop ELK-computed junction points — they were tied to the old
        # bend geometry; the L-route doesn't need them.
        edge.pop("junctionPoints", None)


def _resolve_port_abs(
    port_ref: str,
    cells_by_id: dict[str, dict[str, Any]],
    name_to_kind: dict[str, str],
    skin: dict[str, SymbolTemplate],
) -> tuple[float, float] | None:
    """Translate `"<cell>::<index>"` into an absolute (x, y) port anchor.

    Combines the cell's `(x, y)` from `cells_by_id` with the skin port
    anchor (looked up by terminal index → skin port id via
    `_SKIN_PORT_ORDER`). Returns ``None`` if any piece is missing.
    """
    try:
        cell_name, idx_str = port_ref.rsplit("::", 1)
        term_idx = int(idx_str)
    except ValueError:
        return None
    cell = cells_by_id.get(cell_name)
    if cell is None:
        return None
    skin_key = name_to_kind.get(cell_name, _GENERIC_KIND)
    template = skin.get(skin_key) or skin.get(_GENERIC_KIND)
    if template is None:
        return None
    port_order = _SKIN_PORT_ORDER.get(skin_key, ())
    if term_idx >= len(port_order):
        return None
    anchor = template.ports.get(port_order[term_idx])
    if anchor is None:
        return None
    return (
        float(cell.get("x", 0)) + anchor[0],
        float(cell.get("y", 0)) + anchor[1],
    )


def _call_elk(graph: dict[str, Any]) -> dict[str, Any]:
    """Run ``elk_bridge.js`` via Node and parse the JSON response.

    Identical contract to :func:`elk_backend._call_elk`; copied here so
    the native backend doesn't reach into the legacy ELK backend's
    private helpers.
    """
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


# ---------------------------------------------------------------------------
# Composition step: laid-out ELK graph + skin → SVG ElementTree
# ---------------------------------------------------------------------------


def _make_root_svg(width: float, height: float) -> ET.Element:
    """Create the top-level ``<svg>`` element with viewBox + namespaces."""
    # Pad the canvas a bit so symbols at the edge don't kiss the border.
    pad = 20.0
    svg_width = width + 2 * pad
    svg_height = height + 2 * pad
    svg = ET.Element(
        f"{{{_SVG_NS}}}svg",
        {
            "width": f"{svg_width:.1f}",
            "height": f"{svg_height:.1f}",
            "viewBox": f"{-pad:.1f} {-pad:.1f} {svg_width:.1f} {svg_height:.1f}",
        },
    )
    # Inline a minimal stylesheet so component lines/text render legibly
    # even when the host page has no global SVG styles.
    style = ET.SubElement(svg, f"{{{_SVG_NS}}}style")
    style.text = (
        "svg { stroke: #000; fill: none; }\n"
        "text { fill: #000; stroke: none; "
        "font-size: 10px; font-family: 'Courier New', monospace; }\n"
        ".symbol { stroke-width: 1.5; }\n"
        ".connect { stroke-width: 1.5; }\n"
        ".wire { stroke: #000; stroke-width: 1.0; fill: none; }\n"
        ".junction { fill: #000; }\n"
    )
    return svg


def _instantiate_symbol(
    template: SymbolTemplate,
    x: float,
    y: float,
    *,
    ref: str | None = None,
    value: str | None = None,
    show_node_label: bool = False,
) -> ET.Element:
    """Build a ``<g transform="translate(x,y)">`` wrapping a deep-copy of
    the skin template's inner geometry.

    The skin uses placeholder ``<text>`` elements tagged with
    ``s:attribute="ref"`` / ``"value"`` / ``"name"`` to mark where the
    component identifier and value should be substituted at render time.
    netlistsvg does this substitution itself; we replicate the behavior
    here so our output doesn't ship literal ``X1`` / ``Xk`` strings.

    Args:
        template: Skin symbol template to instantiate.
        x: Translation X (laid-out cell origin from ELK).
        y: Translation Y (laid-out cell origin from ELK).
        ref: Replacement for ``s:attribute="ref"`` text nodes (component
            name). ``None`` keeps the skin's placeholder.
        value: Replacement for ``s:attribute="value"`` text nodes (the
            human-formatted device value, e.g. ``"1k"``). ``None`` clears
            the placeholder so the rendered symbol stays uncluttered when
            we don't have a value (e.g. diode, ground stub).
        show_node_label: If False, drop any ``s:attribute="name"`` text
            (used by the gnd / vcc / vee symbols for the rail label —
            usually noisy in a per-device-gnd-stub rendering).
    """
    g = ET.Element(
        f"{{{_SVG_NS}}}g",
        {"transform": f"translate({x:.2f}, {y:.2f})"},
    )
    for child in template.inner:
        # Deep-copy so multiple instances don't share mutable nodes, and
        # so the skin's namespace declarations don't pollute the output.
        clone = copy.deepcopy(child)
        drop = _substitute_placeholders(
            clone,
            ref=ref,
            value=value,
            show_node_label=show_node_label,
        )
        if drop:
            # Placeholder slot the caller chose not to populate; skip
            # this child entirely so the rendered SVG has no leftover
            # ``X1`` / ``Xk`` / ``name`` literal.
            continue
        _strip_skin_namespace(clone)
        g.append(clone)
    return g


def _substitute_placeholders(
    node: ET.Element,
    *,
    ref: str | None,
    value: str | None,
    show_node_label: bool,
) -> bool:
    """Replace ``<text s:attribute="...">`` placeholders in-place.

    Walks every descendant looking for ``s:attribute`` text nodes. For
    each, swap the placeholder text for the user-supplied value (or
    flag the element for removal if the corresponding argument is
    ``None``). The element passed in may itself be a placeholder (when
    the skin's symbol group lists ``<text s:attribute=...>`` as a direct
    child), so the check applies to ``node`` AND its descendants.

    Returns ``True`` if ``node`` itself should be removed by the caller
    (i.e. ``node`` IS a placeholder whose substitution argument was
    ``None``). The caller is responsible for actually removing it from
    its parent — ElementTree doesn't expose parent links.
    """
    attr_qname = f"{{{_SKIN_NS}}}attribute"

    def _classify(elem: ET.Element) -> tuple[str | None, bool]:
        """Return ``(new_text, remove_self)`` for one element.

        ``new_text`` is the replacement for ``elem.text`` when not None;
        ``remove_self`` is True when the element should be dropped.
        """
        attr = elem.get(attr_qname)
        if attr == "ref":
            if ref is not None:
                return (ref, False)
            return (None, True)
        if attr == "value":
            if value is not None:
                return (value, False)
            return (None, True)
        if attr == "name":
            if show_node_label and ref is not None:
                return (ref, False)
            return (None, True)
        return (None, False)

    # Handle the root node first.
    new_text, root_remove = _classify(node)
    if new_text is not None:
        node.text = new_text

    # Then descendants — collect those to drop, then prune.
    to_remove: list[ET.Element] = []
    for child in list(node):
        _walk_and_collect(child, to_remove, _classify)

    if to_remove:
        _remove_descendants(node, set(map(id, to_remove)))

    return root_remove


def _walk_and_collect(
    elem: ET.Element,
    to_remove: list[ET.Element],
    classify,
) -> None:
    """Apply ``classify`` to ``elem`` and recurse into its children."""
    new_text, remove_self = classify(elem)
    if remove_self:
        to_remove.append(elem)
        # Don't recurse into a removed subtree.
        return
    if new_text is not None:
        elem.text = new_text
    for child in list(elem):
        _walk_and_collect(child, to_remove, classify)


def _remove_descendants(root: ET.Element, target_ids: set[int]) -> None:
    """Remove every descendant whose ``id()`` is in ``target_ids``."""
    for child in list(root):
        if id(child) in target_ids:
            root.remove(child)
            continue
        _remove_descendants(child, target_ids)


def _strip_skin_namespace(node: ET.Element) -> None:
    """Remove ``s:`` namespace attributes from a copied symbol element.

    The skin namespace (``https://github.com/nturley/netlistsvg``) is
    metadata only. After parsing it serves no purpose in the rendered
    SVG, and serializers add it as a noisy ``xmlns:s="..."`` to every
    output. Strip in-place.
    """
    attrs_to_drop = [k for k in node.attrib if k.startswith(f"{{{_SKIN_NS}}}")]
    for k in attrs_to_drop:
        del node.attrib[k]
    for child in list(node):
        _strip_skin_namespace(child)


def _format_value(comp: Any) -> str | None:
    """Format a Pulsim component's primary parameter for the SVG label.

    Returns ``None`` for components with no canonical value (diode, the
    various switching devices). The skin's ``s:attribute="value"`` slot
    is dropped when we return ``None`` so the rendered symbol stays
    visually uncluttered.
    """
    params = dict(comp.params) if hasattr(comp, "params") else {}
    # Canonical short symbol per device family — same convention as
    # netlistsvg_backend._CELL_TYPE keys and ComponentDescriptor.params
    # in runtime_circuit.hpp.
    key = {
        "resistor":             "R",
        "capacitor":            "C",
        "inductor":             "L",
        "voltage_source":       "V",
        "pwm_voltage_source":   "V",
        "sine_voltage_source":  "V",
        "pulse_voltage_source": "V",
        "current_source":       "I",
    }.get(comp.kind)
    if key is None or key not in params:
        return None
    val = params[key]
    return _fmt_eng(val, key)


def _fmt_eng(value: float, symbol: str) -> str:
    """Format ``value`` with an engineering-notation prefix + unit.

    Examples (symbol = "R"): ``1000 -> "1kΩ"``, ``4.7e-6 -> "4.7µF"``
    (symbol="C"). Falls back to ``"{value:g}{symbol}"`` for tiny / huge
    out-of-range values. The unit map keeps this contained — adding a
    new device family with its own unit is a one-line edit.
    """
    units = {"R": "Ω", "C": "F", "L": "H", "V": "V", "I": "A"}
    unit = units.get(symbol, "")
    if value == 0 or not (1e-15 <= abs(value) < 1e15):
        return f"{value:g}{unit}"
    prefixes = [
        (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"),
        (1.0, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n"), (1e-12, "p"),
    ]
    for factor, prefix in prefixes:
        if abs(value) >= factor:
            scaled = value / factor
            # Drop trailing zeros: 1.0 -> "1", 4.70 -> "4.7"
            text = f"{scaled:.3g}"
            return f"{text}{prefix}{unit}"
    return f"{value:g}{unit}"


def _draw_wires(
    svg: ET.Element,
    laid_out: dict[str, Any],
) -> None:
    """Emit one ``<path>`` per edge using ELK's section bend points."""
    for edge in laid_out.get("edges", []):
        for section in edge.get("sections", []):
            start = section.get("startPoint", {})
            end = section.get("endPoint", {})
            bend = section.get("bendPoints", [])
            pts = [(float(start.get("x", 0)), float(start.get("y", 0)))]
            for bp in bend:
                pts.append((float(bp.get("x", 0)), float(bp.get("y", 0))))
            pts.append((float(end.get("x", 0)), float(end.get("y", 0))))
            if len(pts) < 2:
                continue
            d = f"M{pts[0][0]:.2f},{pts[0][1]:.2f} " + " ".join(
                f"L{x:.2f},{y:.2f}" for x, y in pts[1:]
            )
            ET.SubElement(
                svg,
                f"{{{_SVG_NS}}}path",
                {"d": d, "class": "wire"},
            )

        # Junction dots where 3+ edges share a point.
        for jp in edge.get("junctionPoints", []):
            ET.SubElement(
                svg,
                f"{{{_SVG_NS}}}circle",
                {
                    "cx": f"{float(jp.get('x', 0)):.2f}",
                    "cy": f"{float(jp.get('y', 0)):.2f}",
                    "r": "2.5",
                    "class": "junction",
                },
            )


def _compose_svg(
    components: list,
    laid_out: dict[str, Any],
    skin: dict[str, SymbolTemplate],
    ground_id: int,
) -> ET.Element:
    """Build the full output SVG from ELK's laid-out graph + skin templates."""
    children = {c["id"]: c for c in laid_out.get("children", [])}

    # Compute canvas extent from cell bboxes.
    max_x = 0.0
    max_y = 0.0
    for child in children.values():
        right = float(child.get("x", 0)) + float(child.get("width", 0))
        bottom = float(child.get("y", 0)) + float(child.get("height", 0))
        max_x = max(max_x, right)
        max_y = max(max_y, bottom)
    # Also expand to cover any wire bend point outside the cell bboxes.
    for edge in laid_out.get("edges", []):
        for section in edge.get("sections", []):
            for pt_key in ("startPoint", "endPoint"):
                pt = section.get(pt_key) or {}
                max_x = max(max_x, float(pt.get("x", 0)))
                max_y = max(max_y, float(pt.get("y", 0)))
            for bp in section.get("bendPoints", []):
                max_x = max(max_x, float(bp.get("x", 0)))
                max_y = max(max_y, float(bp.get("y", 0)))

    svg = _make_root_svg(max_x, max_y)

    # Draw wires first so symbols paint on top of any visual overlap.
    _draw_wires(svg, laid_out)

    # Then drop in each symbol at its laid-out position. The skin's
    # generic-fallback symbol is used for unrecognized kinds and for
    # ground stubs (we emit one stub per gnd-touching terminal — same
    # convention as the netlistsvg backend).
    generic_template = skin.get(_GENERIC_KIND)
    gnd_template = skin.get("gnd")
    gnd_counter = 0
    for comp in components:
        node = children.get(comp.name)
        if node is None:
            continue
        skin_key = _KIND_TO_SKIN.get(comp.kind, _GENERIC_KIND)
        template = skin.get(skin_key) or generic_template
        if template is None:
            continue
        g = _instantiate_symbol(
            template,
            float(node["x"]),
            float(node["y"]),
            ref=comp.name,
            value=_format_value(comp),
        )
        svg.append(g)

        # Emit a ground stub per terminal connected to ground (mirrors
        # the netlistsvg backend's per-device gnd cells).
        if gnd_template is not None:
            for idx, n in enumerate(comp.nodes):
                if int(n) != ground_id:
                    continue
                port_order = _SKIN_PORT_ORDER.get(skin_key, ())
                if idx >= len(port_order):
                    continue
                port_id = port_order[idx]
                anchor = template.ports.get(port_id)
                if anchor is None:
                    continue
                ax = float(node["x"]) + anchor[0]
                ay = float(node["y"]) + anchor[1]
                gnd_g = _instantiate_symbol(
                    gnd_template,
                    ax - gnd_template.width / 2,
                    ay + 5.0,
                    ref=None,         # gnd stubs don't need a "ref" label
                    value=None,
                    show_node_label=False,
                )
                svg.append(gnd_g)
                gnd_counter += 1

    return svg


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_native(circuit: Any, svg_path: Any) -> Path:
    """Render a Pulsim Circuit to SVG via the pure-Python native backend.

    Pipeline summary:
      1. Load the analog skin (cached after first call).
      2. Resolve every position hint set via ``Circuit.set_position(...)``
         or the YAML ``position:`` field into absolute coordinates.
      3. Build a Pulsim-aware ELK graph (cell sizes from skin geometry,
         hinted cells pinned via ``org.eclipse.elk.position``).
      4. Subprocess ``node elk_bridge.js`` for the layered layout.
      5. Compose the output SVG: skin symbols at laid-out positions +
         orthogonal wire paths + ground stubs.

    Args:
        circuit: Pulsim Circuit (uses ``components()``, ``ground()``,
            and ``position_hints()``).
        svg_path: Destination SVG path (must end in ``.svg``).

    Returns:
        The :class:`pathlib.Path` of the written SVG.

    Raises:
        ImportError: If ``node`` is not on PATH (needed for layout).
        FileNotFoundError: If the skin SVG cannot be located.
        RuntimeError: If the ELK subprocess fails or produces no output.
        ValueError: If two position hints resolve to the same absolute
            coordinates (deterministic error so the user notices the
            collision).
    """
    skin_path = _resolve_skin_path()
    skin = parse_skin(skin_path)

    components = list(circuit.components())
    ground_id = int(circuit.ground())

    # Phase 3: resolve any user-supplied position hints. Empty dict when
    # no hints are set — the un-hinted code path stays bit-identical to
    # the Phase 1 baseline.
    position_hints = _resolve_hints(circuit)

    graph = _build_elk_graph(components, ground_id, skin, position_hints)
    laid_out = _call_elk(graph)

    # Phase 3 post-processing: ELK's layered algorithm doesn't snap to
    # `org.eclipse.elk.position` constraints, so we override the cell
    # coordinates ourselves and rewrite any edge whose endpoint cell got
    # moved. The result: hinted cells land at exactly the user's
    # coordinates; wires touching them get a simple L-route via
    # :func:`_apply_position_hints`.
    if position_hints:
        _apply_position_hints(laid_out, position_hints, components, skin)

    svg_root = _compose_svg(components, laid_out, skin, ground_id)

    out_path = Path(svg_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(svg_root)
    tree.write(out_path, xml_declaration=True, encoding="utf-8")
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"native backend produced no output at {out_path}")
    return out_path
