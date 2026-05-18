"""Force-directed schematic layout with electrical priors.

Phase 2 (MVP) algorithm:
    1. Build a NetworkX graph where nodes are *electrical* nodes (the integer
       IDs returned by ``Circuit.add_node`` plus ground = -1) and edges are
       induced by each component's terminal pairs.
    2. Pin a subset of nodes based on ``Circuit.node_position_hint``:
       ground at the canvas south edge, voltage-source positive terminals
       at the west edge, resistive-load nodes at the east edge.
    3. Run :func:`networkx.spring_layout` with a fixed seed for determinism.
    4. Rescale node positions to a canvas sized by ``60 * sqrt(N)`` mm.
    5. Post-process: enforce ``y(ground) >= max(y_other)`` so the
       "ground at the bottom" gate holds even when the spring forces push
       other nodes downward.
    6. Derive component placements:
       - 2-terminal: centered between its two endpoint coords; rotation
         snapped to ``{0, 90, 180, 270}`` along the endpoint-endpoint angle.
       - 3+ terminal: centered at the centroid of its endpoint coords;
         rotation is 0 (refined in Phase 3 with symbol-aware orientation).
    7. Emit wires as straight lines between every pair of terminals sharing
       an electrical node (star pattern from the first terminal on that
       node). Orthogonal Manhattan routing lands in a follow-up change.

The output is fully deterministic for a given input circuit: same circuit
construction → byte-identical :class:`SchematicLayout` JSON.
"""

from __future__ import annotations

import math
import os
from typing import Any

from .elk_backend import compute_layout_elk
from .templates import apply_templates, recognize_all
from .types import (
    BoundingBox,
    ComponentPlacement,
    SchematicLayout,
    TerminalAnchor,
    Wire,
    WireEndpoint,
)

_SPRING_SEED = 42
_SPRING_ITERATIONS = 200
_CANVAS_BASE_MM = 60.0
_MARGIN_MM = 10.0
_ROTATION_STEPS = (0, 90, 180, 270)
# Step size for perpendicular separation of 2-terminal devices that
# share both nodes (parallel devices like Cout || Rload across vout↔gnd).
# Large enough that schemdraw symbols at SCALE=0.1 don't visually
# overlap; small enough that the separated devices still read as
# parallel rather than independent.
_PARALLEL_OFFSET_MM = 18.0


def _require_networkx():
    """Lazy-import networkx with a clear ImportError pointing at the extra."""
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "pulsim.schematic.compute_layout requires networkx. "
            "Install via: pip install pulsim[schematic]"
        ) from exc
    return nx


def _snap_rotation_deg(dx: float, dy: float) -> int:
    """Snap an arbitrary angle into {0, 90, 180, 270} degrees.

    The angle is the direction from terminal 0 to terminal 1. Snapping
    chooses the nearest orthogonal rotation; ties resolve toward 0 then 90.
    """
    angle = math.degrees(math.atan2(dy, dx))  # (-180, 180]
    # Normalize to [0, 360)
    angle = (angle + 360.0) % 360.0
    # Snap to nearest 90
    idx = int(round(angle / 90.0)) % 4
    return _ROTATION_STEPS[idx]


def _collect_electrical_nodes(
    components: list, ground_id: int
) -> list[int]:
    """All node IDs touched by any component, plus ground if used.

    Order: ground first (if present), then components' nodes in insertion
    order, deduplicated. Determinism: this list seeds NetworkX node order,
    which feeds spring_layout's internal iteration.
    """
    seen: set[int] = set()
    ordered: list[int] = []
    has_ground = any(ground_id in c.nodes for c in components)
    if has_ground:
        seen.add(ground_id)
        ordered.append(ground_id)
    for comp in components:
        for n in comp.nodes:
            if n not in seen:
                seen.add(n)
                ordered.append(n)
    return ordered


def _compute_node_positions(
    circuit,
    components: list,
    nx_module,
) -> dict[int, tuple[float, float]]:
    """Run spring_layout with electrical priors and rescale to mm.

    Returns a mapping ``node_id -> (x_mm, y_mm)``.
    """
    ground_id = circuit.ground()
    electrical_nodes = _collect_electrical_nodes(components, ground_id)
    n_total = len(electrical_nodes)

    # Canvas size scales with sqrt(num_components) — empirically gives
    # enough room to avoid overlaps for small circuits without bloat.
    canvas_side = _CANVAS_BASE_MM * max(1.0, math.sqrt(max(1, len(components))))

    if n_total == 0:
        return {}

    # Build the graph. Edges come from component terminal pairs (full
    # pairwise for 3+ terminal devices). Edge keys are unique per
    # (u, v, component_name) so multiple wires between the same node pair
    # don't collapse.
    G = nx_module.Graph()
    for node_id in electrical_nodes:
        G.add_node(node_id)
    for comp in components:
        nodes_in_pin_order = list(comp.nodes)
        for i in range(len(nodes_in_pin_order)):
            for j in range(i + 1, len(nodes_in_pin_order)):
                u, v = nodes_in_pin_order[i], nodes_in_pin_order[j]
                if u != v:
                    G.add_edge(u, v)

    # Identify "control input" nodes (terminal 0 of 3-terminal switches).
    # A gate-driver voltage source's positive terminal has node_position_hint
    # = "source_pos" and would otherwise get pinned to the canvas west
    # edge — pulling the gate driver miles away from the switch it controls.
    # Suppress the source_pos pin for these specific nodes; the spring
    # layout will then keep them adjacent to the switch they drive.
    control_nodes: set[int] = set()
    for c in components:
        if c.kind in ("mosfet", "igbt", "vcswitch") and len(c.nodes) >= 3:
            control_nodes.add(int(c.nodes[0]))

    # Priors: pin a subset based on role hints. Coordinates here are in
    # *normalized* [0, 1] space; we rescale to mm after the spring layout.
    pos_init: dict[int, tuple[float, float]] = {}
    fixed_ids: list[int] = []

    for node_id in electrical_nodes:
        role = circuit.node_position_hint(node_id)
        if role == "ground":
            pos_init[node_id] = (0.5, 1.0)  # bottom-center
            fixed_ids.append(node_id)
        elif role == "source_pos" and node_id not in control_nodes:
            pos_init[node_id] = (0.0, 0.5)  # left-center
            fixed_ids.append(node_id)
        elif role == "load":
            pos_init[node_id] = (1.0, 0.5)  # right-center
            fixed_ids.append(node_id)

    # Edge case: no nodes pinned. Spring layout still deterministic with
    # seed but spread across the default [-1, 1] range; pass center+scale
    # so positions land inside [0, 1] for the rescale step.
    if fixed_ids:
        pos_norm: dict[int, Any] = nx_module.spring_layout(
            G,
            pos=pos_init,
            fixed=fixed_ids,
            seed=_SPRING_SEED,
            iterations=_SPRING_ITERATIONS,
        )
    else:
        pos_norm = nx_module.spring_layout(
            G,
            seed=_SPRING_SEED,
            iterations=_SPRING_ITERATIONS,
            scale=0.5,
            center=(0.5, 0.5),
        )

    # Cast NetworkX's numpy.ndarray values to plain (float, float) tuples
    # and rescale from normalized coords to mm.
    pos_mm: dict[int, tuple[float, float]] = {}
    for node_id in electrical_nodes:
        nx_pos = pos_norm[node_id]
        x_n = float(nx_pos[0])
        y_n = float(nx_pos[1])
        pos_mm[node_id] = (x_n * canvas_side, y_n * canvas_side)

    # Phase 4: apply topology templates (bridge, boost, half-bridge).
    # Templates override the anchor-node positions to snap recognized
    # sub-topologies into canonical shapes. Run BEFORE the lift-ground
    # post-pass so the final layout still satisfies y(ground) ≥ max(y).
    templates = recognize_all(components)
    if templates:
        pos_mm = apply_templates(templates, pos_mm, ground_node=ground_id)

    # Hard prior: ground must sit at or below every other node. Spring
    # forces (and template overrides) sometimes pull internal nodes below
    # the pinned ground; this post-pass lifts ground's y so the
    # "ground at bottom" gate holds.
    if ground_id in pos_mm:
        gx, gy = pos_mm[ground_id]
        other_ys = [y for node_id, (_, y) in pos_mm.items() if node_id != ground_id]
        if other_ys:
            max_other_y = max(other_ys)
            if gy < max_other_y + 1.0:
                pos_mm[ground_id] = (gx, max_other_y + 5.0)

    return pos_mm


def _component_center_and_rotation(
    comp,
    pos_mm: dict[int, tuple[float, float]],
) -> tuple[float, float, int]:
    """Center + snapped rotation for one component.

    Two-terminal devices: midpoint of the two terminal coords, rotation
    snapped from pin0→pin1.

    Three-terminal switches (MOSFET, IGBT, vcswitch): the device *body*
    sits between pins 1 and 2 — drain↔source for FETs / IGBTs, t1↔t2 for
    vcswitch. Pin 0 (gate / control input) extends sideways and would
    distort the body center if included in a 3-way centroid (e.g. a
    boost MOSFET whose gate is wired far west would have its body
    yanked into the middle of the canvas). So we use midpoint(pin 1,
    pin 2) — the gate becomes a stub connecting outward.

    Four-plus-terminal devices (transformer): centroid of all terminals,
    rotation 0 (the symbol shape is two-sided so a snap-rotation derived
    from arbitrary pin pairs would be misleading).
    """
    coords = [pos_mm[n] for n in comp.nodes]
    if not coords:
        return 0.0, 0.0, 0

    if comp.kind in ("mosfet", "igbt", "vcswitch") and len(coords) >= 3:
        x_a, y_a = coords[1]
        x_b, y_b = coords[2]
        cx = (x_a + x_b) / 2.0
        cy = (y_a + y_b) / 2.0
        rotation = _snap_rotation_deg(x_b - x_a, y_b - y_a)
        return cx, cy, rotation

    if len(coords) == 2:
        x0, y0 = coords[0]
        x1, y1 = coords[1]
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        rotation = _snap_rotation_deg(x1 - x0, y1 - y0)
        return cx, cy, rotation

    cx = sum(c[0] for c in coords) / len(coords)
    cy = sum(c[1] for c in coords) / len(coords)
    return cx, cy, 0


def _route_ground_rail(
    components: list,
    placements: dict[str, ComponentPlacement],
    ground_id: int,
    rail_y: float,
) -> dict[str, ComponentPlacement]:
    """Relocate every gnd-terminal anchor onto a horizontal rail at ``rail_y``.

    Effect on the rendered schematic:

    * Two-terminal devices with one terminal at ground (e.g. ``Cout``,
      ``Rload``, a freewheel diode, a snubber R-C bottom node) become
      **clean vertical bars** running from their non-gnd terminal straight
      down to the rail. The body's x is set to the non-gnd terminal's x
      (or the per-device perpendicular offset for parallel groups
      sharing the same non-gnd node — Cout-and-Rload across vout↔gnd is
      the canonical example).
    * Three-plus-terminal devices (MOSFET / IGBT) with one pin at ground
      keep their body where the placement step put it and only the
      gnd-terminal anchor moves onto the rail at the device's own x.
      The renderer draws a stub from the device's symbol to the rail.

    Star wires emitted by :func:`_emit_wires_and_junctions` for the
    ground node connect every gnd-terminal at its rail-x ↔ another's
    rail-x — all at ``rail_y`` — forming a continuous visible ground
    rail under the circuit.

    Determinism: groups iterated in sorted-non-gnd-node order; devices
    in each group sorted by name.
    """
    two_terminal_groups: dict[int, list[str]] = {}
    three_plus_terminal: list[str] = []
    for c in components:
        gnd_count = sum(1 for n in c.nodes if int(n) == ground_id)
        if gnd_count == 0:
            continue
        if len(c.nodes) == 2:
            other_node = next(int(n) for n in c.nodes if int(n) != ground_id)
            two_terminal_groups.setdefault(other_node, []).append(c.name)
        else:
            three_plus_terminal.append(c.name)

    out = dict(placements)

    for other_node, names in sorted(two_terminal_groups.items()):
        ref = out[names[0]]
        ref_other = next(a for a in ref.terminal_anchors if int(a.node) == other_node)
        other_x = ref_other.x
        other_y = ref_other.y

        n = len(names)
        sorted_names = sorted(names)
        for i, name in enumerate(sorted_names):
            offset = (i - (n - 1) / 2.0) * _PARALLEL_OFFSET_MM if n > 1 else 0.0
            bar_x = other_x + offset
            placement = out[name]
            new_anchors_list = []
            for a in placement.terminal_anchors:
                if int(a.node) == ground_id:
                    new_anchors_list.append(
                        TerminalAnchor(index=a.index, x=bar_x, y=rail_y, node=a.node)
                    )
                else:
                    new_anchors_list.append(
                        TerminalAnchor(index=a.index, x=bar_x, y=other_y, node=a.node)
                    )
            new_anchors = tuple(new_anchors_list)
            term_gnd = next(a for a in new_anchors if int(a.node) == ground_id)
            term_other = next(a for a in new_anchors if int(a.node) != ground_id)
            new_cx = (term_gnd.x + term_other.x) / 2.0
            new_cy = (term_gnd.y + term_other.y) / 2.0
            new_rot = _snap_rotation_deg(
                term_gnd.x - term_other.x, term_gnd.y - term_other.y
            )
            out[name] = ComponentPlacement(
                name=placement.name,
                kind=placement.kind,
                x=new_cx,
                y=new_cy,
                rotation=new_rot,
                terminal_anchors=new_anchors,
            )

    for name in three_plus_terminal:
        placement = out[name]
        new_anchors = tuple(
            TerminalAnchor(index=a.index, x=placement.x, y=rail_y, node=a.node)
            if int(a.node) == ground_id
            else a
            for a in placement.terminal_anchors
        )
        out[name] = ComponentPlacement(
            name=placement.name,
            kind=placement.kind,
            x=placement.x,
            y=placement.y,
            rotation=placement.rotation,
            terminal_anchors=new_anchors,
        )

    return out


def _separate_parallel_devices(
    components: list,
    placements: dict[str, ComponentPlacement],
    ground_id: int,
) -> dict[str, ComponentPlacement]:
    """Snap 2-terminal devices that share both nodes onto an orthogonal axis.

    A circuit can have many 2-terminal devices wired across the same two
    nodes — e.g. ``Cout`` and ``Rload`` both connecting ``vout↔gnd``.
    Before this pass, both share the same midpoint and render on top of
    each other along whatever diagonal axis the layout produced.

    For each group of N devices on shared node pair (A, B):

    1. Pick the **dominant** orthogonal axis — vertical if |Δy| ≥ |Δx|,
       horizontal otherwise. The device bodies all snap to that axis.
    2. The device endpoints (terminal anchors) move onto:
       - the matching A.y / B.y (vertical case) or A.x / B.x (horizontal),
       - a per-device perpendicular offset
       so each device is rendered axis-aligned with neighbours fanned out
       sideways.
    3. The connecting wires emitted by :func:`_emit_wires_and_junctions`
       become axis-aligned stubs to the actual A and B positions, even
       when (A, B) was diagonal — textbook "horizontal rail to a column
       of parallel devices" look without needing full Manhattan routing.

    Determinism: groups iterated in sorted node-pair order; devices
    within each group sorted by name.
    """
    groups: dict[tuple[int, int], list[str]] = {}
    for c in components:
        if len(c.nodes) != 2:
            continue
        a, b = int(c.nodes[0]), int(c.nodes[1])
        # Skip pairs that include ground — those are handled by
        # _route_ground_rail, which fans them out as vertical bars
        # onto the ground rail (preserves the textbook ground-rail look).
        if a == ground_id or b == ground_id:
            continue
        key = (a, b) if a <= b else (b, a)
        groups.setdefault(key, []).append(c.name)

    out = dict(placements)
    for (a, b), names in sorted(groups.items()):
        if len(names) <= 1:
            continue
        ref = out[names[0]]
        ref_anchors = {t.node: t for t in ref.terminal_anchors}
        if a not in ref_anchors or b not in ref_anchors:
            continue
        ax, ay = ref_anchors[a].x, ref_anchors[a].y
        bx, by = ref_anchors[b].x, ref_anchors[b].y
        dx = bx - ax
        dy = by - ay
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue

        sorted_names = sorted(names)
        n = len(sorted_names)
        vertical = abs(dy) >= abs(dx)

        for i, name in enumerate(sorted_names):
            offset = (i - (n - 1) / 2.0) * _PARALLEL_OFFSET_MM
            placement = out[name]
            new_anchors_list = []
            for t in placement.terminal_anchors:
                if vertical:
                    mid = (ax + bx) / 2.0
                    new_x = mid + offset
                    if t.node == a:
                        new_y = ay
                    elif t.node == b:
                        new_y = by
                    else:
                        new_y = t.y
                else:
                    mid = (ay + by) / 2.0
                    new_y = mid + offset
                    if t.node == a:
                        new_x = ax
                    elif t.node == b:
                        new_x = bx
                    else:
                        new_x = t.x
                new_anchors_list.append(
                    TerminalAnchor(index=t.index, x=new_x, y=new_y, node=t.node)
                )
            new_anchors = tuple(new_anchors_list)
            term_a = next(t for t in new_anchors if t.node == a)
            term_b = next(t for t in new_anchors if t.node == b)
            new_cx = (term_a.x + term_b.x) / 2.0
            new_cy = (term_a.y + term_b.y) / 2.0
            new_rot = _snap_rotation_deg(term_b.x - term_a.x, term_b.y - term_a.y)
            out[name] = ComponentPlacement(
                name=placement.name,
                kind=placement.kind,
                x=new_cx,
                y=new_cy,
                rotation=new_rot,
                terminal_anchors=new_anchors,
            )

    return out


def _emit_wires_and_junctions(
    components: list,
    pos_mm: dict[int, tuple[float, float]],
) -> tuple[tuple[Wire, ...], tuple[tuple[float, float], ...]]:
    """One wire per (node, terminal-pair on that node) — star from terminal 0.

    Junctions: nodes where 3+ terminals meet.
    """
    # node_id -> list of (component_name, terminal_index)
    node_to_terms: dict[int, list[tuple[str, int]]] = {}
    for comp in components:
        for terminal_idx, node_id in enumerate(comp.nodes):
            node_to_terms.setdefault(node_id, []).append((comp.name, terminal_idx))

    wires: list[Wire] = []
    junctions: list[tuple[float, float]] = []

    # Iterate nodes in sorted order for deterministic wire emission.
    for node_id in sorted(node_to_terms.keys()):
        terms = node_to_terms[node_id]
        if len(terms) < 2:
            continue
        first_comp, first_term = terms[0]
        node_x, node_y = pos_mm[node_id]
        for other_comp, other_term in terms[1:]:
            wires.append(
                Wire(
                    from_=WireEndpoint(component=first_comp, terminal=first_term),
                    to=WireEndpoint(component=other_comp, terminal=other_term),
                    path=((node_x, node_y), (node_x, node_y)),
                )
            )
        if len(terms) >= 3:
            junctions.append((node_x, node_y))

    return tuple(wires), tuple(junctions)


def _compute_canvas(pos_mm: dict[int, tuple[float, float]]) -> BoundingBox:
    """Bounding box covering all node positions with a margin."""
    if not pos_mm:
        return BoundingBox(x=0.0, y=0.0, width=_CANVAS_BASE_MM, height=_CANVAS_BASE_MM)
    xs = [p[0] for p in pos_mm.values()]
    ys = [p[1] for p in pos_mm.values()]
    return BoundingBox(
        x=min(xs) - _MARGIN_MM,
        y=min(ys) - _MARGIN_MM,
        width=(max(xs) - min(xs)) + 2 * _MARGIN_MM,
        height=(max(ys) - min(ys)) + 2 * _MARGIN_MM,
    )


def compute_layout(circuit) -> SchematicLayout:
    """Build a deterministic :class:`SchematicLayout` for a Circuit.

    Default backend is **ELK** (Eclipse Layout Kernel via elkjs running on
    Node.js) — a Sugiyama layered algorithm with orthogonal edge routing
    that produces textbook-style schematics with left-to-right power flow
    and a clear ground rail at the bottom. The legacy in-tree
    spring-layout + topology-templates + parallel-pair + ground-rail
    pipeline is still callable as a fallback by setting the environment
    variable ``PULSIM_SCHEMATIC_BACKEND=spring`` (used for environments
    without Node.js installed, or for backwards-compat regression).

    Args:
        circuit: Any object exposing ``components()``, ``ground()``, and
            ``node_position_hint(node_id)`` matching the Pulsim Circuit
            contract.

    Returns:
        A :class:`SchematicLayout` with every component placed, wires
        connecting shared-node terminals, junctions where 3+ terminals
        meet, and a canvas bounding box.

    Raises:
        ImportError: On the ELK path, if Node.js isn't on PATH (the user-
            facing message names the install hint). On the spring fallback
            path, if ``networkx`` isn't installed.
    """
    backend = os.environ.get("PULSIM_SCHEMATIC_BACKEND", "elk").strip().lower()
    if backend not in ("elk", "spring"):
        backend = "elk"
    if backend == "elk":
        return compute_layout_elk(circuit)

    # Spring fallback path — original force-directed implementation.
    nx_module = _require_networkx()

    components = list(circuit.components())
    if not components:
        return SchematicLayout(
            components={},
            wires=(),
            junctions=(),
            canvas=BoundingBox(x=0.0, y=0.0, width=_CANVAS_BASE_MM, height=_CANVAS_BASE_MM),
        )

    pos_mm = _compute_node_positions(circuit, components, nx_module)

    placements: dict[str, ComponentPlacement] = {}
    for comp in components:
        cx, cy, rotation = _component_center_and_rotation(comp, pos_mm)
        anchors = tuple(
            TerminalAnchor(
                index=idx,
                x=pos_mm[node_id][0],
                y=pos_mm[node_id][1],
                node=int(node_id),
            )
            for idx, node_id in enumerate(comp.nodes)
        )
        placements[comp.name] = ComponentPlacement(
            name=comp.name,
            kind=comp.kind,
            x=cx,
            y=cy,
            rotation=rotation,
            terminal_anchors=anchors,
        )

    # Step A: separate parallel non-ground pairs (orthogonal snap so the
    # devices fan out perpendicular to the connection axis).
    ground_id = int(circuit.ground())
    placements = _separate_parallel_devices(components, placements, ground_id)

    # Step A.5: relocate switch gate-anchors to a stub left of the body.
    # The gate node is typically driven by a far-away gate-driver source,
    # so leaving the gate-terminal anchor at the gate node's spring
    # position means zero-length wire = no visible gate connection. A
    # stub anchor at (placement.x − 8 mm, placement.y) gives the wire
    # generator a non-degenerate start point — the connecting wire to
    # the driver shows up as a straight line.
    _GATE_STUB_MM = 8.0
    for c in components:
        if c.kind in ("mosfet", "igbt", "vcswitch") and len(c.nodes) >= 3:
            placement = placements[c.name]
            anchors = list(placement.terminal_anchors)
            anchors[0] = TerminalAnchor(
                index=anchors[0].index,
                x=placement.x - _GATE_STUB_MM,
                y=placement.y,
                node=anchors[0].node,
            )
            placements[c.name] = ComponentPlacement(
                name=placement.name,
                kind=placement.kind,
                x=placement.x,
                y=placement.y,
                rotation=placement.rotation,
                terminal_anchors=tuple(anchors),
            )

    # Step B: route every ground terminal onto a horizontal ground rail.
    # The rail y is the post-lift ground position from _compute_node_positions,
    # which is guaranteed to be at or below every other node (gate G.6).
    # 2-terminal devices touching ground become vertical bars; parallel
    # gnd pairs (Cout || Rload across vout↔gnd) get x-offsets so each one
    # owns its own vertical column ending on the rail.
    if ground_id in pos_mm:
        rail_y = pos_mm[ground_id][1]
        placements = _route_ground_rail(components, placements, ground_id, rail_y)

    wires, junctions = _emit_wires_and_junctions(components, pos_mm)
    canvas = _compute_canvas(pos_mm)

    return SchematicLayout(
        components=placements,
        wires=wires,
        junctions=junctions,
        canvas=canvas,
    )
