"""Topology recognizer + canonical layout overrides for the schematic engine.

Phase 4 of the ``add-schematic-rendering`` change. Recognizers scan a list
of :class:`pulsim.ComponentDescriptor` for common power-electronics
sub-topologies (bridge rectifier, boost stage, half-bridge) and emit one
:class:`RecognizedTemplate` per match. The layout engine consumes these
matches to *override* the spring-layout positions of the templates'
anchor nodes, snapping the recognized region into a clean canonical shape.

Each recognizer is a pure function: same component list → same matches.
Priority is fixed (defined by recognizer order in :data:`_RECOGNIZERS`);
when two recognizers would claim the same component, the higher-priority
one wins and the conflicting recognizer is skipped for that component.

Geometric convention used throughout: +x is right, +y is *down* (screen
convention — same as :mod:`pulsim.schematic.layout`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


# Canonical template sizes (mm). These define the spacing between anchor
# nodes within each template's local frame; the template centroid is set
# to the centroid of the anchor nodes' current positions in pos_mm.
#
# Sizing rule: each value must be ≥ 30 mm so the schemdraw symbols (which
# render at ≈30 mm at the layout's SCALE=0.1 mm→unit) have room without
# overlapping their neighbours. Bridge / boost get a bit more so the
# diodes inside the diamond / the L–Q–D row have visual breathing room.
_BRIDGE_SIZE_MM = 35.0
_BOOST_SIZE_MM = 45.0
_HALF_BRIDGE_SIZE_MM = 35.0


@dataclass(frozen=True)
class RecognizedTemplate:
    """One matched sub-topology.

    Attributes:
        template_id: Canonical template name, e.g. ``"bridge_rectifier"``.
        role_to_component: Map from a template-defined role label
            (``"D1"``, ``"L"``, ``"Q_hi"`` …) to the user-visible
            component name in the circuit.
        anchor_nodes: Map from a template-defined anchor-node label
            (``"ac_a"``, ``"dc_pos"`` …) to the integer electrical-node
            index that plays that role.
    """

    template_id: str
    role_to_component: dict[str, str]
    anchor_nodes: dict[str, int]

    @property
    def component_set(self) -> frozenset[str]:
        return frozenset(self.role_to_component.values())


# -----------------------------------------------------------------------------
# Recognizers — each takes a list of ComponentDescriptor-likes and returns
# zero or more matches. Pure functions, deterministic.
# -----------------------------------------------------------------------------


def _recognize_bridge_rectifier(components: list) -> list[RecognizedTemplate]:
    """Find groups of 4 diodes forming an AC→DC bridge rectifier.

    Topology constraint:
      D1: anode=ac_a, cathode=dc_pos
      D2: anode=dc_neg, cathode=ac_a
      D3: anode=ac_b, cathode=dc_pos
      D4: anode=dc_neg, cathode=ac_b

    Algorithm:
      1. Group diodes by cathode node → pairs sharing a cathode are
         (D1, D3) candidates for that dc_pos.
      2. Group diodes by anode node → pairs sharing an anode are
         (D2, D4) candidates for that dc_neg.
      3. The bridge matches when the cathodes of the (D2, D4) pair are
         exactly the anodes of the (D1, D3) pair (= {ac_a, ac_b}).

    Returns deduplicated matches; the same component set is never emitted
    twice even if the cathode-pair / anode-pair enumeration would do so.
    """
    diodes = [c for c in components if c.kind == "diode" and len(c.nodes) == 2]
    if len(diodes) < 4:
        return []

    cathode_to_diodes: dict[int, list] = {}
    anode_to_diodes: dict[int, list] = {}
    for d in diodes:
        anode, cathode = int(d.nodes[0]), int(d.nodes[1])
        cathode_to_diodes.setdefault(cathode, []).append(d)
        anode_to_diodes.setdefault(anode, []).append(d)

    seen_component_sets: set[frozenset[str]] = set()
    matches: list[RecognizedTemplate] = []

    for dc_pos, top_diodes in cathode_to_diodes.items():
        if len(top_diodes) != 2:
            continue
        # Sort for determinism — ties on cathode rank by diode name.
        top_sorted = sorted(top_diodes, key=lambda d: d.name)
        ac_a = int(top_sorted[0].nodes[0])
        ac_b = int(top_sorted[1].nodes[0])
        if ac_a == ac_b:
            continue

        for dc_neg, bot_diodes in anode_to_diodes.items():
            if len(bot_diodes) != 2 or dc_neg == dc_pos:
                continue
            bot_cathodes = {int(bot_diodes[0].nodes[1]), int(bot_diodes[1].nodes[1])}
            if bot_cathodes != {ac_a, ac_b}:
                continue

            d1 = top_sorted[0]   # anode=ac_a, cathode=dc_pos
            d3 = top_sorted[1]   # anode=ac_b, cathode=dc_pos
            d2 = next(d for d in bot_diodes if int(d.nodes[1]) == ac_a)
            d4 = next(d for d in bot_diodes if int(d.nodes[1]) == ac_b)

            match = RecognizedTemplate(
                template_id="bridge_rectifier",
                role_to_component={"D1": d1.name, "D2": d2.name,
                                   "D3": d3.name, "D4": d4.name},
                anchor_nodes={"ac_a": ac_a, "ac_b": ac_b,
                              "dc_pos": dc_pos, "dc_neg": dc_neg},
            )
            key = match.component_set
            if key in seen_component_sets:
                continue
            seen_component_sets.add(key)
            matches.append(match)

    return matches


def _recognize_boost_stage(components: list) -> list[RecognizedTemplate]:
    """Find inductor + (MOSFET|IGBT|vcswitch) + diode in boost arrangement.

    Topology constraint:
      Q (3-terminal): nodes = [ctrl_or_gate, drain, source]
      L (2-terminal): one terminal = Q.drain, other = dc_in
      D (2-terminal): anode = Q.drain, cathode = vout
      gnd = Q.source

    Returns at most one match per (Q, L, D) triple; the first matching
    inductor and first matching diode for each switch are claimed.
    """
    switches = [c for c in components if c.kind in ("mosfet", "igbt", "vcswitch") and len(c.nodes) == 3]
    inductors = [c for c in components if c.kind == "inductor" and len(c.nodes) == 2]
    diodes = [c for c in components if c.kind == "diode" and len(c.nodes) == 2]

    matches: list[RecognizedTemplate] = []
    consumed_inductors: set[str] = set()
    consumed_diodes: set[str] = set()

    def _find_inductor_at(drain_node: int) -> tuple[object, int] | None:
        for ind in inductors:
            if ind.name in consumed_inductors:
                continue
            n0, n1 = int(ind.nodes[0]), int(ind.nodes[1])
            if n0 == drain_node:
                return ind, n1
            if n1 == drain_node:
                return ind, n0
        return None

    def _find_diode_anode_at(drain_node: int) -> tuple[object, int] | None:
        for d in diodes:
            if d.name in consumed_diodes:
                continue
            if int(d.nodes[0]) == drain_node:
                return d, int(d.nodes[1])
        return None

    for sw in switches:
        # Pin convention: [ctrl/gate, drain, source]
        drain = int(sw.nodes[1])
        source = int(sw.nodes[2])

        ind_match = _find_inductor_at(drain)
        if ind_match is None:
            continue
        match_ind, dc_in = ind_match

        diode_match = _find_diode_anode_at(drain)
        if diode_match is None:
            continue
        match_diode, vout = diode_match

        # Sanity: dc_in ≠ vout ≠ source.
        if dc_in == vout or dc_in == source or vout == source:
            continue

        consumed_inductors.add(match_ind.name)  # type: ignore[attr-defined]
        consumed_diodes.add(match_diode.name)  # type: ignore[attr-defined]
        matches.append(RecognizedTemplate(
            template_id="boost_stage",
            role_to_component={
                "L": match_ind.name,  # type: ignore[attr-defined]
                "Q": sw.name,
                "D": match_diode.name,  # type: ignore[attr-defined]
            },
            anchor_nodes={"dc_in": dc_in, "sw_node": drain,
                          "vout": vout, "gnd": source},
        ))

    return matches


def _recognize_half_bridge(components: list) -> list[RecognizedTemplate]:
    """Find two switches in series forming a half-bridge.

    Topology constraint:
      Q_hi: nodes = [_, drain=bus_pos, source=midpoint]
      Q_lo: nodes = [_, drain=midpoint, source=bus_neg]

    No constraint on the gate nodes — the recognizer accepts independent
    or shared gate signals. Returns at most one match per (Q_hi, Q_lo)
    ordering; (Q_lo, Q_hi) is not emitted as a separate match.
    """
    switches = [c for c in components if c.kind in ("mosfet", "igbt", "vcswitch") and len(c.nodes) == 3]
    if len(switches) < 2:
        return []

    by_drain: dict[int, list] = {}
    for sw in switches:
        by_drain.setdefault(int(sw.nodes[1]), []).append(sw)

    matches: list[RecognizedTemplate] = []
    seen_pairs: set[frozenset[str]] = set()

    for q_hi in switches:
        hi_source = int(q_hi.nodes[2])
        # Q_lo candidates: drain == Q_hi.source
        for q_lo in by_drain.get(hi_source, []):
            if q_lo.name == q_hi.name:
                continue
            hi_drain = int(q_hi.nodes[1])
            lo_source = int(q_lo.nodes[2])
            if hi_drain == lo_source:
                continue
            pair = frozenset({q_hi.name, q_lo.name})
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            matches.append(RecognizedTemplate(
                template_id="half_bridge",
                role_to_component={"Q_hi": q_hi.name, "Q_lo": q_lo.name},
                anchor_nodes={"bus_pos": hi_drain,
                              "midpoint": hi_source,
                              "bus_neg": lo_source},
            ))

    return matches


# Priority order: bridge first (most specific / largest), then boost,
# then half-bridge. The first recognizer to claim a component wins.
_RECOGNIZERS: tuple[Callable[[list], list[RecognizedTemplate]], ...] = (
    _recognize_bridge_rectifier,
    _recognize_boost_stage,
    _recognize_half_bridge,
)


def recognize_all(components: list) -> list[RecognizedTemplate]:
    """Run every recognizer in priority order, dropping overlapping matches.

    A match is dropped when any of its components has already been claimed
    by a higher-priority match. Determinism: same input order → same output.
    """
    consumed: set[str] = set()
    out: list[RecognizedTemplate] = []
    for recognizer in _RECOGNIZERS:
        for match in recognizer(components):
            if match.component_set & consumed:
                continue
            consumed |= match.component_set
            out.append(match)
    return out


# -----------------------------------------------------------------------------
# Template application — compute canonical anchor-node positions
# -----------------------------------------------------------------------------


def _anchor_offsets(template: RecognizedTemplate) -> dict[int, tuple[float, float]]:
    """Anchor offsets in template-local frame (centered at origin, mm).

    Caller translates by the centroid of the anchor nodes' current
    spring-layout positions, so the templated region sits where the
    spring put it on the canvas — only the *shape* of the local
    arrangement is overridden, not its global location.
    """
    if template.template_id == "bridge_rectifier":
        s = _BRIDGE_SIZE_MM
        # Canonical textbook diamond:
        #
        #             dc_pos
        #               ●
        #             /   \
        #          D1       D3
        #          /          \
        #     ac_a              ac_b
        #          \          /
        #          D2       D4
        #            \    /
        #               ●
        #             dc_neg
        #
        # Each diode lives entirely in one quadrant between an AC and a
        # DC anchor — no two diodes share a path.
        return {
            template.anchor_nodes["dc_pos"]: (0.0, -s),
            template.anchor_nodes["dc_neg"]: (0.0, +s),
            template.anchor_nodes["ac_a"]:   (-s, 0.0),
            template.anchor_nodes["ac_b"]:   (+s, 0.0),
        }
    if template.template_id == "boost_stage":
        s = _BOOST_SIZE_MM
        # Boost stage canonical layout (horizontal L–Q–D row above gnd):
        #
        #   dc_in ──[L]── sw_node ──[D]── vout
        #                    │             │
        #                   [Q]          (Cout || Rload)
        #                    │             │
        #                   gnd ────────── gnd
        return {
            template.anchor_nodes["dc_in"]:   (-s,  -s / 2.0),
            template.anchor_nodes["sw_node"]: (0.0, -s / 2.0),
            template.anchor_nodes["vout"]:    (+s,  -s / 2.0),
            template.anchor_nodes["gnd"]:     (0.0, +s / 2.0),
        }
    if template.template_id == "half_bridge":
        s = _HALF_BRIDGE_SIZE_MM
        return {
            template.anchor_nodes["bus_pos"]:  (0.0, -s),
            template.anchor_nodes["midpoint"]: (0.0, 0.0),
            template.anchor_nodes["bus_neg"]:  (0.0, +s),
        }
    return {}


def apply_templates(
    templates: Iterable[RecognizedTemplate],
    pos_mm: dict[int, tuple[float, float]],
    *,
    ground_node: int | None = None,
) -> dict[int, tuple[float, float]]:
    """Override anchor-node positions to match each template's canonical shape.

    Strategy: visit templates in iteration order. As each template's
    anchors are positioned, they become *locked* — subsequent templates
    that share an anchor with a higher-priority template treat that
    anchor as a fixed reference and translate their own local frame to
    keep it where it is. The first locked anchor a template encounters
    (or ``ground_node`` if it's part of the template) becomes the
    translation pivot; if no anchor is locked yet, the centroid of the
    anchors' current spring-layout positions is used.

    This prevents two recognizers that share a node (e.g. a bridge
    rectifier's ``dc_pos`` is also a boost stage's ``dc_in``) from
    overwriting each other's geometry — instead they chain into a
    coherent shape where the shared node sits at exactly one place.

    Returns a new dict; the input is not mutated.
    """
    out = dict(pos_mm)
    # Lock map: nodes already positioned by a previous template (or by
    # the global ground prior) — never re-overridden, always usable as
    # translation pivots.
    locked: set[int] = set()
    if ground_node is not None and ground_node in out:
        locked.add(ground_node)

    for template in templates:
        offsets = _anchor_offsets(template)
        if not offsets:
            continue
        anchor_ids = [nid for nid in offsets if nid in out]
        if len(anchor_ids) < 2:
            continue

        # Translation pivot: prefer a previously locked non-ground anchor
        # (so chained templates share their interface node cleanly), then
        # ground (anchored to canvas south), then the centroid of the
        # anchors' current positions.
        #
        # Ground-as-pivot is *last resort* among locked anchors because
        # ground tends to be far from the templated region (pinned at
        # canvas south); using it as the pivot would force the entire
        # template upward into the canvas. A non-ground locked anchor
        # is almost always the actual interface to a neighbouring
        # template and keeps the chained shape coherent.
        pivot: int | None = None
        for nid in anchor_ids:
            if nid in locked and nid != ground_node:
                pivot = nid
                break
        if pivot is None and ground_node is not None and ground_node in offsets:
            pivot = ground_node

        if pivot is not None:
            pivot_actual = out[pivot]
            pivot_local = offsets[pivot]
            tx = pivot_actual[0] - pivot_local[0]
            ty = pivot_actual[1] - pivot_local[1]
        else:
            cx = sum(out[nid][0] for nid in anchor_ids) / len(anchor_ids)
            cy = sum(out[nid][1] for nid in anchor_ids) / len(anchor_ids)
            ox = sum(offsets[nid][0] for nid in anchor_ids) / len(anchor_ids)
            oy = sum(offsets[nid][1] for nid in anchor_ids) / len(anchor_ids)
            tx, ty = cx - ox, cy - oy

        for nid, (lx, ly) in offsets.items():
            if nid not in out or nid in locked:
                continue
            out[nid] = (lx + tx, ly + ty)
            locked.add(nid)

    return out
