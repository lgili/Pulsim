"""Phase 4 tests for the topology recognizer + canonical templates.

Cover the recognizer outputs for bridge / boost / half-bridge patterns,
the chained-anchor lock between templates, and the no-false-match
behavior on circuits that should NOT trigger any recognizer.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import pulsim as ps
from pulsim.schematic import recognize_all
from pulsim.schematic.templates import _BRIDGE_SIZE_MM, apply_templates


REPO_ROOT = Path(__file__).resolve().parents[2]
BUCK_YAML = REPO_ROOT / "examples" / "buck_converter.yaml"


pytest.importorskip("networkx")


# See test_schematic_layout.py for the spring-vs-ELK backend rationale.
# Tests that assert geometry of the legacy spring backend (canonical
# bridge diamond, ground-rail at one y, parallel pairs at distinct x)
# only run when `PULSIM_SCHEMATIC_BACKEND=spring`. The recognizer
# (`recognize_all`) and `apply_templates` themselves remain reachable
# under both backends — those tests stay enabled.
_SPRING_ONLY = pytest.mark.skipif(
    os.environ.get("PULSIM_SCHEMATIC_BACKEND", "elk").lower() != "spring",
    reason="Asserts a property of the legacy spring backend; ELK uses a different geometry.",
)


# -----------------------------------------------------------------------------
# bridge_rectifier recognizer
# -----------------------------------------------------------------------------


def _build_bridge_only() -> ps.Circuit:
    ckt = ps.Circuit()
    ac_a = ckt.add_node("ac_a")
    ac_b = ckt.add_node("ac_b")
    dc_pos = ckt.add_node("dc_pos")
    gnd = ckt.ground()
    ckt.add_bridge_rectifier("DBR", ac_a, ac_b, dc_pos, gnd, 0.7, 0.01)
    # Driving source on the AC side so the bridge isn't dangling.
    sine = ps.SineParams()
    sine.amplitude = 100.0
    sine.frequency = 50.0
    ckt.add_sine_voltage_source("Vac", ac_a, ac_b, sine)
    # Resistive load on the DC side so dc_pos is classified.
    ckt.add_resistor("Rload", dc_pos, gnd, 100.0)
    return ckt


def test_recognize_bridge_rectifier_finds_canonical_bridge() -> None:
    ckt = _build_bridge_only()
    matches = recognize_all(list(ckt.components()))
    bridges = [m for m in matches if m.template_id == "bridge_rectifier"]
    assert len(bridges) == 1, f"expected 1 bridge, got {len(bridges)}"

    bridge = bridges[0]
    # All four DBR__D* diodes consumed
    assert bridge.component_set == frozenset({"DBR__D1", "DBR__D2", "DBR__D3", "DBR__D4"})
    # All four anchor roles populated
    assert set(bridge.anchor_nodes.keys()) == {"ac_a", "ac_b", "dc_pos", "dc_neg"}
    # dc_neg is the circuit ground in this construction
    assert bridge.anchor_nodes["dc_neg"] == ckt.ground()
    # ac_a and ac_b are distinct
    assert bridge.anchor_nodes["ac_a"] != bridge.anchor_nodes["ac_b"]


@_SPRING_ONLY
def test_bridge_template_places_diodes_in_diamond() -> None:
    """After compute_layout (spring backend only), the 4 bridge diodes
    sit in distinct quadrants around the bridge centroid (textbook
    diamond pattern from the in-tree template).

    ELK lays the bridge out as a layered ladder rather than a diamond —
    correct topologically, different geometrically — so the quadrant
    assertion no longer holds."""
    ckt = _build_bridge_only()
    layout = ps.schematic.compute_layout(ckt)

    diode_names = {"DBR__D1", "DBR__D2", "DBR__D3", "DBR__D4"}
    diode_centers = {n: (p.x, p.y) for n, p in layout.components.items() if n in diode_names}
    assert set(diode_centers.keys()) == diode_names

    # Centroid of the 4 diode centers
    cx = sum(c[0] for c in diode_centers.values()) / 4
    cy = sum(c[1] for c in diode_centers.values()) / 4

    # Each diode lives in a unique quadrant relative to centroid
    quadrants = {
        (dx > 0, dy > 0)
        for dx, dy in ((x - cx, y - cy) for x, y in diode_centers.values())
    }
    assert len(quadrants) == 4, (
        f"expected 4 distinct quadrants for textbook bridge, "
        f"got {quadrants} from centers {diode_centers}"
    )


# -----------------------------------------------------------------------------
# boost_stage recognizer
# -----------------------------------------------------------------------------


def _build_boost_only() -> ps.Circuit:
    ckt = ps.Circuit()
    vin = ckt.add_node("vin")
    sw_node = ckt.add_node("sw_node")
    vout = ckt.add_node("vout")
    gate = ckt.add_node("gate")
    gnd = ckt.ground()

    ckt.add_voltage_source("Vin", vin, gnd, 24.0)
    ckt.add_inductor("L1", vin, sw_node, 1e-3, 0.0)
    ckt.add_mosfet("Q1", gate, sw_node, gnd)
    ckt.add_diode("D1", sw_node, vout)
    ckt.add_capacitor("Cout", vout, gnd, 100e-6, 0.0)
    ckt.add_resistor("Rload", vout, gnd, 50.0)
    pulse = ps.PulseParams()
    pulse.v_initial = 0.0
    pulse.v_pulse = 12.0
    pulse.t_width = 5e-6
    pulse.period = 10e-6
    ckt.add_pulse_voltage_source("Vgate", gate, gnd, pulse)
    return ckt


def test_recognize_boost_stage_finds_L_Q_D() -> None:
    ckt = _build_boost_only()
    matches = recognize_all(list(ckt.components()))
    boost = [m for m in matches if m.template_id == "boost_stage"]
    assert len(boost) == 1
    b = boost[0]
    assert b.role_to_component == {"L": "L1", "Q": "Q1", "D": "D1"}
    assert b.anchor_nodes["dc_in"] == ckt.get_node("vin")
    assert b.anchor_nodes["sw_node"] == ckt.get_node("sw_node")
    assert b.anchor_nodes["vout"] == ckt.get_node("vout")
    assert b.anchor_nodes["gnd"] == ckt.ground()


def test_boost_template_lays_out_anchors_horizontally() -> None:
    """After compute_layout, dc_in / sw_node / vout sit in a row with
    sw_node between them on the x axis.
    """
    ckt = _build_boost_only()
    layout = ps.schematic.compute_layout(ckt)

    # Pull anchor coords back from terminal anchors of any templated comp.
    by_name = {c.name: c for c in ckt.components()}
    inductor = layout.components["L1"]
    diode = layout.components["D1"]
    inductor_node_to_anchor = {a.node: a for a in inductor.terminal_anchors}
    diode_node_to_anchor = {a.node: a for a in diode.terminal_anchors}

    dc_in_node = by_name["L1"].nodes[0] if by_name["L1"].nodes[0] != ckt.get_node("sw_node") else by_name["L1"].nodes[1]
    sw_node = ckt.get_node("sw_node")
    vout = ckt.get_node("vout")

    dc_in_x = inductor_node_to_anchor[dc_in_node].x
    sw_x = inductor_node_to_anchor[sw_node].x
    vout_x = diode_node_to_anchor[vout].x

    # sw_node sits between dc_in and vout horizontally
    assert dc_in_x < sw_x < vout_x or vout_x < sw_x < dc_in_x


# -----------------------------------------------------------------------------
# half_bridge recognizer
# -----------------------------------------------------------------------------


def _build_half_bridge() -> ps.Circuit:
    ckt = ps.Circuit()
    bus_pos = ckt.add_node("bus_pos")
    midpoint = ckt.add_node("midpoint")
    g_hi = ckt.add_node("g_hi")
    g_lo = ckt.add_node("g_lo")
    gnd = ckt.ground()

    ckt.add_voltage_source("Vbus", bus_pos, gnd, 400.0)
    ckt.add_mosfet("Q_hi", g_hi, bus_pos, midpoint)
    ckt.add_mosfet("Q_lo", g_lo, midpoint, gnd)
    ckt.add_resistor("Rload", midpoint, gnd, 10.0)

    pulse = ps.PulseParams()
    pulse.v_initial = 0.0
    pulse.v_pulse = 15.0
    pulse.t_width = 5e-6
    pulse.period = 10e-6
    ckt.add_pulse_voltage_source("Vgh", g_hi, gnd, pulse)
    ckt.add_pulse_voltage_source("Vgl", g_lo, gnd, pulse)
    return ckt


def test_recognize_half_bridge_finds_series_switches() -> None:
    ckt = _build_half_bridge()
    matches = recognize_all(list(ckt.components()))
    hb = [m for m in matches if m.template_id == "half_bridge"]
    assert len(hb) == 1
    m = hb[0]
    assert m.role_to_component == {"Q_hi": "Q_hi", "Q_lo": "Q_lo"}
    assert m.anchor_nodes["bus_pos"] == ckt.get_node("bus_pos")
    assert m.anchor_nodes["midpoint"] == ckt.get_node("midpoint")
    assert m.anchor_nodes["bus_neg"] == ckt.ground()


def test_half_bridge_template_stacks_switches_vertically() -> None:
    """After compute_layout, bus_pos.y < midpoint.y < bus_neg.y (screen
    convention: smaller y is higher).
    """
    ckt = _build_half_bridge()
    layout = ps.schematic.compute_layout(ckt)

    q_hi = layout.components["Q_hi"]
    q_lo = layout.components["Q_lo"]
    by_node_hi = {a.node: a for a in q_hi.terminal_anchors}
    by_node_lo = {a.node: a for a in q_lo.terminal_anchors}

    bus_pos_y = by_node_hi[ckt.get_node("bus_pos")].y
    midpoint_y = by_node_hi[ckt.get_node("midpoint")].y
    bus_neg_y = by_node_lo[ckt.ground()].y

    assert bus_pos_y < midpoint_y, "bus_pos should sit ABOVE midpoint (smaller y)"
    assert midpoint_y < bus_neg_y, "midpoint should sit ABOVE bus_neg (smaller y)"


# -----------------------------------------------------------------------------
# No-false-match cases
# -----------------------------------------------------------------------------


def test_recognize_returns_empty_on_pure_rc_circuit() -> None:
    ckt = ps.Circuit()
    n_in = ckt.add_node("in")
    n_out = ckt.add_node("out")
    gnd = ckt.ground()
    ckt.add_voltage_source("V1", n_in, gnd, 5.0)
    ckt.add_resistor("R1", n_in, n_out, 1000.0)
    ckt.add_capacitor("C1", n_out, gnd, 1e-6, 0.0)
    assert recognize_all(list(ckt.components())) == []


def test_recognize_does_not_match_buck_as_boost() -> None:
    """Buck and boost differ in diode/switch orientation. The buck
    converter YAML (synchronous-ish: switch high-side, freewheel diode)
    must NOT trigger the boost recognizer.
    """
    parser = ps.YamlParser(ps.YamlParserOptions())
    ckt, _opts = parser.load(str(BUCK_YAML))
    matches = recognize_all(list(ckt.components()))
    assert [m for m in matches if m.template_id == "boost_stage"] == []


# -----------------------------------------------------------------------------
# Chained-anchor lock: bridge + boost in a Boost PFC
# -----------------------------------------------------------------------------


def _build_boost_pfc_topology() -> ps.Circuit:
    """Boost PFC: bridge + boost share rect_pos (dc_pos of bridge = dc_in of boost)."""
    ckt = ps.Circuit()
    vac_a = ckt.add_node("vac_a")
    vac_b = ckt.add_node("vac_b")
    rect_pos = ckt.add_node("rect_pos")
    sw_node = ckt.add_node("sw_node")
    vout = ckt.add_node("vout")
    gate = ckt.add_node("gate")
    gnd = ckt.ground()

    sine = ps.SineParams()
    sine.amplitude = 230.0
    sine.frequency = 50.0
    ckt.add_sine_voltage_source("Vac", vac_a, vac_b, sine)
    ckt.add_bridge_rectifier("DBR", vac_a, vac_b, rect_pos, gnd, 0.7, 0.01)
    ckt.add_inductor("L1", rect_pos, sw_node, 1e-3, 0.0)
    ckt.add_mosfet("Q1", gate, sw_node, gnd)
    ckt.add_diode("D1", sw_node, vout)
    ckt.add_capacitor("Cout", vout, gnd, 470e-6, 0.0)
    ckt.add_resistor("Rload", vout, gnd, 100.0)
    pulse = ps.PulseParams()
    pulse.v_initial = 0.0
    pulse.v_pulse = 12.0
    pulse.t_width = 10e-6
    pulse.period = 20e-6
    ckt.add_pulse_voltage_source("Vgate", gate, gnd, pulse)
    return ckt


def test_boost_pfc_recognizes_both_bridge_and_boost() -> None:
    ckt = _build_boost_pfc_topology()
    matches = recognize_all(list(ckt.components()))
    ids = sorted(m.template_id for m in matches)
    assert ids == ["boost_stage", "bridge_rectifier"]

    # The shared node (rect_pos): bridge's dc_pos == boost's dc_in
    bridge = next(m for m in matches if m.template_id == "bridge_rectifier")
    boost = next(m for m in matches if m.template_id == "boost_stage")
    assert bridge.anchor_nodes["dc_pos"] == boost.anchor_nodes["dc_in"]


@_SPRING_ONLY
def test_ground_rail_routes_all_gnd_terminals_to_same_y() -> None:
    """Spring-backend ground-rail property: every gnd-terminal anchor
    lands on a single horizontal rail (same y).

    ELK doesn't collapse ground terminals to a single y — it places
    each gnd-port on the SOUTH side of its component, so y values vary
    by component height/position. The visual ground-rail effect is
    achieved by ELK's orthogonal routing rather than by anchor collapse."""
    ckt = _build_boost_only()
    layout = ps.schematic.compute_layout(ckt)
    ground_id = ckt.ground()

    rail_ys = []
    for placement in layout.components.values():
        for anchor in placement.terminal_anchors:
            if anchor.node == ground_id:
                rail_ys.append(anchor.y)

    # All gnd-terminal anchors share the same y (within float epsilon)
    assert rail_ys, "boost circuit must have at least one gnd-terminal"
    assert max(rail_ys) - min(rail_ys) < 1e-6, (
        f"expected one rail y, got {sorted(set(rail_ys))}"
    )


@_SPRING_ONLY
def test_parallel_gnd_pair_becomes_side_by_side_vertical_bars() -> None:
    """Spring-backend parallel-pair separation: Cout and Rload share
    vout↔gnd → after `_separate_parallel_devices` they sit at distinct
    x columns.

    ELK doesn't manipulate ports per device pair — when two devices share
    both nodes, ELK gives them separate layer slots (so their x can be
    equal if both are in the same layer). The visual separation comes
    from the orthogonal router, not from a perpendicular x-offset on
    the placement."""
    ckt = _build_boost_only()
    layout = ps.schematic.compute_layout(ckt)
    cout = layout.components["Cout"]
    rload = layout.components["Rload"]

    # Different x — fanned out perpendicular to the rail.
    assert abs(cout.x - rload.x) > 5.0

    # Both vertical (rotation perpendicular to the rail).
    assert cout.rotation in {90, 270}
    assert rload.rotation in {90, 270}

    # Each device's two terminals share an x (vertical bar).
    for placement in (cout, rload):
        xs = [a.x for a in placement.terminal_anchors]
        assert max(xs) - min(xs) < 1e-6


def test_three_terminal_switch_center_uses_drain_source_midpoint() -> None:
    """A boost-stage MOSFET's body sits between drain (sw_node) and
    source (gnd), NOT at the 3-way centroid that would pull it toward
    a far-away gate node.

    Regression guard for the case where a gate driven from the canvas
    west edge yanked Q1 into the middle of an unrelated sub-circuit
    (visually rendered Q1 on top of the bridge rectifier).
    """
    ckt = _build_boost_only()
    layout = ps.schematic.compute_layout(ckt)
    q1 = layout.components["Q1"]
    anchors = {a.node: a for a in q1.terminal_anchors}

    sw = anchors[ckt.get_node("sw_node")]
    gnd = anchors[ckt.ground()]
    gate = anchors[ckt.get_node("gate")]

    # Q1 center is the midpoint of drain (sw_node) and source (gnd).
    expected_cx = (sw.x + gnd.x) / 2.0
    expected_cy = (sw.y + gnd.y) / 2.0
    assert abs(q1.x - expected_cx) < 1e-6
    assert abs(q1.y - expected_cy) < 1e-6
    # And explicitly NOT the 3-way centroid (which would lean toward the gate).
    centroid_cx = (sw.x + gnd.x + gate.x) / 3.0
    centroid_cy = (sw.y + gnd.y + gate.y) / 3.0
    assert abs(q1.x - centroid_cx) > 0.1 or abs(q1.y - centroid_cy) > 0.1


def test_parallel_devices_snap_to_orthogonal_axis() -> None:
    """Cout and Rload share both vout↔gnd nodes; both should snap to
    the dominant orthogonal axis so the renderer doesn't stack them on
    the same diagonal.
    """
    ckt = _build_boost_only()
    layout = ps.schematic.compute_layout(ckt)
    cout = layout.components["Cout"]
    rload = layout.components["Rload"]

    # Both devices share the same node pair → after snap they're
    # rendered at orthogonal angles (0/90/180/270) only, never an
    # arbitrary diagonal.
    assert cout.rotation in {0, 90, 180, 270}
    assert rload.rotation in {0, 90, 180, 270}

    # And the two device centers are at different positions (no longer
    # overlapping at the same midpoint).
    assert (cout.x, cout.y) != (rload.x, rload.y)


def test_apply_templates_chains_shared_anchor_position() -> None:
    """The shared rect_pos node receives one position from the bridge
    (first-priority template) and the boost stage respects it (uses it
    as its translation pivot rather than overwriting).
    """
    ckt = _build_boost_pfc_topology()
    matches = recognize_all(list(ckt.components()))
    # Synthetic starting positions — irrelevant since both templates pin
    # via their anchors. We just need rect_pos in the input.
    rect_pos = ckt.get_node("rect_pos")
    gnd = ckt.ground()
    initial = {
        ckt.get_node("vac_a"):   (0.0, 0.0),
        ckt.get_node("vac_b"):   (10.0, 10.0),
        rect_pos:                (20.0, -20.0),
        ckt.get_node("sw_node"): (30.0, -30.0),
        ckt.get_node("vout"):    (40.0, -40.0),
        ckt.get_node("gate"):    (-50.0, -10.0),
        gnd:                     (15.0, 100.0),
    }
    out = apply_templates(matches, initial, ground_node=gnd)
    # Ground should be untouched
    assert out[gnd] == initial[gnd]
    # rect_pos got positioned by bridge; boost did not override it
    assert out[rect_pos] != initial[rect_pos]  # bridge moved it
    # Bridge anchor-offset frame: dc_neg at (0, +s), dc_pos at (0, -s).
    # Pivot = ground (the only locked node initially), so the translation
    # carries ground's local position (0, +s) onto its actual (15, 100).
    # That means dc_pos lands at ground.x + 0, ground.y - 2*s.
    expected_x = 15.0
    expected_y = 100.0 - 2 * _BRIDGE_SIZE_MM
    assert abs(out[rect_pos][0] - expected_x) < 1e-9
    assert abs(out[rect_pos][1] - expected_y) < 1e-9
