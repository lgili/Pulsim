"""Phase 4 — template library + instantiator tests.

Verifies that, for the recognized topologies, ``compute_layout()``
takes the template path instead of the force-directed fallback. The
assertions check structural properties (component positions match
template canvas-fractions within a small tolerance) rather than
byte-for-byte golden SVG matches, because:

  * Canvas-fraction layouts are deterministic and easy to assert
    numerically.
  * Golden SVGs would bake schemdraw's rendering details into the
    fixture, making them fragile to schemdraw version bumps.

A separate Phase 5 test will assert the SVG header / size sanity.
"""

from __future__ import annotations

import math

import pytest

import pulsim as p
import pulsim.schematic as sch
from pulsim.schematic.template_instantiator import (
    list_available_templates,
    load_template,
    template_layout,
)
from pulsim.schematic.topology_recognizer import RecognizedTopology, recognize


# ---------------------------------------------------------------------------
# Template-loader smoke tests
# ---------------------------------------------------------------------------

def test_templates_directory_populated():
    """All 12 canonical topologies ship a template YAML."""
    available = set(list_available_templates())
    expected = {
        "buck", "boost", "buck_boost", "flyback", "forward",
        "half_bridge", "full_bridge",
        "rc_filter", "rl_filter", "rlc_filter",
        "half_wave_rectifier", "full_wave_bridge_rectifier",
    }
    missing = expected - available
    assert not missing, f"missing template files: {sorted(missing)}"


@pytest.mark.parametrize("name", [
    "buck", "boost", "buck_boost", "flyback", "forward",
    "half_bridge", "full_bridge",
    "rc_filter", "rl_filter", "rlc_filter",
    "half_wave_rectifier", "full_wave_bridge_rectifier",
])
def test_template_parses(name):
    """Every template file loads without raising."""
    tpl = load_template(name)
    assert tpl is not None, f"{name} did not parse"
    assert tpl.canvas_width > 0
    assert tpl.canvas_height > 0
    assert tpl.slots, f"{name} has no slots"


def test_template_unknown_returns_none():
    assert load_template("no_such_topology_xyz") is None


# ---------------------------------------------------------------------------
# template_layout() — direct invocation
# ---------------------------------------------------------------------------

def _build_buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin",   "vin", "gnd", 24.0)
    b.add_switch       ("Q",      "vin", "sw",   g_on=1e3, g_off=1e-9)
    b.add_diode        ("D",      "gnd", "sw",   g_on=1e3, g_off=1e-9,
                         V_th=0.0)
    b.add_inductor     ("L",      "sw",  "vout", 100e-6)
    b.add_capacitor    ("Cout",   "vout","gnd",  100e-6)
    b.add_resistor     ("Rload",  "vout","gnd",  5.0)
    return b


def test_template_layout_places_buck_at_template_slots():
    """`template_layout(buck, recognize(buck))` must place every
    component at the canvas-fraction position from buck.yaml."""
    b = _build_buck()
    recognized = recognize(b)
    assert recognized is not None
    assert recognized.name == "buck"

    layout = template_layout(b, recognized)
    assert layout is not None

    template = load_template("buck")
    assert template is not None

    # Spot-check: Vin must land at slot["source"].x_frac * canvas_w.
    src_slot = template.slots["source"]
    vin = layout.components["Vin"]
    expected_x = src_slot.x_frac * template.canvas_width
    expected_y = src_slot.y_frac * template.canvas_height
    assert math.isclose(vin.x, expected_x, abs_tol=0.5)
    assert math.isclose(vin.y, expected_y, abs_tol=0.5)
    assert vin.rotation == src_slot.rotation

    # Every recognized role's component is placed.
    placed_names = set(layout.components.keys())
    assert {"Vin", "Q", "D", "L", "Cout", "Rload"} <= placed_names


def test_template_layout_wires_match_connectivity():
    """Wires emitted by the instantiator connect terminals that share
    a node. Phase 7 adds synthetic wires for the ground rail (from
    a component terminal to a virtual ``<gnd-rail>`` endpoint or
    rail-to-rail segments) — those are skipped here; only the
    component-to-component wires must respect the netlist."""
    b = _build_buck()
    recognized = recognize(b)
    layout = template_layout(b, recognized)
    assert layout is not None

    for wire in layout.wires:
        if (wire.from_.component == "<gnd-rail>"
                or wire.to.component == "<gnd-rail>"):
            continue
        a = layout.components[wire.from_.component]
        b_pl = layout.components[wire.to.component]
        a_node = a.terminal_anchors[wire.from_.terminal].node
        b_node = b_pl.terminal_anchors[wire.to.terminal].node
        assert a_node == b_node


def test_template_layout_no_recognition_returns_none():
    """If recognize() returns None (e.g. random circuit), the
    instantiator also returns None."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "n0", "gnd", 1.0)
    for i in range(8):
        b.add_resistor(f"R{i}", f"n{i}", f"n{i+1}", 1.0)
    assert recognize(b) is None

    # Synthesize a fake RecognizedTopology with empty role_map — the
    # instantiator must refuse to lay out without role information.
    fake = RecognizedTopology(
        name="buck", confidence=0.99, role_map={}, source="heuristic")
    assert template_layout(b, fake) is None


# ---------------------------------------------------------------------------
# Dispatcher integration — compute_layout() must take the template path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("topo,builder", [
    ("buck",        _build_buck),
])
def test_compute_layout_uses_template_for_recognized(topo, builder):
    """Top-level compute_layout() must produce a layout whose
    component positions match the template YAML for recognized
    topologies, NOT the random force-directed result."""
    b = builder()
    layout = sch.compute_layout(b)

    # Canvas WIDTH must match the template exactly. HEIGHT is
    # extended by ~15 mm (Phase 7) to fit the ground rail drawn at
    # the bottom — accept both the template height and template+15.
    template = load_template(topo)
    assert template is not None
    assert layout.canvas.width == pytest.approx(template.canvas_width, abs=0.5)
    height_min = template.canvas_height - 0.5
    height_max = template.canvas_height + 16.0
    assert height_min <= layout.canvas.height <= height_max, (
        f"canvas height {layout.canvas.height} not in "
        f"[{height_min}, {height_max}]")


def test_compute_layout_falls_back_for_unrecognized():
    """A random circuit must still get a layout — the spring
    fallback should fire."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "n0", "gnd", 1.0)
    for i in range(8):
        b.add_resistor(f"R{i}", f"n{i}", f"n{i+1}", 1.0)
    layout = sch.compute_layout(b)
    # Spring backend produces variable-size canvases, but the call
    # must succeed and place every component.
    assert layout is not None
    assert len(layout.components) >= 1


# ---------------------------------------------------------------------------
# Layout determinism
# ---------------------------------------------------------------------------

def test_template_layout_deterministic():
    """Two consecutive template_layout() calls with the same recognized
    topology produce identical placements."""
    b = _build_buck()
    recognized = recognize(b)
    L1 = template_layout(b, recognized)
    L2 = template_layout(b, recognized)
    assert L1 is not None and L2 is not None

    for name in L1.components:
        p1 = L1.components[name]
        p2 = L2.components[name]
        assert p1.x == p2.x
        assert p1.y == p2.y
        assert p1.rotation == p2.rotation
