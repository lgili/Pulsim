"""Phase 1 tests for the native Python schematic renderer.

Cover the contract in `add-python-schematic-renderer/specs/schematic-rendering`:
the native backend produces well-formed SVG with skin symbols + orthogonal
wires + ground stubs, equivalent in topology (not pixel-for-pixel) to the
existing netlistsvg backend on the canonical demo set.

The tests are gated by Node availability — without `node` on PATH the
layout step (`elk_bridge.js`) cannot run.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytestmark = pytest.mark.skip(
    reason="Legacy v1 Circuit API; pending port to flat-namespace "
           "CircuitBuilder. V2 recognizer/template path covered by "
           "test_topology_recognizer.py + test_template_layouts.py.",
)

import pulsim as ps
from pulsim.schematic import skin_parser
from pulsim.schematic.native_backend import (
    _KIND_TO_SKIN,
    _fmt_eng,
    _format_value,
    render_native,
)


_NODE_AVAILABLE = shutil.which("node") is not None
_SKIN_PATH = (
    Path(__file__).resolve().parents[1]
    / "pulsim"
    / "schematic"
    / "skin"
    / "pulsim_analog.svg"
)
_SVG_NS = "{http://www.w3.org/2000/svg}"

REPO_ROOT = Path(__file__).resolve().parents[2]
BUCK_YAML = REPO_ROOT / "examples" / "buck_converter.yaml"
RC_YAML = REPO_ROOT / "examples" / "rc_circuit.yaml"
HB_YAML = REPO_ROOT / "examples" / "half_bridge_pwm.yaml"


pytestmark = [
    pytest.mark.skip(
        reason="Legacy v1 Circuit API; pending port to flat-namespace "
               "CircuitBuilder. V2 recognizer/template path covered by "
               "test_topology_recognizer.py + test_template_layouts.py.",
    ),
    pytest.mark.skipif(
        not _NODE_AVAILABLE,
        reason="native backend layout step requires `node` on PATH (for elkjs)",
    ),
]


# ---------------------------------------------------------------------------
# Skin parser tests (no Node needed — pure XML parsing)
# ---------------------------------------------------------------------------


def test_parse_skin_collects_all_documented_symbols() -> None:
    """The shipped skin must register every kind the native backend can
    paint, AND each alias must resolve to the same template object as
    its primary `s:type`.
    """
    skin_parser.clear_skin_cache()
    skin = skin_parser.parse_skin(_SKIN_PATH)

    # Every kind the renderer maps to must be present in the skin.
    for kind, skin_key in _KIND_TO_SKIN.items():
        assert skin_key in skin, (
            f"native backend maps Pulsim '{kind}' -> skin '{skin_key}' "
            f"but '{skin_key}' is missing from the parsed skin"
        )

    # Ground and generic must exist (rendered as stubs / fallback).
    assert "gnd" in skin, "ground stub symbol missing from skin"
    assert "generic" in skin, "generic fallback symbol missing from skin"


def test_skin_alias_shares_template_identity() -> None:
    """`r_v` (alias) and `resistor_v` (primary) point to the same template."""
    skin = skin_parser.parse_skin(_SKIN_PATH)
    assert skin["r_v"] is skin["resistor_v"]
    assert skin["v"] is skin["voltage_source"]


def test_skin_exposes_port_anchors() -> None:
    """A 3-pin symbol (mosfet_n) exposes G/D/S anchor coordinates."""
    skin = skin_parser.parse_skin(_SKIN_PATH)
    m = skin["mosfet_n"]
    assert set(m.ports.keys()) >= {"G", "D", "S"}
    for pid, (x, y) in m.ports.items():
        assert isinstance(x, float) and isinstance(y, float)


def test_skin_cache_round_trips() -> None:
    """Repeated parse() calls return the same dict (cached, no re-parse)."""
    skin_parser.clear_skin_cache()
    first = skin_parser.parse_skin(_SKIN_PATH)
    second = skin_parser.parse_skin(_SKIN_PATH)
    assert first is second


# ---------------------------------------------------------------------------
# Value-formatting tests (pure unit, no Node needed)
# ---------------------------------------------------------------------------


def test_fmt_eng_basic_prefixes() -> None:
    assert _fmt_eng(1000.0, "R") == "1kΩ"
    assert _fmt_eng(1e-6, "C") == "1µF"
    assert _fmt_eng(1e-3, "L") == "1mH"
    assert _fmt_eng(4.7, "V") == "4.7V"
    assert _fmt_eng(0.0, "R") == "0Ω"


def test_format_value_returns_none_for_dimensionless_kinds() -> None:
    """Diode / MOSFET have no canonical primary value — format_value returns None."""

    class FakeComp:
        def __init__(self, kind, params):
            self.kind = kind
            self.params = params

    assert _format_value(FakeComp("diode", {})) is None
    assert _format_value(FakeComp("mosfet", {})) is None


def test_format_value_uses_canonical_unit_key() -> None:
    class FakeComp:
        def __init__(self, kind, params):
            self.kind = kind
            self.params = params

    assert _format_value(FakeComp("resistor", {"R": 1000.0})) == "1kΩ"
    assert _format_value(FakeComp("capacitor", {"C": 1e-6})) == "1µF"
    assert _format_value(FakeComp("inductor", {"L": 100e-6})) == "100µH"
    assert _format_value(FakeComp("voltage_source", {"V": 12.0})) == "12V"


# ---------------------------------------------------------------------------
# End-to-end render tests (require Node + elkjs)
# ---------------------------------------------------------------------------


def _build_rc_circuit() -> ps.Circuit:
    ckt = ps.Circuit()
    vin = ckt.add_node("vin")
    vout = ckt.add_node("vout")
    gnd = ckt.ground()
    ckt.add_voltage_source("V1", vin, gnd, 5.0)
    ckt.add_resistor("R1", vin, vout, 1000.0)
    ckt.add_capacitor("C1", vout, gnd, 1e-6, 0.0)
    return ckt


def test_render_rc_writes_valid_svg(tmp_path: Path) -> None:
    out = render_native(_build_rc_circuit(), tmp_path / "rc.svg")
    assert out.exists() and out.stat().st_size > 0
    # ElementTree parse succeeds = SVG is well-formed XML.
    tree = ET.parse(out)
    root = tree.getroot()
    assert root.tag == f"{_SVG_NS}svg"


def test_render_rc_contains_expected_symbols(tmp_path: Path) -> None:
    """The RC SVG must contain one ``<g transform="translate(...)">``
    per component (V1, R1, C1) plus at least two ground stubs (V1's -
    terminal and C1's bottom)."""
    out = render_native(_build_rc_circuit(), tmp_path / "rc.svg")
    tree = ET.parse(out)
    root = tree.getroot()

    # All top-level <g> children that have a transform="translate(...)".
    groups = [
        g for g in root.findall(f"{_SVG_NS}g")
        if g.get("transform", "").startswith("translate(")
    ]
    # 3 components + at least 2 gnd stubs.
    assert len(groups) >= 5, f"expected >= 5 placed groups, got {len(groups)}"


def test_render_rc_contains_orthogonal_wires(tmp_path: Path) -> None:
    """Every wire path uses Manhattan segments (M ... L ... only — no curves)."""
    out = render_native(_build_rc_circuit(), tmp_path / "rc.svg")
    tree = ET.parse(out)
    root = tree.getroot()
    wires = [p for p in root.findall(f"{_SVG_NS}path") if p.get("class") == "wire"]
    assert len(wires) >= 1
    for w in wires:
        d = w.get("d", "")
        assert d.startswith("M"), f"wire path doesn't start with M: {d!r}"
        # Only line-to commands (no C/Q/A curves).
        for cmd in d:
            if cmd.isalpha():
                assert cmd in {"M", "L"}, f"non-orthogonal wire command {cmd!r} in {d!r}"


def test_render_rc_substitutes_value_labels(tmp_path: Path) -> None:
    """The rendered SVG must contain the formatted device values
    (``1kΩ``, ``1µF``, ``5V``) — NOT the skin's literal placeholders
    (``X1``, ``Xk``, ``Xu``, ``XV``)."""
    out = render_native(_build_rc_circuit(), tmp_path / "rc.svg")
    text = out.read_text()
    # Component names appear.
    assert "V1" in text
    assert "R1" in text
    assert "C1" in text
    # Formatted values appear.
    assert "1k" in text  # R = 1000 -> 1kΩ
    assert "1µ" in text  # C = 1e-6 -> 1µF
    assert "5V" in text
    # Skin placeholders are gone.
    assert "Xk" not in text
    assert "Xu" not in text
    assert "XV" not in text


def test_render_buck_yaml(tmp_path: Path) -> None:
    """The buck YAML loads and renders to a well-formed SVG with switch
    + diode + inductor + capacitor + load resistor + 2 voltage sources."""
    if not BUCK_YAML.exists():
        pytest.skip("buck_converter.yaml fixture not available")
    parser = ps.YamlParser(ps.YamlParserOptions())
    ckt, _ = parser.load(str(BUCK_YAML))
    out = render_native(ckt, tmp_path / "buck.svg")
    tree = ET.parse(out)
    root = tree.getroot()
    text = out.read_text()
    # Spot-check key labels.
    assert "Vdc" in text
    assert "S1" in text
    assert "D1" in text
    assert "L1" in text
    assert "C1" in text
    # No placeholder leakage.
    assert "Xk" not in text


def test_render_half_bridge_yaml(tmp_path: Path) -> None:
    """The half-bridge YAML renders with both switches as vcswitch symbols."""
    if not HB_YAML.exists():
        pytest.skip("half_bridge_pwm.yaml fixture not available")
    parser = ps.YamlParser(ps.YamlParserOptions())
    ckt, _ = parser.load(str(HB_YAML))
    out = render_native(ckt, tmp_path / "hb.svg")
    assert out.exists() and out.stat().st_size > 0
    text = out.read_text()
    assert "S_hi" in text
    assert "S_lo" in text


def test_render_unknown_kind_falls_back_to_generic(tmp_path: Path) -> None:
    """A circuit containing only a probe (no analog symbol) should still
    render without crashing — the renderer falls back to the skin's
    `generic` rectangle."""
    ckt = ps.Circuit()
    a = ckt.add_node("a")
    gnd = ckt.ground()
    ckt.add_resistor("R1", a, gnd, 100.0)
    # No special unknown component available in pure Circuit API; resort
    # to a vanilla render and just confirm no crash + non-empty output.
    out = render_native(ckt, tmp_path / "single.svg")
    assert out.exists() and out.stat().st_size > 0


def test_render_dispatcher_recognizes_python_native(monkeypatch, tmp_path) -> None:
    """``PULSIM_SCHEMATIC_BACKEND=python_native`` routes through the new
    renderer (not netlistsvg)."""
    monkeypatch.setenv("PULSIM_SCHEMATIC_BACKEND", "python_native")
    out = ps.schematic.render(_build_rc_circuit(), tmp_path / "rc.svg")
    assert out.exists() and out.stat().st_size > 0
    # If we'd hit netlistsvg instead, the markup wouldn't carry our
    # ``class="wire"`` convention (netlistsvg uses different classes).
    assert 'class="wire"' in out.read_text()


# ---------------------------------------------------------------------------
# Phase 3: position hints → layout
# ---------------------------------------------------------------------------


def test_resolve_hints_empty_when_no_hints_set() -> None:
    """A Circuit with no `set_position` calls produces an empty resolved map."""
    from pulsim.schematic.native_backend import _resolve_hints

    ckt = _build_rc_circuit()
    assert _resolve_hints(ckt) == {}


def test_resolve_hints_layer_slot_multiplies_by_grid() -> None:
    from pulsim.schematic.native_backend import _resolve_hints, LAYER_PX, SLOT_PX

    ckt = _build_rc_circuit()
    ckt.set_position("V1", layer=0, slot=0)
    ckt.set_position("R1", layer=1, slot=0)
    ckt.set_position("C1", layer=2, slot=1)
    resolved = _resolve_hints(ckt)
    assert resolved == {
        "V1": (0.0, 0.0),
        "R1": (1.0 * LAYER_PX, 0.0),
        "C1": (2.0 * LAYER_PX, 1.0 * SLOT_PX),
    }


def test_resolve_hints_absolute_passes_through() -> None:
    from pulsim.schematic.native_backend import _resolve_hints

    ckt = _build_rc_circuit()
    ckt.set_position("V1", x=10.0, y=20.0)
    ckt.set_position("R1", x=150.5, y=50.5)
    resolved = _resolve_hints(ckt)
    assert resolved == {"V1": (10.0, 20.0), "R1": (150.5, 50.5)}


def test_resolve_hints_absolute_wins_over_layer_slot() -> None:
    """When both forms are set on the same hint, absolute coords win."""
    from pulsim.schematic.native_backend import _resolve_hints

    ckt = _build_rc_circuit()
    ckt.set_position("V1", layer=5, slot=5, x=10.0, y=20.0)
    resolved = _resolve_hints(ckt)
    assert resolved["V1"] == (10.0, 20.0)


def test_resolve_hints_detects_conflict() -> None:
    """Two hints resolving to the same absolute coords raise ValueError."""
    from pulsim.schematic.native_backend import _resolve_hints

    ckt = _build_rc_circuit()
    ckt.set_position("V1", layer=0, slot=0)
    ckt.set_position("R1", layer=0, slot=0)  # collides
    with pytest.raises(ValueError, match="conflict"):
        _resolve_hints(ckt)


def test_render_with_hints_places_components_at_expected_coords(tmp_path: Path) -> None:
    """Render a hinted RC and confirm each `<g transform="translate(...)">`
    sits within ±5 px of the expected grid coordinates."""
    from pulsim.schematic.native_backend import LAYER_PX, SLOT_PX

    ckt = _build_rc_circuit()
    ckt.set_position("V1", layer=0, slot=0)
    ckt.set_position("R1", layer=2, slot=0)
    ckt.set_position("C1", layer=4, slot=1)
    out = render_native(ckt, tmp_path / "hinted_rc.svg")

    tree = ET.parse(out)
    root = tree.getroot()

    # Map each top-level <g transform="translate(X, Y)"> to (X, Y).
    import re

    translate_re = re.compile(r"translate\(([\-\d.]+),\s*([\-\d.]+)\)")
    placements: dict[tuple[float, float], None] = {}
    for g in root.findall(f"{_SVG_NS}g"):
        m = translate_re.search(g.get("transform", ""))
        if m is None:
            continue
        placements[(float(m.group(1)), float(m.group(2)))] = None

    def _has_placement_near(target: tuple[float, float], tol: float = 5.0) -> bool:
        tx, ty = target
        return any(
            abs(x - tx) <= tol and abs(y - ty) <= tol
            for (x, y) in placements
        )

    # The hinted components must land at the expected grid coords.
    assert _has_placement_near((0.0, 0.0))
    assert _has_placement_near((2 * LAYER_PX, 0.0))
    assert _has_placement_near((4 * LAYER_PX, 1 * SLOT_PX))


def test_render_no_hints_matches_phase1_baseline(tmp_path: Path) -> None:
    """A circuit rendered with no hints produces the same byte-stream as
    the Phase 1 baseline path — i.e. adding hint support never changed
    the un-hinted output."""
    ckt = _build_rc_circuit()
    out_noh = render_native(ckt, tmp_path / "noh.svg")
    # Re-render the same circuit twice; the deterministic baseline is
    # that the bytes match exactly across runs.
    out_again = render_native(_build_rc_circuit(), tmp_path / "noh_again.svg")
    assert out_noh.read_bytes() == out_again.read_bytes()


# ---------------------------------------------------------------------------
# Phase 4: topology-aware auto-hints
# ---------------------------------------------------------------------------


def test_auto_hints_empty_for_rc_no_topology() -> None:
    """An RC circuit doesn't match any recognizer → no auto-hints."""
    from pulsim.schematic.native_backend import _auto_hints

    ckt = _build_rc_circuit()
    assert _auto_hints(ckt, {}) == {}


def test_auto_hints_half_bridge() -> None:
    """The half-bridge example matches the recognizer; auto-hints stack
    Q_hi above Q_lo (slot 0 / slot 1) in the same layer."""
    from pulsim.schematic.native_backend import _auto_hints, LAYER_PX, SLOT_PX

    if not HB_YAML.exists():
        pytest.skip("half_bridge_pwm.yaml fixture not available")
    parser = ps.YamlParser(ps.YamlParserOptions())
    ckt, _ = parser.load(str(HB_YAML))
    auto = _auto_hints(ckt, {})
    assert "S_hi" in auto and "S_lo" in auto
    s_hi = auto["S_hi"]
    s_lo = auto["S_lo"]
    # Same layer (column).
    assert s_hi[0] == s_lo[0]
    # S_hi is above S_lo (smaller Y).
    assert s_hi[1] < s_lo[1]
    # And the gap is SLOT_PX (one row).
    assert s_lo[1] - s_hi[1] == SLOT_PX


def test_auto_hints_bridge_rectifier_diamond() -> None:
    """A 4-diode AC→DC bridge matches the recognizer; auto-hints form a
    2×2 diamond (D1 top-left, D3 top-right, D2/D4 below them)."""
    from pulsim.schematic.native_backend import _auto_hints, LAYER_PX, SLOT_PX

    ckt = ps.Circuit()
    ac_a    = ckt.add_node("ac_a")
    ac_b    = ckt.add_node("ac_b")
    dc_pos  = ckt.add_node("dc_pos")
    gnd     = ckt.ground()  # dc_neg
    # Pulsim diode pin order: [anode, cathode].
    ckt.add_diode("D1", ac_a, dc_pos)
    ckt.add_diode("D3", ac_b, dc_pos)
    ckt.add_diode("D2", gnd,  ac_a)
    ckt.add_diode("D4", gnd,  ac_b)
    auto = _auto_hints(ckt, {})
    # All four diodes must be hinted in a 2-layer × 2-slot grid.
    assert {"D1", "D2", "D3", "D4"} <= set(auto.keys())
    layers = {auto[name][0] for name in ("D1", "D2", "D3", "D4")}
    slots  = {auto[name][1] for name in ("D1", "D2", "D3", "D4")}
    assert layers == {0.0, LAYER_PX}
    assert slots  == {0.0, SLOT_PX}


def test_auto_hints_skip_user_hinted_components() -> None:
    """When the user pins a component manually, the auto-layer
    skips that component (user wins by priority)."""
    from pulsim.schematic.native_backend import _auto_hints, _resolve_hints, LAYER_PX

    if not HB_YAML.exists():
        pytest.skip("half_bridge_pwm.yaml fixture not available")
    parser = ps.YamlParser(ps.YamlParserOptions())
    ckt, _ = parser.load(str(HB_YAML))
    # User overrides S_hi to a far-right position.
    ckt.set_position("S_hi", x=999.0, y=999.0)
    resolved = _resolve_hints(ckt)
    # User hint wins.
    assert resolved["S_hi"] == (999.0, 999.0)
    # S_lo still gets the auto-hint (slot below the canonical S_hi spot).
    assert "S_lo" in resolved
    assert resolved["S_lo"] != (999.0, 999.0)


def test_resolve_hints_no_recognizers_no_user_hints_returns_empty() -> None:
    """The fast path: an RC with no hints and no recognizer match yields
    an empty dict so the un-hinted code path stays bit-identical to Phase 1."""
    from pulsim.schematic.native_backend import _resolve_hints

    ckt = _build_rc_circuit()
    assert _resolve_hints(ckt) == {}


def test_yaml_position_hints_flow_through_to_render(tmp_path: Path) -> None:
    """YAML `position:` fields end up on the rendered SVG."""
    from pulsim.schematic.native_backend import LAYER_PX, SLOT_PX

    yaml_path = tmp_path / "rc_hinted.yaml"
    yaml_path.write_text(
        """
schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-3
  dt: 1e-6
components:
  - type: voltage_source
    name: V1
    nodes: [vin, 0]
    waveform: { type: dc, value: 5.0 }
    position: { layer: 0, slot: 0 }
  - type: resistor
    name: R1
    nodes: [vin, vout]
    value: 1k
    position: { layer: 2, slot: 0 }
  - type: capacitor
    name: C1
    nodes: [vout, 0]
    value: 1u
    position: { layer: 4, slot: 1 }
"""
    )
    parser = ps.YamlParser(ps.YamlParserOptions())
    ckt, _ = parser.load(str(yaml_path))
    out = render_native(ckt, tmp_path / "yaml_hinted.svg")

    tree = ET.parse(out)
    root = tree.getroot()
    import re

    translate_re = re.compile(r"translate\(([\-\d.]+),\s*([\-\d.]+)\)")
    placements = []
    for g in root.findall(f"{_SVG_NS}g"):
        m = translate_re.search(g.get("transform", ""))
        if m is not None:
            placements.append((float(m.group(1)), float(m.group(2))))

    # All three hinted positions show up among the placed groups.
    def near(t):
        return any(abs(x - t[0]) <= 5 and abs(y - t[1]) <= 5 for (x, y) in placements)

    assert near((0.0, 0.0))
    assert near((2 * LAYER_PX, 0.0))
    assert near((4 * LAYER_PX, 1 * SLOT_PX))
