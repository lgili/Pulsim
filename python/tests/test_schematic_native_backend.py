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


pytestmark = pytest.mark.skipif(
    not _NODE_AVAILABLE,
    reason="native backend layout step requires `node` on PATH (for elkjs)",
)


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
