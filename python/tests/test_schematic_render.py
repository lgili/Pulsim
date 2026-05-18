"""Phase 3 tests for ``pulsim.schematic.render`` (SVG + PNG via schemdraw).

Cover gates G.2 (PNG renders for buck_converter.yaml), G.3 (SVG renders +
parseable as XML), G.7 (lazy schemdraw import gate), and the
graceful-handling-of-unknown-kind spec scenario.
"""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import pulsim as ps


REPO_ROOT = Path(__file__).resolve().parents[2]
BUCK_YAML = REPO_ROOT / "examples" / "buck_converter.yaml"


# schemdraw is an optional dep — if the [schematic] extra isn't installed,
# every render test below will skip. Layout-only tests in
# test_schematic_layout.py still run.
pytest.importorskip("schemdraw")
pytest.importorskip("networkx")


def _build_rc_circuit() -> ps.Circuit:
    ckt = ps.Circuit()
    n_in = ckt.add_node("in")
    n_out = ckt.add_node("out")
    gnd = ckt.ground()
    ckt.add_voltage_source("V1", n_in, gnd, 12.0)
    ckt.add_resistor("R1", n_in, n_out, 1_000.0)
    ckt.add_capacitor("C1", n_out, gnd, 1e-6, 0.0)
    return ckt


# -----------------------------------------------------------------------------
# Gate G.3: SVG render is well-formed XML
# -----------------------------------------------------------------------------


def test_render_svg_writes_well_formed_svg(tmp_path: Path) -> None:
    out = ps.schematic.render(_build_rc_circuit(), tmp_path / "rc.svg")
    assert out.exists()
    assert out.stat().st_size > 0

    # Parseable by stdlib XML
    tree = ET.parse(out)
    root = tree.getroot()
    assert root.tag.endswith("svg")
    # Width / height set on the root
    assert "width" in root.attrib
    assert "height" in root.attrib


def test_render_svg_for_buck_yaml(tmp_path: Path) -> None:
    parser = ps.YamlParser(ps.YamlParserOptions())
    ckt, _opts = parser.load(str(BUCK_YAML))
    out = ps.schematic.render(ckt, tmp_path / "buck.svg")
    assert out.exists()
    assert out.stat().st_size > 0
    ET.parse(out)  # raises on malformed XML


# -----------------------------------------------------------------------------
# Gate G.2: PNG render produces a valid PNG
# -----------------------------------------------------------------------------


def test_render_png_writes_valid_png(tmp_path: Path) -> None:
    out = ps.schematic.render(_build_rc_circuit(), tmp_path / "rc.png")
    assert out.exists()
    assert out.stat().st_size > 0
    # PNG file signature: 89 50 4E 47 0D 0A 1A 0A
    with open(out, "rb") as fp:
        sig = fp.read(8)
    assert sig == b"\x89PNG\r\n\x1a\n"


def test_render_png_for_buck_yaml(tmp_path: Path) -> None:
    parser = ps.YamlParser(ps.YamlParserOptions())
    ckt, _opts = parser.load(str(BUCK_YAML))
    out = ps.schematic.render(ckt, tmp_path / "buck.png")
    assert out.exists()
    assert out.stat().st_size > 0


# -----------------------------------------------------------------------------
# Format dispatch: extension vs explicit `format=`
# -----------------------------------------------------------------------------


def test_render_explicit_format_override(tmp_path: Path) -> None:
    """format='svg' wins over an arbitrary extension."""
    out = tmp_path / "anyname.bin"
    written = ps.schematic.render(_build_rc_circuit(), out, format="svg")
    assert written.exists()
    # Resulting file is SVG (xml-parseable) despite the .bin extension
    ET.parse(written)


def test_render_unknown_extension_without_format_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="format"):
        ps.schematic.render(_build_rc_circuit(), tmp_path / "anyname.bin")


# -----------------------------------------------------------------------------
# render_layout: skip the compute_layout step
# -----------------------------------------------------------------------------


def test_render_layout_accepts_precomputed_layout(tmp_path: Path) -> None:
    layout = ps.schematic.compute_layout(_build_rc_circuit())
    out = ps.schematic.render_layout(layout, tmp_path / "from_layout.svg")
    assert out.exists()
    ET.parse(out)


# -----------------------------------------------------------------------------
# Graceful handling of unknown component kinds (spec scenario)
# -----------------------------------------------------------------------------


def test_render_unknown_kind_does_not_crash(tmp_path: Path) -> None:
    """A SchematicLayout with an unmapped kind renders as a labeled box."""
    layout = ps.schematic.compute_layout(_build_rc_circuit())
    # Mutate one placement to carry a kind not in the symbol map.
    name, original = next(iter(layout.components.items()))
    mutated = ps.schematic.ComponentPlacement(
        name=original.name,
        kind="some_exotic_unknown_kind",
        x=original.x,
        y=original.y,
        rotation=original.rotation,
        terminal_anchors=original.terminal_anchors,
    )
    new_components = {**layout.components, name: mutated}
    mutated_layout = ps.schematic.SchematicLayout(
        components=new_components,
        wires=layout.wires,
        junctions=layout.junctions,
        canvas=layout.canvas,
        schema_version=layout.schema_version,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = ps.schematic.render_layout(mutated_layout, tmp_path / "with_unknown.svg")
        # At least one UserWarning naming the unknown kind
        unknown_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "some_exotic_unknown_kind" in str(w.message)
        ]
        assert unknown_warnings, "expected UserWarning naming the unknown kind"
    assert out.exists()
    ET.parse(out)


# -----------------------------------------------------------------------------
# Lazy import (G.7): pulsim.schematic always importable; only render() raises
# -----------------------------------------------------------------------------


def test_schematic_top_level_is_lazy_importable() -> None:
    """`import pulsim.schematic` succeeds without schemdraw being needed
    at import time. This test runs while schemdraw IS installed, but
    asserts the surface still re-exports render/compute_layout as
    documented."""
    import pulsim.schematic as ps_schematic
    assert hasattr(ps_schematic, "render")
    assert hasattr(ps_schematic, "render_layout")
    assert hasattr(ps_schematic, "compute_layout")


# -----------------------------------------------------------------------------
# Pulsim analog skin (add-pulsim-analog-skin): switching devices render
# with dedicated MOSFET / IGBT / vcswitch symbols, not generic boxes.
# -----------------------------------------------------------------------------

# These tests only apply when the default netlistsvg backend is active.
# They rely on the Pulsim-extended analog skin (pulsim_analog.svg) being
# present in the build tree; CI environments without Node.js will skip
# the netlistsvg backend entirely.
import os

_NETLISTSVG_DEFAULT = pytest.mark.skipif(
    os.environ.get("PULSIM_SCHEMATIC_BACKEND", "netlistsvg").lower() != "netlistsvg",
    reason="Analog-skin symbol tests apply only to the netlistsvg backend.",
)


@_NETLISTSVG_DEFAULT
def test_vcswitch_renders_with_pulsim_skin_symbol(tmp_path: Path) -> None:
    """The buck converter's vcswitch (S1) should render with the
    dedicated `vcswitch` symbol from pulsim_analog.svg, not as a
    `generic` fallback rectangle.
    """
    parser = ps.YamlParser(ps.YamlParserOptions())
    ckt, _opts = parser.load(str(BUCK_YAML))
    out = ps.schematic.render(ckt, tmp_path / "buck.svg")
    svg_text = out.read_text()
    # The Pulsim skin tags the vcswitch glyph with `s:type="vcswitch"`.
    assert 's:type="vcswitch"' in svg_text, (
        "buck_converter.yaml's S1 should render with the vcswitch symbol "
        "from pulsim_analog.svg, but the SVG contains no s:type=\"vcswitch\" tag."
    )
    # And NOT fall back to the generic labeled-rectangle for S1.
    assert ('id="cell_S1"' not in svg_text) or ('s:type="generic"' not in svg_text.split('id="cell_S1"')[0].rsplit('<g', 1)[1]), \
        "S1 should not be a generic-cell fallback"


@_NETLISTSVG_DEFAULT
def test_mosfet_renders_with_pulsim_skin_symbol(tmp_path: Path) -> None:
    """A circuit with a MOSFET renders with the `mosfet_n` symbol."""
    ckt = ps.Circuit()
    vbus = ckt.add_node("vbus")
    sw = ckt.add_node("sw")
    g = ckt.add_node("g")
    gnd = ckt.ground()
    ckt.add_voltage_source("Vbus", vbus, gnd, 12.0)
    ckt.add_mosfet("Q1", g, vbus, sw)
    ckt.add_resistor("Rload", sw, gnd, 10.0)
    pulse = ps.PulseParams()
    pulse.v_initial = 0.0
    pulse.v_pulse = 5.0
    pulse.t_width = 5e-6
    pulse.period = 10e-6
    ckt.add_pulse_voltage_source("Vg", g, gnd, pulse)

    out = ps.schematic.render(ckt, tmp_path / "mosfet.svg")
    svg_text = out.read_text()
    assert 's:type="mosfet_n"' in svg_text, (
        "MOSFET Q1 should render with the mosfet_n symbol from "
        "pulsim_analog.svg."
    )


@_NETLISTSVG_DEFAULT
def test_passive_only_circuit_does_not_emit_switching_symbols(tmp_path: Path) -> None:
    """Regression guard: a pure RC circuit (no switching devices) renders
    WITHOUT any of the new pulsim_analog.svg switching-device symbols —
    confirms the skin's switching glyphs are only emitted on demand,
    not as a side effect of loading the skin.
    """
    ckt = ps.Circuit()
    n_in = ckt.add_node("in")
    n_out = ckt.add_node("out")
    gnd = ckt.ground()
    ckt.add_voltage_source("V1", n_in, gnd, 5.0)
    ckt.add_resistor("R1", n_in, n_out, 1000.0)
    ckt.add_capacitor("C1", n_out, gnd, 1e-6, 0.0)

    out = ps.schematic.render(ckt, tmp_path / "rc.svg")
    svg_text = out.read_text()
    for switching_kind in ("mosfet_n", "mosfet_p", "igbt", "vcswitch"):
        assert f's:type="{switching_kind}"' not in svg_text, (
            f"Pure RC circuit should not emit {switching_kind!r} symbol, "
            "but the rendered SVG contains it."
        )
