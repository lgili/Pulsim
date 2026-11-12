"""``pulsim.schematic.render()`` on 2.0 builders.

Until 2.0.x the default backend (``python_native``) raised
``AttributeError: 'str' object has no attribute 'x'`` for EVERY
``CircuitBuilder``: the builder's V0 ``position_hints()`` stub maps node
ids to role strings (``{0: 'internal', -1: 'ground'}``) and the backend
dereferenced them as placement-hint objects. Measured before the fix:
0 of the 25 v2 examples rendered.

Now the default is the pure-Python layout path (topology template, else
force-directed), Node-dependent backends name what is missing, and no
input reaches the user as an ``AttributeError``.
"""
from __future__ import annotations

import shutil
import subprocess
import xml.etree.ElementTree as ET

import pytest

import pulsim as p

sch = pytest.importorskip("pulsim.schematic")
pytest.importorskip("yaml")          # templates are YAML files


@pytest.fixture(autouse=True)
def _offline_default_backend(monkeypatch):
    # Default backend, and never the Tier-2 LLM classifier (network).
    monkeypatch.delenv("PULSIM_SCHEMATIC_BACKEND", raising=False)
    monkeypatch.setenv("PULSIM_LLM_LAYOUT_HINTS", "0")


def _rc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "in", "gnd", 10.0)
    b.add_resistor("R1", "in", "out", 1e3)
    b.add_capacitor("C1", "out", "gnd", 1e-6)
    return b


def _rlc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "in", "gnd", 10.0)
    b.add_inductor("L1", "in", "m", 1e-3)
    b.add_resistor("R1", "m", "out", 10.0)
    b.add_capacitor("C1", "out", "gnd", 1e-6)
    return b


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("Q", "vin", "sw", g_on=1e3, g_off=1e-9)
    b.add_diode("D_FW", "gnd", "sw", g_on=1e3, g_off=1e-9, V_th=0.0)
    b.add_inductor("L1", "sw", "vout", 100e-6)
    b.add_capacitor("Cout", "vout", "gnd", 100e-6)
    b.add_resistor("Rload", "vout", "gnd", 5.0)
    return b


def _mosfet_buck():
    # MOSFET + body diode: no template matches -> force-directed path.
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_mosfet("Q1", "vin", "sw")
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("R", "vout", "gnd", 10.0)
    return b


def _texts(svg_path):
    return {el.text for el in ET.parse(svg_path).iter()
            if el.tag.endswith("text") and el.text}


def test_position_hints_stub_is_not_read_as_placement_hints():
    from pulsim.schematic.native_backend import _resolve_user_hints
    b = _rc()
    raw = b.position_hints()
    assert raw and all(isinstance(v, str) for v in raw.values())
    assert _resolve_user_hints(sch._maybe_adapt(b)) == {}


@pytest.mark.parametrize("make, template", [
    (_rc, True), (_rlc, True), (_buck, True), (_mosfet_buck, False),
])
def test_default_render_writes_an_svg_with_every_component(
        tmp_path, make, template):
    from pulsim.schematic.layout import _try_template_layout
    b = make()
    assert (_try_template_layout(sch._maybe_adapt(b)) is not None) == template
    if not template:
        pytest.importorskip("networkx")
    out = sch.render(b, tmp_path / "c.svg")
    assert out.stat().st_size > 0
    names = {c["name"] for c in b.components()}
    assert names <= _texts(out)


def test_render_accepts_a_loaded_circuit(tmp_path):
    lc = p.load_yaml_string("""
circuit:
  devices:
    - {type: voltage_source, name: V1, from: a, to: gnd, V: 5.0}
    - {type: resistor,       name: R1, from: a, to: b,   R: 1.0e3}
    - {type: capacitor,      name: C1, from: b, to: gnd, C: 1.0e-6}
simulation: {t_end: 1.0e-3, dt: 1.0e-6}
""")
    out = sch.render(lc, tmp_path / "lc.svg")
    assert {"V1", "R1", "C1"} <= _texts(out)


def test_empty_builder_renders(tmp_path):
    pytest.importorskip("networkx")    # no template fits; the layout needs it
    assert sch.render(p.CircuitBuilder(), tmp_path / "e.svg").exists()


def test_non_circuit_is_a_type_error(tmp_path):
    with pytest.raises(TypeError, match="expects a CircuitBuilder"):
        sch.render(42, tmp_path / "x.svg")


def test_unknown_backend_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("PULSIM_SCHEMATIC_BACKEND", "bogus")
    with pytest.raises(ValueError, match="'bogus' is not a schematic backend"):
        sch.render(_rc(), tmp_path / "x.svg")


def test_png_either_writes_or_names_the_missing_library(tmp_path):
    try:
        out = sch.render(_rc(), tmp_path / "x.png")
    except ImportError as exc:                    # not OSError / AttributeError
        assert "cairo" in str(exc)
    else:
        assert out.stat().st_size > 0


def _elkjs_available():
    node = shutil.which("node")
    if node is None:
        return False
    from pulsim.schematic.native_backend import _BRIDGE_JS
    r = subprocess.run([node, "-e", "require.resolve('elkjs')"],
                       cwd=str(_BRIDGE_JS.parent), capture_output=True)
    return r.returncode == 0


@pytest.mark.skipif(_elkjs_available(), reason="elkjs is installed here")
@pytest.mark.parametrize("backend", ["python_native", "elk"])
def test_elk_backends_name_the_missing_elkjs(tmp_path, monkeypatch, backend):
    if shutil.which("node") is None:
        pytest.skip("node not on PATH (the ImportError names Node instead)")
    monkeypatch.setenv("PULSIM_SCHEMATIC_BACKEND", backend)
    with pytest.raises(ImportError, match="elkjs"):
        sch.render(_mosfet_buck(), tmp_path / "x.svg")
