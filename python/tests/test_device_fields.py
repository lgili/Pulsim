"""add-unified-device-interface group 12: the device field tables.

One table per kind (``ir::kKindFields``: name, unit, bound, default,
description) drives the bound check of ``with_value`` / ``bind_param`` /
the YAML reader (not the ``add_*`` methods'), the ``components()`` params, the YAML registry keys and the generated device reference
(``docs/device-reference.md``, kept current by
``scripts/generate_device_reference.py --check``, which this module runs).
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

import pulsim as p
from pulsim import _pulsim as k

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "generate_device_reference.py"


def _gen():
    spec = importlib.util.spec_from_file_location("gen_devref", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not SCRIPT.exists(), reason="source checkout only")
def test_the_device_reference_page_is_current():
    gen = _gen()
    assert gen.PAGE.read_text(encoding="utf-8") == gen.render(), (
        "docs/device-reference.md is stale: run "
        "python scripts/generate_device_reference.py")


def test_every_kind_has_a_documented_row():
    rows = k.device_field_table()
    assert len(rows) == len(k.StoredKind.__members__)
    for r in rows:
        for f in r["fields"]:
            assert f["unit"] and f["description"], (r["kind"], f["name"])
            assert set(f) == {"name", "key", "unit", "bound", "default",
                              "description", "listed", "derived"}


def test_device_fields_has_one_shape_for_builtin_and_external_kinds():
    sh = p.device_fields("shockley_diode")
    assert [f["name"] for f in sh] == ["I_S", "n", "V_T", "G_min", "i_lim", "BV"]
    assert sh[0]["unit"] == "A" and sh[0]["default"] == 1e-12

    class _Dev:   # the shape of pulsim.xdev.Device.fields
        fields = [("L_0", 1e-3, "H", 0.0, float("inf"))]
    ex = p.device_fields(_Dev())
    assert set(ex[0]) == set(sh[0])
    assert ex[0]["name"] == "L_0" and ex[0]["unit"] == "H" and ex[0]["default"] == 1e-3
    with pytest.raises(KeyError, match="no device kind"):
        p.device_fields("flux_capacitor")


# ---- components() params are the listed rows; devices() has every row -------

def _newton_circuit():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 1.0)
    b.add_resistor("R", "in", "a", 1.0)
    b.add_shockley_diode("D", "a", "gnd", I_S=2e-12, n=1.5, BV=40.0)
    b.add_sine_voltage_source("S", "s", "gnd", 0.5, 2.0, 50.0)
    return b


def test_components_lists_exactly_the_listed_rows_and_devices_has_all():
    b = _newton_circuit()
    comp = {d["name"]: d for d in b.components()}
    for name, kind in (("R", "resistor"), ("D", "shockley_diode"), ("S", "sine_voltage_source")):
        listed = [f["key"] for f in p.device_fields(kind) if f["listed"]]
        assert list(comp[name]["params"]) == listed
    assert comp["R"]["params"] == {"R_ohms": 1.0}
    assert comp["S"]["params"] == {"frequency": 50.0}   # the legacy surface, unchanged
    assert comp["D"]["params"] == {}
    dev = {d["name"]: d for d in b.build().devices()}
    assert dev["D"]["fields"] == dict(I_S=2e-12, n=1.5, V_T=0.025852, G_min=1e-12,
                                      i_lim=1e6, BV=40.0)
    assert list(dev["S"]["fields"]) == [f["name"] for f in p.device_fields("sine_voltage_source")]


# ---- builder validation names the device and the field ----------------------

def test_builder_errors_name_the_device_and_the_field():
    with pytest.raises(ValueError, match=r'add_shockley_diode\("D"\): .*I_S'):
        p.CircuitBuilder().add_shockley_diode("D", "a", "gnd", I_S=-1.0)
    with pytest.raises(ValueError, match=r'add_lauritzen_diode\("DL"\): .*T_M'):
        p.CircuitBuilder().add_lauritzen_diode("DL", "a", "gnd", tau=1e-8, T_M=1e-7)


def test_the_builder_does_not_apply_the_table_bounds():
    # Pinned contracts: the builder accepts what with_value / the YAML
    # loader refuse (a negative resistor, m = 0 as a linear capacitor, an
    # ideal branch-form switch's g_off = 0); the table bound is theirs.
    b = p.CircuitBuilder()
    b.add_resistor("Rn", "a", "gnd", -5.0)
    b.add_nonlinear_capacitor("C", "a", "gnd", 1e-9, 10.0, 0.0)
    b.add_switch("S", "a", "gnd", form="branch")       # ideal: g_off = 0
    c = b.build()
    with pytest.raises(ValueError, match=r'field R of device "Rn" must be a finite number > 0'):
        c.with_value("Rn.R", -1.0)


# ---- YAML: the registry entry of the Newton kinds round-trips ---------------

def _yaml_netlist(entries):
    devs = [{"type": "voltage_source", "name": "V", "from": "in", "to": "gnd", "V": 1.0},
            {"type": "resistor", "name": "R", "from": "in", "to": "a", "R": 1.0}] + entries
    return yaml.safe_dump({"circuit": {"devices": devs},
                           "simulation": {"t_end": 2e-6, "dt": 1e-8}})


def _builder_pair(kind):
    """(builder-made circuit, the same device as a YAML registry entry)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 1.0)
    b.add_resistor("R", "in", "a", 1.0)
    if kind == "shockley_diode":
        b.add_shockley_diode("X", "a", "gnd", I_S=3e-12, n=1.2, V_T=0.026, G_min=1e-11, BV=50.0)
        e = dict(I_S=3e-12, n=1.2, V_T=0.026, G_min=1e-11, BV=50.0)
    elif kind == "lauritzen_diode":
        b.add_lauritzen_diode("X", "a", "gnd", I_S=2e-12, n=1.1, tau=3e-7, T_M=2e-8)
        e = dict(I_S=2e-12, n=1.1, tau=3e-7, T_M=2e-8)
    else:
        b.add_nonlinear_capacitor("X", "a", "gnd", 2e-9, 0.8, 0.33)
        e = dict(C0=2e-9, V0=0.8, m=0.33)
    return b, {"type": kind, "name": "X", "pins": ["a", "gnd"], **e}


def _same_run(b1, b2, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r1 = p.simulate(b1, **kw)
        r2 = p.simulate(b2, **kw)
    np.testing.assert_array_equal(np.asarray(r1.times), np.asarray(r2.times))
    np.testing.assert_array_equal(np.asarray(r1.states), np.asarray(r2.states))


@pytest.mark.parametrize("kind", ["shockley_diode", "lauritzen_diode", "nonlinear_capacitor"])
def test_yaml_registry_entry_round_trips_and_simulates_bit_for_bit(kind):
    b, entry = _builder_pair(kind)
    by = p.load_yaml_string(_yaml_netlist([entry])).builder
    assert by.components() == b.components()
    fb = {d["name"]: d["fields"] for d in b.build().devices()}
    fy = {d["name"]: d["fields"] for d in by.build().devices()}
    assert fy == fb
    for key, v in entry.items():
        if key not in ("type", "name", "pins"):
            assert fy["X"][key] == v
    _same_run(b, by, t_end=2e-6, dt=1e-8, engine="pwl")


def test_pmsm_mna_yaml_round_trips_and_simulates_bit_for_bit():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, 5.0, 50.0)
    b.add_pmsm_mna("M", "a", "gnd", "gnd", "gnd", "w", "th", 0.4, 1e-3, 1.5e-3, 0.06, 3.0,
                   1e-4, B=1e-3, T_load=0.1, L_0=1e-3, theta0=0.3)
    # the emitter's document (a `kind: pmsm_mna` registry entry) as a netlist
    text = b.build().to_yaml()
    by = p.load_yaml_string(text).builder
    pm = [d for d in yaml.safe_load(text)["devices"] if d["kind"] == "pmsm_mna"]
    assert len(pm) == 1 and pm[0]["L_q"] == 1.5e-3
    assert by.components() == b.components()
    fa = {d["name"]: d["fields"] for d in b.build().devices()}
    fb = {d["name"]: d["fields"] for d in by.build().devices()}
    assert fa == fb
    m = [f for n, f in fa.items() if "psi_pm" in f]
    assert m and m[0]["psi_pm"] == 0.06 and m[0]["T_load"] == 0.1 and m[0]["L_q"] == 1.5e-3
    _same_run(b, by, t_end=1e-3, dt=1e-5, engine="pwl")
