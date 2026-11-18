"""add-immutable-circuit-ir group 4: ``Circuit.to_yaml`` / ``from_yaml``,
``clone``, pickle and ``sweep(circuit, backend="process")``.

For one circuit per registry kind (the engine-conformance cells), every
shipped example and every harness circuit: ``from_yaml(to_yaml(c))`` has
the same hashes, ``to_yaml`` is a fixed point after one round, and the
round-tripped circuit simulates bit for bit like the original (or is
refused with the same message).
"""
from __future__ import annotations

import pickle
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import pulsim as p
from pulsim import _pulsim as k

sys.path.insert(0, str(Path(__file__).parent))
from test_engine_conformance import (  # noqa: E402
    CELLS, FORM_CELLS, VARIANT_CELLS, _switch_fn)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = [q for q in sorted((ROOT / "examples").glob("*.yaml"))
            if q.name != "refrigerator_compressor_cc.yaml"]   # v1, refused
V1_CASES = sorted((ROOT / "benchmarks" / "circuits").rglob("*.yaml"))
STEPS = 50


def _round(c):
    y = c.to_yaml()
    c2 = p.Circuit.from_yaml(y)
    assert c2.value_hash() == c.value_hash()
    assert c2.structure_hash() == c.structure_hash()
    assert c2.to_yaml() == y
    return c2


def _run(c, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return p.simulate(c, **kw)
        except Exception as e:  # noqa: BLE001 — compared, not swallowed
            return (type(e).__name__, str(e))


def _same_run(c, c2, **kw):
    a, b = _run(c, **kw), _run(c2, **kw)
    if isinstance(a, tuple) or isinstance(b, tuple):
        assert a == b
        return
    np.testing.assert_array_equal(np.asarray(a.times), np.asarray(b.times))
    np.testing.assert_array_equal(np.asarray(a.states), np.asarray(b.states))


ALL_CELLS = {**CELLS, **{n: c[0] for n, c in VARIANT_CELLS.items()},
             **{f"{n}[branch]": c for n, c in FORM_CELLS.items()}}


@pytest.mark.parametrize("engine", ("pwl", "trbdf2", "dsed"))
@pytest.mark.parametrize("name", list(ALL_CELLS))
def test_one_circuit_per_kind_round_trips_bit_identically(name, engine):
    build, _node, _dev, t_end, dt, needs_sw = ALL_CELLS[name]
    b = build(True)
    c = b.build()
    kw = {"engine": engine, "switch_fn": _switch_fn(b, needs_sw)}
    if engine == "pwl":
        kw["dt"] = dt
    _same_run(c, _round(c), t_end=min(t_end, STEPS * dt), **kw)


def test_every_registry_kind_has_a_yaml_entry():
    # The C++ side refuses to compile a kind without a name or a reader
    # (static_asserts in yaml/emitter.hpp); this pins that the cells
    # above write every kind, so each one is round-tripped.
    kinds = set()
    for build, *_ in ALL_CELLS.values():
        kinds |= set(re.findall(r"kind: (\w+)", build(True).build().to_yaml()))
    squash = {n.replace("_", "").lower() for n in kinds}
    assert squash == {n.lower() for n in k.StoredKind.__members__}


@pytest.mark.parametrize("path", EXAMPLES, ids=[q.name for q in EXAMPLES])
def test_shipped_example_round_trips(path):
    lc = p.load_yaml_file(str(path))
    c = lc.builder.build()
    _same_run(c, _round(c), t_end=lc.t_start + STEPS * lc.dt, dt=lc.dt,
              engine="pwl")


def _v1(path):
    sys.path[:0] = [str(ROOT / "benchmarks" / "tools")]
    try:
        import v1_reader
    finally:
        del sys.path[0]
    return v1_reader.load(path)


@pytest.mark.parametrize("path", V1_CASES,
                         ids=[str(q.relative_to(ROOT / "benchmarks" / "circuits"))
                              for q in V1_CASES])
def test_harness_circuit_round_trips(path):
    try:
        case = _v1(path)
        built = case.build()
    except Exception:  # noqa: BLE001 — not a v1 scenario (a manifest, a v2 file)
        return
    b, sw = built[0], built[1]
    c = b.build()
    kw = {"switch_fn": sw} if sw is not None else {}
    _same_run(c, _round(c), t_end=STEPS * case.dt, dt=case.dt, engine="pwl", **kw)


def test_the_text_is_what_the_schema_says():
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "in", "gnd", 10.0)
    b.add_resistor("R1", "in", "out", 1e3)
    b.add_capacitor("C1", "out", "gnd", 1e-6, c0=2.0)
    b.add_diode("D1", "out", "gnd", form="branch", R_on=0.0, g_off=0.0)
    b.param("Rload", 1e3, "ohm")
    b.add_resistor("R2", "out", "gnd", b.parameters()[0])
    y = b.build().to_yaml()
    # stored members (G, not R), defaults omitted, ground by name
    assert '{kind: resistor, name: "R1", pins: ["in", "out"], G: 0.001}' in y
    assert "g_off" not in y.split("D1")[1].split("\n")[0]
    assert '{param: "Rload", field: "R2.R"}' in y
    assert '{device: "C1", value: 2}' in y


def test_hashes_are_pinned():
    # FNV-1a over names and IEEE bit patterns: the same on every platform
    # and after a round trip.
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "in", "gnd", 10.0)
    b.add_resistor("R1", "in", "out", 1e3)
    b.add_capacitor("C1", "out", "gnd", 1e-6)
    c = b.build()
    pins = (c.structure_hash(), c.value_hash())
    assert pins == (STRUCTURE_PIN, VALUE_PIN), [hex(x) for x in pins]
    assert (lambda c2: (c2.structure_hash(), c2.value_hash()))(_round(c)) == pins


STRUCTURE_PIN = 0x57BF94FE7FA50967
VALUE_PIN = 0x746D6010F3DC01E0


def test_tables_hash_by_content():
    def make():
        b = p.CircuitBuilder()
        b.add_current_source("I", "gnd", "a", 1.0)
        b.add_saturable_inductor_table("L", "a", "gnd", [0.0, 1.0, 2.0],
                                       [0.0, 1e-3, 1.5e-3])
        return b.build()
    assert make().value_hash() == make().value_hash()


@pytest.mark.parametrize(("text", "match"), [
    ('pulsim_circuit: 1\nnodes: ["a"]\ndevices:\n  - {kind: bogus, name: "X"}\n',
     r"from_yaml: bogus: unknown kind"),
    ('pulsim_circuit: 1\nnodes: ["a"]\ndevices:\n'
     '  - {kind: resistor, name: "R1", pins: ["a", "gnd"], R: 5}\n',
     r'from_yaml: R1: unknown key "R"'),
    ('pulsim_circuit: 1\nnodes: ["a"]\ndevices:\n'
     '  - {kind: resistor, name: "R1", pins: ["a", "b"], G: 1}\n',
     r'from_yaml: R1: pin "b" is not in nodes'),
    ('pulsim_circuit: 1\nnodes: ["a"]\ndevices:\n'
     '  - {kind: resistor, name: "R1", pins: ["a", "gnd"], G: one}\n',
     r'from_yaml: R1: "G" is not a number: "one"'),
    ('pulsim_circuit: 2\n', r"pulsim_circuit must be 1"),
    ('pulsim_circuit: 1\nwires: []\n', r'unknown key "wires"'),
])
def test_a_malformed_document_names_the_device_and_the_key(text, match):
    with pytest.raises(ValueError, match=match):
        p.Circuit.from_yaml(text)


def test_controls_round_trip_and_a_c_block_is_refused():
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "in", "gnd", 1.0)
    b.add_resistor("R1", "in", "gnd", 1.0)
    c = b.build().with_controls([("drive", "graph_driven", 0xABCDEF0123456789, None)])
    assert _round(c).controls() == c.controls()
    cb = c.with_controls([("pi", "c_block", 1, object())])
    with pytest.raises(ValueError, match=r'control "pi" \(c_block\) carries a Python callable'):
        cb.to_yaml()
    with pytest.raises(ValueError, match="Python callable"):
        pickle.dumps(cb)


def test_pickle_and_clone():
    build = CELLS["Inductor"][0]
    c = build(True).build()
    c2 = pickle.loads(pickle.dumps(c))
    assert (c2.value_hash(), c2.structure_hash()) == (c.value_hash(), c.structure_hash())
    assert c2.components() == c.components()
    assert c.clone().value_hash() == c.value_hash()


def test_power_mosfet_record_survives():
    build = VARIANT_CELLS["power_mosfet"][0][0]
    c = build(True).build()
    assert _round(c).components() == c.components()


def _rc_param():
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "in", "gnd", 1.0)
    b.param("R", 1e3, "ohm")
    b.add_resistor("R1", "in", "out", b.parameters()[0])
    b.add_capacitor("C1", "out", "gnd", 1e-6)
    return b.build()


def _kpi_final(res, _params):
    return {"v": float(res.states[-1][-1])}


def test_a_circuit_sweeps_in_worker_processes():
    kw = dict(params={"R": [500.0, 1e3, 2e3]}, kpi_fn=_kpi_final,
              t_end=1e-4, dt=1e-6, engine="pwl", verbose=False)
    ref = p.sweep(_rc_param(), parallel=False, **kw)
    got = p.sweep(_rc_param(), parallel=True, backend="process", n_workers=2, **kw)
    assert got.backend == "process" and got.failed == []
    np.testing.assert_array_equal(got.kpis["v"], ref.kpis["v"])
