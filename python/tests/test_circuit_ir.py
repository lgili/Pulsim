"""add-immutable-circuit-ir: the immutable ``pulsim.Circuit``.

Group 1: ``builder.build()`` is a snapshot whose lowered form is the
builder's own graph and pool, in insertion order — checked on every
shipped example and every harness circuit (``components()``, the unknown
names, the switch order, ``initial_state()``), and ``simulate(circuit)``
equals ``simulate(builder)`` bit for bit on every engine.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import pulsim as p

ROOT = Path(__file__).resolve().parents[2]
PROBES = ROOT / "openspec" / "changes" / "add-immutable-circuit-ir" / "probes"
sys.path[:0] = [str(PROBES), str(ROOT / "benchmarks"), str(ROOT / "benchmarks" / "tools")]

import sweep_probe  # noqa: E402

pytest.importorskip("yaml")
CIRCUITS = sweep_probe.circuits()
RECORD = json.loads((PROBES / "components_baseline.json").read_text())


def test_runs_this_worktree():
    assert Path(p.__file__).resolve().is_relative_to(ROOT / "python")


def test_every_circuit_is_recorded():
    # Task 0.2's record: a circuit added to examples/ or benchmarks/ must
    # be added to the record (re-run `sweep_probe.py components`).
    assert {cid for cid, _ in CIRCUITS} == set(RECORD)
    assert sum(not isinstance(b, Exception) for _, b in CIRCUITS) == 136


def _switch_names(b):
    out = []
    for c in b.components():
        if c["name"]:
            try:
                out.append((c["name"], b.switch_index_of(c["name"])))
            except IndexError:
                pass
    return out


@pytest.mark.parametrize("cid,b", CIRCUITS, ids=[cid for cid, _ in CIRCUITS])
def test_lowered_equals_builder(cid, b):
    if isinstance(b, Exception):
        assert RECORD[cid] == f"unloadable: {type(b).__name__}"
        return
    c = b.build()
    # The record of task 0.2 is today's components(); the snapshot too.
    assert {"n": len(c.components()), "sha": sweep_probe.digest(c)} == RECORD[cid]
    assert c.components() == b.components()
    assert c.state_var_names() == b.state_var_names()
    assert c.nodes == [n["name"] for n in b.graph.nodes]
    assert c.num_switches == b.graph.num_switches
    assert _switch_names(c) == _switch_names(b)
    np.testing.assert_array_equal(c.initial_state(), b.initial_state())
    assert [d["name"] for d in c.devices()] == [x["name"] for x in b.components()]


def _rc_buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_mosfet("Q1", "vin", "sw")
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6, 0.5)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 10.0)
    fn = p.make_pwm_switch_fn(100e3, 0.5, b.switch_index_of("Q1"), b.graph.num_switches)
    return b, fn


@pytest.mark.parametrize("engine,kw", [
    ("pwl", dict(dt=1e-7)), ("trbdf2", {}), ("dsed", {})])
def test_simulate_circuit_equals_builder(engine, kw):
    b, fn = _rc_buck()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_b = p.simulate(b, 3e-4, engine=engine, switch_fn=fn, **kw)
        b2, _ = _rc_buck()
        r_c = p.simulate(b2.build(), 3e-4, engine=engine, switch_fn=fn, **kw)
    np.testing.assert_array_equal(r_c.times, r_b.times)
    np.testing.assert_array_equal(r_c.states, r_b.states)
    # The builder's i0= travels with the circuit.
    assert r_c.states[0, b.state_var_names().index("I(L)")] == 0.5


def test_circuit_is_a_snapshot():
    b, _ = _rc_buck()
    c = b.build()
    h, s = c.value_hash(), c.structure_hash()
    b.add_resistor("Rx", "vout", "gnd", 1.0)
    b.set_value("RL.R", 20.0)
    assert (c.value_hash(), c.structure_hash()) == (h, s)
    assert c.get_value("RL.R") == 10.0
    copy = c.to_builder()
    copy.set_value("RL.R", 30.0)
    assert c.get_value("RL.R") == 10.0
    assert copy.build().structure_hash() == s
    assert not hasattr(c, "__dict__")  # no Python-side state can be hung on it
