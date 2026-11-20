"""add-immutable-circuit-ir group 8: scopes.

A subsystem instance is a scope of the circuit: the tree is data
(``Circuit.scopes()``), parameters declared in a body belong to their
instance and expressions look names up from it outward, and the
flattened names are the old prefix names, so a scoped circuit is its
hand-prefixed twin element for element and simulates bit for bit like
it on every engine.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest

import pulsim as p

ROOT = Path(__file__).resolve().parents[2]
N_CELLS = 100            # 3 branches per cell + source + load = 302 branches
G_HALF = 500.0           # the arm's own value; the root's is a decoy
C_CELL = 2e-3


class _Mask:
    """A fixed mask (Sb of every third cell open, every Si closed) with
    no edges: the same switching on every engine."""

    def __init__(self, n):
        self.m = p.SwitchStateMask(n)
        for i in range(n):
            cell, which = divmod(i, 2)
            self.m.set(i, which == 1 or cell % 3 != 0)

    def __call__(self, _t):
        return self.m

    def next_edge_after(self, _t):
        return math.inf


def _scoped():
    cell = p.define_subsystem("HalfBridgeSM", ports=("top", "bot"),
                              params={"C": C_CELL})

    @cell.body
    def _(s):
        gon = s.param("Gon", "2*g_half")      # the arm's g_half, not the root's
        cv = s.param("Cv", s.p.C)
        s.add_switch("Sb", "top", "bot", gon, 1e-9)
        s.add_switch("Si", "top", "m", gon, 1e-9)
        s.add_capacitor("C", "m", "bot", cv)

    arm = p.define_subsystem("Arm", ports=("hi", "lo"))

    @arm.body
    def _(s):
        s.param("g_half", G_HALF)               # shadows the root's g_half
        prev = "hi"
        for i in range(N_CELLS):
            nxt = "lo" if i == N_CELLS - 1 else f"n{i}"
            cell.instantiate(s, f"sm{i}", top=prev, bot=nxt)
            prev = nxt

    b = p.CircuitBuilder()
    b.param("g_half", 1.0)                      # decoy: must not be read
    b.param("Gon", 99.0)                        # decoy: shadowed in every cell
    b.add_voltage_source("V", "dc_p", "gnd", 100.0)
    arm.instantiate(b, "leg_a", hi="dc_p", lo="out")
    b.add_resistor("RL", "out", "gnd", 10.0)
    return b


def _flat(g_on=2 * G_HALF, c_of=lambda i: C_CELL):
    """The same circuit written by hand, the way it was before scopes."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "dc_p", "gnd", 100.0)
    prev = "dc_p"
    for i in range(N_CELLS):
        nxt = "out" if i == N_CELLS - 1 else f"leg_a/n{i}"
        pre = f"leg_a/sm{i}"
        b.add_switch(f"{pre}/Sb", prev, nxt, g_on, 1e-9)
        b.add_switch(f"{pre}/Si", prev, f"{pre}/m", g_on, 1e-9)
        b.add_capacitor(f"{pre}/C", f"{pre}/m", nxt, c_of(i))
        prev = nxt
    b.add_resistor("RL", "out", "gnd", 10.0)
    return b


def _run(c, engine):
    kw = {"engine": engine, "switch_fn": _Mask(c.num_switches),
          "t_end": 2e-5}
    if engine == "pwl":
        kw["dt"] = 1e-7
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return p.simulate(c, **kw)
        except Exception as e:  # noqa: BLE001 — compared, not swallowed
            return (type(e).__name__, str(e))


def _same_run(a_c, b_c, engine):
    a, b = _run(a_c, engine), _run(b_c, engine)
    if isinstance(a, tuple) or isinstance(b, tuple):
        assert a == b
        return a
    np.testing.assert_array_equal(np.asarray(a.times), np.asarray(b.times))
    np.testing.assert_array_equal(np.asarray(a.states), np.asarray(b.states))
    return a


def test_the_scope_tree_is_data():
    c = _scoped().build()
    tree = c.scopes()
    assert tree[0] == {"path": "leg_a", "name": "leg_a", "kind": "Arm",
                       "parent": ""}
    assert tree[1] == {"path": "leg_a/sm0", "name": "sm0",
                       "kind": "HalfBridgeSM", "parent": "leg_a"}
    assert len(tree) == 1 + N_CELLS
    # membership follows the flattened name
    assert c.scope_of("leg_a/sm17/C") == "leg_a/sm17"
    assert c.scope_of("leg_a/sm17/m") == "leg_a/sm17"
    assert c.scope_of("leg_a/n3") == "leg_a"
    assert c.scope_of("RL") == ""
    # parameters belong to their instance
    names = [q.name for q in c.parameters()]
    assert "leg_a/g_half" in names and "leg_a/sm42/Gon" in names


@pytest.mark.parametrize("engine", ("pwl", "trbdf2", "dsed"))
def test_a_scoped_circuit_is_its_flattened_twin(engine):
    s, f = _scoped().build(), _flat().build()
    assert len(s.devices()) == 2 + 3 * N_CELLS == 302
    assert s.components() == f.components()
    assert s.nodes == f.nodes
    # the tree and the parameters are structure the twin does not have
    assert s.structure_hash() != f.structure_hash()
    r = _same_run(s, f, engine)
    assert not isinstance(r, tuple), r   # every engine runs it: the claim is bit for bit
    assert r.v("leg_a/sm3/m") is not None


def test_shadowing_and_outer_lookup():
    c = _scoped().build()
    # every cell's Gon = 2 * the ARM's g_half, never the root decoys
    assert c.get_value("leg_a/sm5/Sb.g_on") == 2 * G_HALF
    vals = {q.name: q.value for q in c.parameters()}
    assert vals["Gon"] == 99.0 and vals["g_half"] == 1.0
    # an inner name that reads itself is a cycle, as in YAML, not the outer one
    b = p.CircuitBuilder()
    b.param("Ron", 1.0)
    with b.scope("x"), pytest.raises(ValueError, match="depends on itself"):
        b.param("Ron", "2*Ron")
    # bind_param / set_param read names as an expression does: the scope's own first
    with b.scope("y"):
        b.param("Ron", 3.0)
        b.add_resistor("y/R", "a", "gnd", 1.0)
        b.bind_param("y/R.G", "Ron")
        assert b.build().get_value("y/R.G") == 3.0
        b.set_param("Ron", 4.0)
    assert b.build().get_value("y/R.G") == 4.0
    assert {q.name: q.value for q in b.build().parameters()}["Ron"] == 1.0


def test_with_params_addresses_one_instance_or_the_arm():
    c = _scoped().build()
    one = c.with_params({"leg_a/sm3/Cv": 5e-3})
    assert one.get_value("leg_a/sm3/C.C") == 5e-3
    assert one.get_value("leg_a/sm4/C.C") == C_CELL
    assert one.structure_hash() == c.structure_hash()
    flat = _flat(c_of=lambda i: 5e-3 if i == 3 else C_CELL).build()
    assert one.components() == flat.components()
    _same_run(one, flat, "pwl")
    # the arm's parameter re-values every cell's expression
    arm = c.with_params({"leg_a/g_half": 250.0})
    _same_run(arm, _flat(g_on=500.0).build(), "pwl")
    # the root decoy drives nothing
    assert c.with_params({"g_half": 7.0}).components() == c.components()
    # with_value addresses a scoped device by its flattened name
    assert c.with_value("leg_a/sm3/C.C", 5e-3).components() == flat.components()


def test_the_tree_survives_yaml_and_pickle():
    import pickle
    c = _scoped().build()
    y = c.to_yaml()
    c2 = p.Circuit.from_yaml(y)
    assert c2.scopes() == c.scopes()
    assert c2.value_hash() == c.value_hash()
    assert c2.structure_hash() == c.structure_hash()
    assert c2.to_yaml() == y
    assert [q.name for q in c2.parameters()] == [q.name for q in c.parameters()]
    # the round-tripped expressions still resolve in their scope
    assert c2.with_params({"leg_a/g_half": 250.0}).get_value(
        "leg_a/sm9/Si.g_on") == 500.0
    assert pickle.loads(pickle.dumps(c)).scopes() == c.scopes()


def test_a_circuit_without_scopes_has_no_tree():
    # its hashes are pinned in test_circuit_yaml.test_hashes_are_pinned
    b = p.CircuitBuilder()
    b.add_resistor("R", "a", "gnd", 1.0)
    assert b.build().scopes() == []


def test_builder_scope_context_manager_and_refusals():
    b = p.CircuitBuilder()
    with b.scope("x/y", kind="K") as path:
        assert path == "x/y" and b.current_scope == "x/y"
        q = b.param("a", 2.0)
        assert q.name == "x/y/a"
        assert b.scoped_name("R") == "x/y/R"
        with pytest.raises(ValueError, match="no '/'"):
            b.param("p/q", 1.0)
        with pytest.raises(ValueError, match="already exists in scope"):
            b.param("a", 3.0)
    assert b.current_scope == ""
    assert [s["path"] for s in b.scopes()] == ["x", "x/y"]
    assert b.scopes()[0]["kind"] == ""
    with pytest.raises(ValueError, match="already used by an instance"):
        b.open_scope("x/y", "K2")
    with pytest.raises(ValueError, match="non-empty names"):
        b.open_scope("a//b")
    with pytest.raises(Exception, match="no scope is open"):
        b.close_scope()
    # an expression cannot see into a scope: names resolve outward only
    with pytest.raises(ValueError, match='unknown name "a"'):
        b.param("z", "2*a")


def test_nested_instantiate_after_the_body_returned_is_refused():
    kept = {}
    inner = p.define_subsystem("In", ports=("a",))

    @inner.body
    def _(s):
        s.add_resistor("R", "a", "gnd", 1.0)

    outer = p.define_subsystem("Out", ports=("a",))

    @outer.body
    def _(s):
        kept["s"] = s
        s.add_resistor("R", "a", "gnd", 1.0)

    b = p.CircuitBuilder()
    outer.instantiate(b, "o", a="x")
    with pytest.raises(ValueError, match="has returned"):
        inner.instantiate(kept["s"], "i", a="a")



def test_instantiate_under_a_hand_opened_scope_composes():
    cell = p.define_subsystem("Cell", ports=("a",))

    @cell.body
    def _(s):
        s.add_resistor("R", "a", "gnd", 1.0)

    b = p.CircuitBuilder()
    b.add_voltage_source("V", "x", "gnd", 1.0)
    with b.scope("leg_a", kind="Arm"):
        inst = cell.instantiate(b, "sm0", a="x")
    assert inst.path == "leg_a/sm0"
    c = b.build()
    assert [x["path"] for x in c.scopes()] == ["leg_a", "leg_a/sm0"]
    assert [d["name"] for d in c.devices()] == ["V", "leg_a/sm0/R"]


# ---------------------------------------------------------------- YAML

_ARM_YAML = """
circuit:
  parameters:
    - {name: g_half, value: 1.0}
  devices:
    - {type: voltage_source, name: V, from: dc_p, to: gnd, V: 100.0}
    - {type: resistor, name: RL, from: out, to: gnd, R: 10.0}
  subsystems:
    - name: leg_a
      kind: Arm
      ports: {hi: dc_p, lo: out}
      parameters:
        - {name: Gon, value: "2*g_half"}
        - {name: g_half, value: 500.0}
      devices:
        - {type: switch, name: Sb, from: hi, to: mid, g_on: Gon, g_off: 1.0e-9}
      subsystems:
        - name: sm0
          kind: Cell
          ports: {top: mid, bot: lo}
          parameters: [{name: Cv, value: 2.0e-3}]
          devices:
            - {type: switch, name: Si, from: top, to: m, g_on: "Gon", g_off: 1.0e-9}
            - {type: capacitor, name: C, from: m, to: bot, C: "Cv"}
"""


def test_yaml_subsystems_are_scopes():
    c = p.load_yaml_string(_ARM_YAML).builder.build()
    assert [x["path"] for x in c.scopes()] == ["leg_a", "leg_a/sm0"]
    assert [x["kind"] for x in c.scopes()] == ["Arm", "Cell"]
    names = [d["name"] for d in c.devices()]
    assert names == ["V", "RL", "leg_a/Sb", "leg_a/sm0/Si", "leg_a/sm0/C"]
    assert c.nodes == ["dc_p", "out", "leg_a/mid", "leg_a/sm0/m"]
    # the block's own g_half (listed after the expression) is the one read
    assert c.get_value("leg_a/sm0/Si.g_on") == 1000.0
    # a field expression is a derived parameter of the device's scope
    assert "leg_a/sm0/Si:g_on" in [q.name for q in c.parameters()]
    c2 = p.Circuit.from_yaml(c.to_yaml())
    assert c2.scopes() == c.scopes() and c2.value_hash() == c.value_hash()
    assert c.with_params({"leg_a/g_half": 2.0}).get_value(
        "leg_a/sm0/Si.g_on") == 4.0


@pytest.mark.parametrize("text, match", [
    ("- {name: s, parameters: [{name: a, value: 'b'}, {name: b, value: 'a'}]}",
     "depends on itself"),
    ("- {name: s, devices: [{type: resistor, name: R, from: a, to: gnd, R: 'nope'}]}",
     'unknown name "nope"'),
    ("- {name: s}\n    - {name: s}", "listed twice"),
    ("- {name: a/b}", "one non-empty path segment"),
])
def test_yaml_subsystem_errors_name_the_problem(text, match):
    doc = ("circuit:\n  devices:\n    - {type: voltage_source, name: V, "
           "from: a, to: gnd, V: 1.0}\n  subsystems:\n    " + text + "\n")
    with pytest.raises(Exception, match=match):
        p.load_yaml_string(doc)


def _wrapped(text: str) -> str:
    """`text`'s circuit moved into one subsystem block named X, no ports:
    every name the block writes is scoped, ground excepted."""
    import re

    import yaml

    class _Loader(yaml.SafeLoader):   # YAML 1.2 floats, as yaml-cpp reads them
        pass
    _Loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(r"^[-+]?(?:[0-9][0-9_]*\.[0-9_]*(?:[eE][-+]?[0-9]+)?"
                   r"|[0-9][0-9_]*[eE][-+]?[0-9]+|\.[0-9_]+(?:[eE][-+]?[0-9]+)?)$"),
        list("-+0123456789."))
    doc = yaml.load(text, Loader=_Loader)  # noqa: S506 — a SafeLoader subclass
    circ = doc["circuit"]
    blk = {"name": "X"}
    for k in ("parameters", "nodes", "devices"):
        if k in circ:
            blk[k] = circ.pop(k)
    circ["devices"] = []
    circ["subsystems"] = [blk]
    return yaml.safe_dump(doc)


_ZOO = """
circuit:
  devices:
    - {type: voltage_source, name: V, from: a, to: gnd, V: 1.0}
    - {type: current_source, name: I, from: a, to: b, I: 1.0}
    - {type: pwm_voltage_source, name: Vp, from: c, to: gnd, v_high: 1, v_low: 0, frequency: 1.0e3, duty: 0.5}
    - {type: sine_voltage_source, name: Vs, from: d, to: gnd, v_amplitude: 1, frequency: 50}
    - {type: pulse_voltage_source, name: Vq, from: e, to: gnd, v_pulsed: 1, pulse_width: 1.0e-3}
    - {type: mosfet_level1, name: M1, drain: a, source: b, gate: c, K: 1.0, V_T: 1.0}
    - {type: vcvs, name: E1, in_pos: a, in_neg: b, out_pos: f, out_neg: gnd, gain: 2}
    - {type: vccs, name: G1, in_pos: a, in_neg: b, from: f, to: gnd, gm: 2}
    - {type: resistor, name: R1, from: f, to: g, R: 1.0}
    - {type: ccvs, name: H1, control: V, out_pos: h, out_neg: gnd, rm: 2}
    - {type: cccs, name: F1, control: V, from: h, to: gnd, gain: 2}
    - {type: gyrator, name: Y1, p1_pos: a, p1_neg: gnd, p2_pos: i, p2_neg: gnd, g: 1}
    - {type: sine_current_source, name: Is, from: i, to: gnd, i_dc: 0, i_amplitude: 1, frequency: 50}
    - {type: pulse_current_source, name: Ip, from: i, to: gnd, i_initial: 0, i_pulsed: 1, t_start: 0, pulse_width: 1.0e-3}
    - {type: op_amp_ideal, name: U1, in_pos: a, in_neg: j, out: j}
    - {type: saturable_inductor, name: Ls, from: g, to: k, L_0: 1.0e-3, I_sat: 1.0}
    - {type: igbt_level1, name: Q1, collector: k, emitter: gnd, gate: c}
    - {type: capacitor, name: C1, from: k, to: gnd, C: 1.0e-6}
    - {type: inductor, name: L1, from: k, to: l, L: 1.0e-3}
    - {type: dc_machine, name: DM, L_a: 1.0e-3, K: 0.1, J: 1.0e-3, a_pos: l, a_neg: gnd, omega_node: w}
    - {type: diode, name: D1, anode: l, cathode: m, g_on: 1.0e3, g_off: 1.0e-9}
    - {type: nonlinear_diode, name: D2, anode: m, cathode: gnd, allow_non_passive: true}
    - {type: switch, name: S1, from: m, to: n, g_on: 1.0e3, g_off: 1.0e-9}
    - {type: mosfet, name: M2, drain: n, source: gnd}
    - {type: mosfet_with_body_diode, name: M3, drain: n, source: o}
    - {type: igbt, name: Q2, collector: o, emitter: gnd}
    - {type: transformer, name: T1, p_from: o, p_to: gnd, s_from: p, s_to: gnd, L_p: 1.0e-3, L_s: 1.0e-3}
    - {type: gapped_core_inductor, name: Lg, from: p, to: q, N: 10, Ae: 1.0e-4, le: 0.1, lg: 1.0e-3}
    - {type: multi_winding_transformer, name: T2, windings: [{from: q, to: gnd, L: 1.0e-3}, {from: r, to: gnd, L: 1.0e-3}], k: [[1, 0.9], [0.9, 1]]}
    - {type: saturable_transformer, name: T3, N_p: 10, Ae: 1.0e-4, le: 0.1, lg: 1.0e-3, p_from: r, p_to: gnd, s_from: s, s_to: gnd, N_s: 10}
    - {type: ideal_transformer, name: T4, p_from: s, p_to: gnd, s_from: t, s_to: gnd, n: 2}
    - {type: transmission_line, name: TL, a_pos: t, a_neg: gnd, b_pos: tb, b_neg: gnd, Z0: 50, delay: 1.0e-6}
    - {type: induction_motor, name: IM, R_s: 1, L_s: 1.0e-2, R_r: 1, L_r: 1.0e-2, L_m: 9.0e-3, phase_nodes: [u, v, x], neutral_node: nn}
    - {type: hysteretic_inductor, name: Lh, from: t, to: gnd, N_turns: 10, l_m: 0.1, A_core: 1.0e-4, Ms: 1.6e6, a: 1000, alpha: 1.0e-3, c: 0.2, k: 2000}
    - {type: power_mosfet, name: PM, drain: u, source: gnd, gate: c, K: 1.0, V_T: 3.0, C_gs: 1.0e-9}
"""


def _compare_wrapped(text, run=None):
    flat = p.load_yaml_string(text)
    scoped = p.load_yaml_string(_wrapped(text))
    a, s = flat.builder.build(), scoped.builder.build()
    scope = lambda n: n if n in ("gnd", "GND", "0") else f"X/{n}"  # noqa: E731
    # every node and every device renamed, nothing else: a node key the
    # scoping missed would appear here unprefixed
    assert s.nodes == [scope(n) for n in a.nodes]
    da, ds = a.devices(), s.devices()
    assert [d["name"] for d in ds] == [scope(d["name"]) if d["name"] else ""
                                       for d in da]
    for x, y in zip(da, ds):
        assert (x["kind"], x["pins"], x["fields"]) == (y["kind"], y["pins"], y["fields"])
    assert [q.value for q in a.parameters()] == [q.value for q in s.parameters()]
    if run is not None:
        ra, rs = (p.simulate(c, **run) for c in (a, s))
        np.testing.assert_array_equal(np.asarray(ra.times), np.asarray(rs.times))
        np.testing.assert_array_equal(np.asarray(ra.states), np.asarray(rs.states))


def test_every_authoring_type_scopes_every_node():
    _compare_wrapped(_ZOO)


EXAMPLES = [q for q in sorted((ROOT / "examples").glob("*.yaml"))
            if q.name != "refrigerator_compressor_cc.yaml"]   # v1, refused


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda q: q.stem)
def test_a_shipped_example_in_a_subsystem_is_the_example(path):
    lc = p.load_yaml_file(str(path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _compare_wrapped(path.read_text(encoding="utf-8"),
                         run={"t_end": lc.t_start + 50 * lc.dt, "dt": lc.dt,
                              "engine": "pwl"})
