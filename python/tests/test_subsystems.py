"""Subsystems: define once, instantiate many, scoped names (A.7).

The audit's complaint, measured before the feature existed: a
100-submodule arm is 302 branches whose every name is an f-string
the author keeps unique by discipline — and a duplicate is
accepted SILENTLY, `branch_id_of` returning the first, so the
second device is unreachable by name for the rest of the run.
Nothing says "inside SM17".

The feature is entirely builder-level: instantiation FLATTENS, so
the kernel never learns about hierarchy — the scoped string simply
IS the device's name, which is why results, diagnostics and traces
all speak the path without being told about it.
"""

import numpy as np
import pytest

import pulsim as p


def _sm():
    sm = p.define_subsystem("HalfBridgeSM", ports=("top", "bot"),
                             params={"C": 2e-3, "r_on": 1e-3})

    @sm.body
    def _(s):
        s.add_switch("Sb", "top", "bot", 1 / s.p.r_on, 1e-9)
        s.add_switch("Si", "top", "m", 1 / s.p.r_on, 1e-9)
        s.add_capacitor("C", "m", "bot", s.p.C)

    return sm


def test_names_are_scoped_and_ports_bind_to_caller_nets():
    sm = _sm()
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc_p", "gnd", 1000.0)
    i0 = sm.instantiate(b, "leg_a/sm0", top="dc_p", bot="n0")
    i1 = sm.instantiate(b, "leg_a/sm1", top="n0", bot="gnd")

    assert i0.devices == ["leg_a/sm0/Sb", "leg_a/sm0/Si",
                           "leg_a/sm0/C"]
    assert i1.name_of("C") == "leg_a/sm1/C"
    # Ports BIND — no node is created for them.
    assert i0.ports == {"top": "dc_p", "bot": "n0"}
    assert b.node_id_of("dc_p") == b.node_id_of("dc_p")
    # Internal nets are scoped, so two instances do not collide.
    assert b.node_id_of("leg_a/sm0/m") != b.node_id_of(
        "leg_a/sm1/m")
    # Ground passes through untouched.
    assert i1.ports["bot"] == "gnd"


def test_parameters_default_and_override():
    sm = _sm()
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 10.0)
    d = sm.instantiate(b, "d", top="a", bot="gnd")
    o = sm.instantiate(b, "o", top="a", bot="gnd", C=5e-3)
    assert d.params["C"] == 2e-3
    assert o.params["C"] == 5e-3


def test_nested_instantiation_composes_paths():
    sm = _sm()
    leg = p.define_subsystem("Leg", ports=("hi", "lo"),
                              params={"n": 3})

    @leg.body
    def _(s):
        prev = "hi"
        n = int(s.p.n)
        for i in range(n):
            node = "lo" if i == n - 1 else f"x{i}"
            sm.instantiate(s, f"sm{i}", top=prev, bot=node)
            prev = node

    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc_p", "gnd", 600.0)
    b.add_resistor("Rl", "dc_p", "mid", 50.0)
    leg.instantiate(b, "leg_a", hi="mid", lo="gnd", n=3)

    assert b.branch_id_of("leg_a/sm2/C") >= 0
    # The LEG's own internal net is scoped to the leg, and the
    # SM's port binding resolved through it.
    assert b.node_id_of("leg_a/x0") >= 0
    assert b.node_id_of("leg_a/sm1/m") >= 0
    # Two legs of the same definition do not collide.
    b.add_resistor("Rl2", "dc_p", "mid2", 50.0)
    leg.instantiate(b, "leg_b", hi="mid2", lo="gnd", n=3)
    assert b.branch_id_of("leg_b/sm2/C") != b.branch_id_of(
        "leg_a/sm2/C")


def test_results_and_diagnostics_speak_the_path():
    """Nothing downstream was taught about hierarchy — the scoped
    string IS the name, so it just works."""
    import warnings

    sm = _sm()
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc_p", "gnd", 600.0)
    b.add_resistor("Rl", "dc_p", "mid", 50.0)
    sm.instantiate(b, "leg_a/sm0", top="mid", bot="gnd")

    n = b.graph.num_switches

    def sf(t):
        m = p.SwitchStateMask(n)
        m.set(b.switch_index_of("leg_a/sm0/Si"), True)
        return m

    res = p.simulate(b, t_end=1e-4, dt=1e-7, switch_fn=sf)
    assert np.isfinite(res.v("leg_a/sm0/m")).all()
    assert np.isfinite(res.i("leg_a/sm0/C")).all()

    # A named diagnostic reports the PATH, which is the audit's
    # "diagnóstico não diz 'dentro do SM17'" complaint.
    b2 = p.CircuitBuilder()
    b2.add_voltage_source("V", "in", "gnd", 48.0)
    opener = p.define_subsystem("Opener", ports=("a", "b"),
                                 params={})

    @opener.body
    def _(s):
        s.add_inductor("L", "a", "mid", 1e-3)
        s.add_switch("S", "mid", "b", 1e3, 1e-9)

    opener.instantiate(b2, "block7", a="in", b="gnd")
    n2 = b2.graph.num_switches

    def sf2(t):
        m = p.SwitchStateMask(n2)
        m.set(0, (t % 1e-4) < 5e-5)
        return m

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p.simulate(b2, t_end=3e-4, dt=1e-7, switch_fn=sf2)
    said = [str(x.message) for x in w if "block7/mid" in
            str(x.message)]
    assert said, "the diagnostic did not name the scoped node"


def test_hundred_cells_is_a_loop_not_a_naming_scheme():
    sm = _sm()
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc_p", "gnd", 10e3)
    prev = "dc_p"
    for i in range(100):
        node = "gnd" if i == 99 else f"n{i}"
        sm.instantiate(b, f"arm_u/sm{i}", top=prev, bot=node)
        prev = node
    assert b.graph.num_switches == 200
    assert len(sm.instances) == 100
    assert b.branch_id_of("arm_u/sm73/C") >= 0


def test_duplicate_device_names_are_refused():
    """It used to be silent: `branch_id_of` answered with the
    FIRST, and the second device was unreachable by name forever.
    Instancing makes that a hundred devices per typo."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 5.0)
    b.add_resistor("R1", "in", "n1", 1e3)
    with pytest.raises(Exception, match="already used"):
        b.add_resistor("R1", "n1", "gnd", 2e3)

    # And the same path twice is the instancing form of it.
    sm = _sm()
    b2 = p.CircuitBuilder()
    b2.add_voltage_source("V", "a", "gnd", 5.0)
    sm.instantiate(b2, "dup", top="a", bot="gnd")
    with pytest.raises(Exception, match="already used"):
        sm.instantiate(b2, "dup", top="a", bot="gnd")


def test_declaration_errors_name_what_is_wrong():
    with pytest.raises(ValueError, match="at least one port"):
        p.define_subsystem("NoPorts", ports=())
    with pytest.raises(ValueError, match="duplicate port"):
        p.define_subsystem("Dup", ports=("a", "a"))
    with pytest.raises(ValueError, match="separates path"):
        p.define_subsystem("Slash", ports=("a/b",))

    sm = p.define_subsystem("NoBody", ports=("a",))
    b = p.CircuitBuilder()
    with pytest.raises(ValueError, match="no body"):
        sm.instantiate(b, "x", a="gnd")


def test_instantiation_errors_name_what_is_missing():
    sm = _sm()
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 5.0)
    with pytest.raises(ValueError, match="not connected"):
        sm.instantiate(b, "x", top="a")
    with pytest.raises(ValueError, match="neither a port nor"):
        sm.instantiate(b, "x", top="a", bot="gnd", Q=1.0)
    with pytest.raises(ValueError, match="path must be"):
        sm.instantiate(b, "", top="a", bot="gnd")


def test_undeclared_parameter_access_is_loud():
    sm = p.define_subsystem("P", ports=("a",), params={"x": 1.0})

    @sm.body
    def _(s):
        s.add_resistor("R", "a", "gnd", s.p.nope)

    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 5.0)
    with pytest.raises(AttributeError, match="not declared"):
        sm.instantiate(b, "i", a="a")


def test_scoped_builder_exposes_net_resolution():
    """A body sometimes needs a net name for something that is not
    an `add_*` — a closure, a probe, a nested instantiate."""
    seen = {}
    sm = p.define_subsystem("N", ports=("a",), params={})

    @sm.body
    def _(s):
        seen["port"] = s.net("a")
        seen["internal"] = s.net("mid")
        seen["gnd"] = s.net("gnd")
        seen["path"] = s.path
        s.add_resistor("R", "a", "mid", 1.0)
        s.add_resistor("R2", "mid", "gnd", 1.0)

    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 5.0)
    sm.instantiate(b, "u1", a="vin")
    assert seen == {"port": "vin", "internal": "u1/mid",
                     "gnd": "gnd", "path": "u1"}


def test_ground_is_whatever_the_kernel_says_it_is():
    """The first version of this module hard-coded a ground-alias
    set that included "ground" — which the C++ builder does NOT
    treat as ground. Every instance's "ground" then collapsed onto
    one ordinary floating node, silently shorting the instances
    together. `net()` asks the builder instead of guessing.
    """
    sm = p.define_subsystem("G", ports=("a",), params={})

    @sm.body
    def _(s):
        s.add_resistor("R", "a", "ground", 1.0)   # NOT ground
        s.add_resistor("R2", "a", "gnd", 1.0)     # IS ground

    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 5.0)
    sm.instantiate(b, "u1", a="vin")
    sm.instantiate(b, "u2", a="vin")

    # "ground" is an ordinary internal net -> scoped per instance
    assert b.node_id_of("u1/ground") != b.node_id_of("u2/ground")
    # "gnd" is the real thing -> untouched, shared, never scoped
    with pytest.raises(Exception):
        b.node_id_of("u1/gnd")


def test_building_many_cells_stays_linear():
    """"Circuits of any size" is the point, so the duplicate-name
    check must not make the build quadratic. It did at first (a
    scan per add: 4x the cells cost 7x the time); a name index
    fixes both that and `branch_id_of`, which was itself an
    O(num_branches) scan.
    """
    import time

    def build(n):
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "p", "gnd", 1.0)
        t0 = time.perf_counter()
        prev = "p"
        for i in range(n):
            node = f"n{i}"
            b.add_switch(f"sm{i}__Sb", prev, node, 1e3, 1e-9)
            b.add_switch(f"sm{i}__Si", prev, f"sm{i}__m", 1e3,
                          1e-9)
            b.add_capacitor(f"sm{i}__C", f"sm{i}__m", node, 2e-3)
            prev = node
        return time.perf_counter() - t0, b

    t_small, _ = build(200)
    t_big, big = build(1600)
    # 8x the cells in well under 8^2 the time. Generous bound so a
    # loaded CI box cannot make this flaky; quadratic would be ~64x
    # and shows up immediately.
    assert t_big < t_small * 24, (t_small, t_big)
    # And name lookup does not degrade with size.
    t0 = time.perf_counter()
    for i in range(0, 1600, 32):
        big.branch_id_of(f"sm{i}__C")
    assert (time.perf_counter() - t0) < 0.05
