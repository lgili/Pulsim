"""The circuit IR against what arrived after its base (integration batch 8):
the branch-form ideal switch and diode, the capacitor-IC rule of the
examples audit, and the compiled control graph."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p
from pulsim import _pulsim as k


def _dependent(diode=True):
    # Without the diode the switch opens an inductor cut-set: pwl +
    # restart re-initialises (one reinit record).
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S", "vin", "a", 0.0, 0.0)
    b.add_resistor("R", "a", "m", 1.0)
    b.add_inductor("L", "m", "gnd", 1e-3, 0.25)
    if diode:
        b.add_diode("D", "gnd", "m", form="branch", R_on=1e-3, g_off=1e-9,
                    V_th=0.0, V_f=0.0)
    return b


def _fn(t):
    m = p.SwitchStateMask(1)
    m.set(0, t < 2e-4)
    return m


def test_branch_form_circuit_is_lowered_as_built():
    b = _dependent()
    c = b.build()
    assert c.components() == b.components()
    assert c.state_var_names() == b.state_var_names()
    # one current unknown per branch-form element, at the builder's index
    assert sum("S" in n or "(D)" in n for n in c.state_var_names()) >= 2
    assert [c.switch_index_of(n) for n in ("S", "D")] == \
        [b.switch_index_of(n) for n in ("S", "D")]
    np.testing.assert_array_equal(c.initial_state(), b.initial_state())
    assert c.to_builder().components() == b.components()


def test_branch_form_run_through_a_circuit_equals_the_builder():
    kw = dict(engine="pwl", dt=5e-6, switch_fn=_fn, discontinuity_restart=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rb = p.simulate(_dependent(False), 3e-4, **kw)
        rc = p.simulate(_dependent(False).build(), 3e-4, **kw)
    np.testing.assert_array_equal(rc.states, rb.states)
    assert len(rb.reinit_records) == len(rc.reinit_records) >= 1
    assert [r.t for r in rc.reinit_records] == [r.t for r in rb.reinit_records]


def test_branch_form_is_structure_and_its_r_on_a_value():
    def one(branch, R_on=1e-3):
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "a", "gnd", 1.0)
        if branch:
            b.add_ideal_switch("S", "a", "o", R_on, 1e-9)
        else:
            b.add_switch("S", "a", "o", 1.0 / R_on, 1e-9)
        b.add_resistor("R", "o", "gnd", 10.0)
        return b.build()
    # same kind, same g_on / g_off: the form alone tells them apart
    assert one(True).structure_hash() != one(False).structure_hash()
    assert one(True, 1e-3).structure_hash() == one(True, 2e-3).structure_hash()
    assert one(True, 1e-3).value_hash() != one(True, 2e-3).value_hash()
    assert one(True, 0.0).structure_hash() != one(True, 1e-3).structure_hash()


# kind -> (builder of a branch-form element "X", one value per field)
FORMS = {
    "Switch": (lambda b, R_on, g_off: b.add_ideal_switch("X", "a", "o", R_on, g_off)),
    "Diode": (lambda b, R_on, g_off: b.add_diode(
        "X", "a", "o", form="branch", R_on=R_on, g_off=g_off, V_th=0.0, V_f=0.0)),
}


def _form(kind, R_on=0.0, g_off=0.0):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vs", "a", "gnd", 5.0)
    FORMS[kind](b, R_on, g_off)
    b.add_resistor("R", "o", "gnd", 10.0)
    return b


def test_every_branch_form_has_a_row():
    assert set(FORMS) == set(k._branch_form_capabilities())


@pytest.mark.parametrize("field,old,new", [
    ("R_on", 0.0, 1e-3), ("R_on", 1e-3, 0.0), ("g_off", 0.0, 1e-9)])
@pytest.mark.parametrize("kind", sorted(FORMS))
def test_branch_form_with_value_equals_rebuilding(kind, field, old, new):
    args = {"R_on": 0.0, "g_off": 0.0, field: old}
    receiver = _form(kind, **args).build()
    derived = receiver.with_value(f"X.{field}", new)
    rebuilt = _form(kind, **{**args, field: new}).build()
    assert derived.get_value(f"X.{field}") == new
    assert receiver.get_value(f"X.{field}") == old
    assert derived.value_hash() == rebuilt.value_hash()
    assert derived.structure_hash() == rebuilt.structure_hash()
    # R_on = 0 (an exact short) and g_off = 0 (an exact open) are structure:
    # crossing zero changes it, every case here crosses zero.
    assert derived.structure_hash() != receiver.structure_hash()
    fn = p.make_pwm_switch_fn(1e4, 0.5, 0, 1) if kind == "Switch" else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ra = p.simulate(derived, 2e-4, dt=1e-6, engine="pwl", switch_fn=fn)
        rb = p.simulate(rebuilt, 2e-4, dt=1e-6, engine="pwl", switch_fn=fn)
    np.testing.assert_array_equal(ra.states, rb.states)


@pytest.mark.parametrize("kind", sorted(FORMS))
@pytest.mark.parametrize("field", ["g_on", "R_off"])
def test_branch_form_refuses_a_field_its_law_does_not_read(kind, field):
    b = _form(kind)
    with pytest.raises(ValueError, match=r'"X" is in branch form.*R_on, g_off'):
        b.build().with_value(f"X.{field}", 1.0)
    with pytest.raises(ValueError, match="is in branch form"):
        b.set_value(f"X.{field}", 1.0)


def test_branch_form_r_on_keeps_the_builder_bound():
    with pytest.raises(ValueError, match="R_on"):
        _form("Switch").build().with_value("X.R_on", -1.0)


def test_a_parameter_cannot_drive_a_branch_form_r_on_yet():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vs", "a", "gnd", 5.0)
    r = b.param("Ron", 1e-3)
    with pytest.raises(ValueError, match=r'"X" is in branch form; a parameter'):
        b.add_ideal_switch("X", "a", "o", r, 0.0)


# ------------------------------------------------- capacitor-IC rule
def _kvl_loop():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 1.0)
    b.add_resistor("R", "in", "a", 1e3)
    b.add_capacitor("C1", "a", "gnd", 1e-6, c0=5.0)
    b.add_capacitor("C2", "a", "gnd", 1e-6, c0=3.0)
    return b


def test_a_kvl_breaking_capacitor_ic_is_refused_through_a_circuit():
    b = _kvl_loop()
    c = b.build()   # the IR stores what the builder stores; build() accepts
    with pytest.raises(ValueError, match="break KVL"):
        c.initial_state()
    for engine, kw in (("pwl", {"dt": 1e-6}), ("trbdf2", {})):
        with pytest.raises(ValueError, match="break KVL"):
            p.simulate(c, 1e-4, engine=engine, **kw)


# ---------------------------------------------------- control graph
def test_a_control_graph_rides_along_as_the_step_observer():
    # A Circuit holds no controller; the graph is carried by simulate()'s
    # step_observer, bound to the builder's node indices, which the
    # circuit keeps. Same result through the builder and the circuit.
    def build():
        b = p.CircuitBuilder()
        b.add_voltage_source("V1", "in", "gnd", 1.0)
        b.add_resistor("R", "in", "out", 1e3)
        b.add_capacitor("C", "out", "gnd", 1e-6)
        b.add_resistor("Rd", "drv", "gnd", 1e3)
        ch = p.MixedDomainBlockChain(rate=p.Rate(1e-4))
        ch.add("g", p.Gain(k=2.0), inputs=dict(x="out"), output="y")
        ch.drive_source(b, "y", ("v", "drv", "gnd"))
        return b, ch
    b1, ch1 = build()
    r1 = p.simulate(b1, 2e-3, engine="pwl", dt=1e-6,
                    step_observer=ch1.make_step_observer(b1))
    b2, ch2 = build()
    obs2 = ch2.make_step_observer(b2)
    r2 = p.simulate(b2.build(), 2e-3, engine="pwl", dt=1e-6,
                    step_observer=obs2)
    np.testing.assert_array_equal(r2.states, r1.states)
    assert ch1.tick_counts == ch2.tick_counts == [20]
    assert np.asarray(r2.v("drv"))[-1] > 1.0



# ---- group 3.1: initial conditions per device, KVL, graph-driven sources ---
def test_initial_conditions_are_per_device():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_resistor("R", "a", "x", 1.0)
    b.add_capacitor("C1", "x", "y", 1e-6)
    b.add_capacitor("C2", "y", "gnd", 1e-6)
    b.add_inductor("L", "x", "gnd", 1e-3)
    b.set_initial("C1", 3.0)       # the voltage ACROSS C1
    b.set_initial("C2", 2.0)
    b.set_initial("L", 0.5)        # the current THROUGH L
    c = b.build()
    assert c.initial_conditions() == {"C1": 3.0, "C2": 2.0, "L": 0.5}
    x0 = np.asarray(c.initial_state())
    assert x0[c.node_id_of("x")] - x0[c.node_id_of("y")] == 3.0
    assert x0[c.node_id_of("y")] == 2.0
    assert x0[c.state_var_names().index("I(L)")] == 0.5


def test_graph_driven_sources_ride_in_the_circuit():
    from pulsim._circuit_ir import graph_driven_of
    from pulsim.periodic_steady_state import _refuse_graph_driven
    b = p.CircuitBuilder()
    b.add_voltage_source("graph_u_0", "a", "gnd", 0.0)
    b.add_resistor("R", "a", "gnd", 1.0)
    b._graph_driven = ["graph_u_0"]    # what blockchain.drive_source records
    c = b.build()
    assert graph_driven_of(c) == ["graph_u_0"]
    assert graph_driven_of(c.to_builder()) == ["graph_u_0"]
    with pytest.raises(p.UnsupportedError, match="graph_u_0"):
        _refuse_graph_driven(c)
