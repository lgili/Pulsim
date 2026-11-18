"""add-immutable-circuit-ir group 6: generic value updates and named
parameters.

``Circuit.with_value("dev.field", v)`` must equal rebuilding the circuit
with the new literal, bit for bit, for a field of EVERY device kind — the
table below is checked against ``engine_capabilities()`` so a new kind
cannot be forgotten — and must leave the receiver unchanged.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p


def _src(b):
    b.add_voltage_source("Vs", "a", "gnd", 5.0)


# kind -> (field "X<suffix>.<field>", old, new, build(b, value)).
# Each adds the device under test as "X" beside a small support circuit.
KINDS = {
    "Resistor": ("X.R", 10.0, 22.0, lambda b, v: (
        _src(b), b.add_resistor("X", "a", "gnd", v))),
    "VoltageSource": ("X.V", 5.0, 7.5, lambda b, v: (
        b.add_voltage_source("X", "a", "gnd", v), b.add_resistor("R", "a", "gnd", 10.0))),
    "Switch": ("X.g_on", 1e3, 2e3, lambda b, v: (
        _src(b), b.add_switch("X", "a", "o", v, 1e-9), b.add_resistor("R", "o", "gnd", 10.0))),
    "Capacitor": ("X.C", 1e-6, 4.7e-6, lambda b, v: (
        _src(b), b.add_resistor("R", "a", "o", 10.0), b.add_capacitor("X", "o", "gnd", v))),
    "Inductor": ("X.L", 1e-3, 2.2e-3, lambda b, v: (
        _src(b), b.add_inductor("X", "a", "o", v), b.add_resistor("R", "o", "gnd", 10.0))),
    "Diode": ("X.V_f", 0.0, 0.7, lambda b, v: (
        _src(b), b.add_diode("X", "a", "o", 1e3, 1e-9, 0.0, v), b.add_resistor("R", "o", "gnd", 10.0))),
    "NonlinearDiode": ("X.R_d", 0.01, 0.05, lambda b, v: (
        _src(b), b.add_nonlinear_diode("X", "a", "o", p.IdealDiodeParams(R_d=v),
                                       allow_non_passive=True),
        b.add_resistor("R", "o", "gnd", 10.0))),
    "CurrentSource": ("X.I", 1.0, 2.5, lambda b, v: (
        b.add_current_source("X", "gnd", "a", v), b.add_resistor("R", "a", "gnd", 10.0))),
    "PWMVoltageSource": ("X.duty", 0.5, 0.3, lambda b, v: (
        b.add_pwm_voltage_source("X", "a", "gnd", 1.0, 0.0, 1e4, v),
        b.add_resistor("R", "a", "gnd", 10.0))),
    "SineVoltageSource": ("X.v_amplitude", 1.0, 3.0, lambda b, v: (
        b.add_sine_voltage_source("X", "a", "gnd", 0.0, v, 1e3),
        b.add_resistor("R", "a", "gnd", 10.0))),
    "PulseVoltageSource": ("X.v_pulsed", 1.0, 4.0, lambda b, v: (
        b.add_pulse_voltage_source("X", "a", "gnd", 0.0, v, 0.0, 1e-4, 2e-4, 0.0, 0.0),
        b.add_resistor("R", "a", "gnd", 10.0))),
    "MosfetLevel1": ("X.K", 5.0, 7.0, lambda b, v: (
        _src(b), b.add_voltage_source("Vg", "g", "gnd", 10.0),
        b.add_resistor("R", "a", "d", 1.0), b.add_mosfet_level1("X", "d", "gnd", "g", v, 3.0))),
    "IgbtLevel1": ("X.V_CE_sat", 1.5, 1.2, lambda b, v: (
        _src(b), b.add_voltage_source("Vg", "g", "gnd", 15.0),
        b.add_resistor("R", "a", "c", 1.0), b.add_igbt_level1("X", "c", "gnd", "g", v))),
    "VCVS": ("X.gain", 2.0, 3.0, lambda b, v: (
        _src(b), b.add_resistor("Ra", "a", "gnd", 10.0),
        b.add_vcvs("X", "a", "gnd", "o", "gnd", v), b.add_resistor("R", "o", "gnd", 10.0))),
    "SaturableInductor": ("X.L_0", 1e-3, 2e-3, lambda b, v: (
        b.add_sine_voltage_source("Vs", "a", "gnd", 0.0, 5.0, 1e3),
        b.add_resistor("R", "a", "o", 1.0), b.add_saturable_inductor("X", "o", "gnd", v, 2.0, 1e-5))),
    "NonlinearCapacitor": ("X.C0", 1e-9, 2e-9, lambda b, v: (
        _src(b), b.add_resistor("R", "a", "o", 10.0),
        b.add_nonlinear_capacitor("X", "o", "gnd", v, 1.0, 0.5))),
    "ShockleyDiode": ("X.I_S", 1e-12, 1e-11, lambda b, v: (
        _src(b), b.add_shockley_diode("X", "a", "o", v), b.add_resistor("R", "o", "gnd", 10.0))),
    "LauritzenDiode": ("X.I_S", 1e-12, 1e-11, lambda b, v: (
        _src(b), b.add_lauritzen_diode("X", "a", "o", I_S=v), b.add_resistor("R", "o", "gnd", 10.0))),
    "PmsmMna": ("X_a.R_s", 0.5, 0.8, lambda b, v: (
        b.add_sine_voltage_source("Va", "pa", "gnd", 0.0, 10.0, 50.0),
        b.add_sine_voltage_source("Vb", "pb", "gnd", 0.0, 10.0, 50.0, -2.0943951023931953),
        b.add_sine_voltage_source("Vc", "pc", "gnd", 0.0, 10.0, 50.0, 2.0943951023931953),
        b.add_pmsm_mna("X", "pa", "pb", "pc", "n", "om", "th", v, 2e-3, 2e-3, 0.05, 4.0, 1e-3),
        b.add_resistor("Rn", "n", "gnd", 1e6))),
    "IdealTransformer": ("X.n", 2.0, 3.0, lambda b, v: (
        _src(b), b.add_resistor("Rp", "a", "p", 1.0),
        b.add_ideal_transformer("X", "p", "gnd", "s", "gnd", v), b.add_resistor("R", "s", "gnd", 10.0))),
    "VCCS": ("X.gm", 0.1, 0.2, lambda b, v: (
        _src(b), b.add_resistor("Ra", "a", "gnd", 10.0),
        b.add_vccs("X", "a", "gnd", "o", "gnd", v), b.add_resistor("R", "o", "gnd", 10.0))),
    "CCCS": ("X.gain", 2.0, 3.0, lambda b, v: (
        _src(b), b.add_resistor("Ra", "a", "gnd", 10.0),
        b.add_cccs("X", "Vs", "o", "gnd", v), b.add_resistor("R", "o", "gnd", 10.0))),
    "CCVS": ("X.rm", 2.0, 3.0, lambda b, v: (
        _src(b), b.add_resistor("Ra", "a", "gnd", 10.0),
        b.add_ccvs("X", "Vs", "o", "gnd", v), b.add_resistor("R", "o", "gnd", 10.0))),
    "Gyrator": ("X.g", 0.1, 0.3, lambda b, v: (
        _src(b), b.add_resistor("Ra", "a", "p", 10.0),
        b.add_gyrator("X", "p", "gnd", "o", "gnd", v), b.add_resistor("R", "o", "gnd", 10.0))),
    "SineCurrentSource": ("X.i_amplitude", 1.0, 2.0, lambda b, v: (
        b.add_sine_current_source("X", "gnd", "a", 0.0, v, 1e3),
        b.add_resistor("R", "a", "gnd", 10.0))),
    "PulseCurrentSource": ("X.i_pulsed", 1.0, 2.0, lambda b, v: (
        b.add_pulse_current_source("X", "gnd", "a", 0.0, v, 0.0, 1e-4, 2e-4, 0.0, 0.0),
        b.add_resistor("R", "a", "gnd", 10.0))),
}
# The PWL current source's knots are a table, not a scalar field: it has
# no `with_value` field in this slice (DECISIONS.md 4) and is refused by
# name — checked in test_table_kind_is_refused_by_name.
TABLE_KINDS = {"PwlCurrentSource"}


def _build(kind, v):
    b = p.CircuitBuilder()
    KINDS[kind][3](b, v)
    return b


def test_every_kind_has_a_row():
    assert set(KINDS) | TABLE_KINDS == set(p.engine_capabilities())


def _run(c):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = p.simulate(c, 2e-4, dt=1e-6, engine="pwl")
            return np.asarray(r.states).copy()
        except Exception as exc:  # noqa: BLE001 — compared, not swallowed
            return f"{type(exc).__name__}: {exc}"


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_with_value_equals_rebuilding(kind):
    field, old, new, _ = KINDS[kind]
    receiver = _build(kind, old).build()
    h0 = receiver.value_hash()
    derived = receiver.with_value(field, new)
    rebuilt = _build(kind, new).build()
    # The kind under test is the one edited.
    dev = field.rsplit(".", 1)[0]
    assert next(d for d in derived.devices() if d["name"] == dev)["kind"] == kind
    assert derived.get_value(field) == new
    assert derived.value_hash() == rebuilt.value_hash()
    assert derived.structure_hash() == receiver.structure_hash()
    assert receiver.value_hash() == h0 and receiver.get_value(field) == old
    a, b = _run(derived), _run(rebuilt)
    if isinstance(a, str) or isinstance(b, str):
        assert a == b
    else:
        np.testing.assert_array_equal(a, b)


def test_table_kind_is_refused_by_name():
    b = p.CircuitBuilder()
    b.add_pwl_current_source("X", "gnd", "a", [0.0, 1e-3], [0.0, 1.0])
    b.add_resistor("R", "a", "gnd", 10.0)
    with pytest.raises(ValueError, match=r'device "X" \(PwlCurrentSource\) has no '
                                         r'value field "i"; its fields: none'):
        b.build().with_value("X.i", 1.0)


def test_with_value_refusals_name_device_and_field():
    c = _build("Resistor", 10.0).build()
    with pytest.raises(ValueError, match='no device named "Y"'):
        c.with_value("Y.R", 1.0)
    with pytest.raises(ValueError, match=r'field R of device "X" must be a finite number > 0 \(got -1\)'):
        c.with_value("X.R", -1.0)
    with pytest.raises(ValueError, match=r'has no value field "Q"; its fields: R, G'):
        c.with_value("X.Q", 1.0)
    with pytest.raises(ValueError, match=r'field duty of device "X" must be a number in \[0, 1\]'):
        _build("PWMVoltageSource", 0.5).build().with_value("X.duty", 1.5)


def test_power_mosfet_composite_is_refused():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vs", "a", "gnd", 12.0)
    b.add_voltage_source("Vg", "g", "gnd", 10.0)
    b.add_resistor("R", "a", "d", 1.0)
    b.add_power_mosfet("Q", "d", "gnd", "g", K=1.43, V_T=3.0, C_gs=1e-9, C_gd=100e-12, C_ds=0.0)
    with pytest.raises(ValueError, match="belongs to the power MOSFET"):
        b.build().with_value("Q.K", 1.0)


# ---------------------------------------------------------------------------
# Named parameters (6.1, 6.2)
# ---------------------------------------------------------------------------
def _rlc(Lv, Rv):
    """Buck-like RLC; L and the load bound to parameters."""
    b = p.CircuitBuilder()
    L = b.param("Lbase", Lv, "H", lo=1e-9)
    R = b.param("Rload", Rv, "Ohm")
    b.add_voltage_source("Vs", "a", "gnd", 12.0)
    b.add_inductor("L1", "a", "o", L)            # positional
    b.add_capacitor("C1", "o", "gnd", 10e-6)
    b.add_resistor("RL", "o", "gnd", R_ohms=R)    # keyword
    b.add_resistor("R2", "o", "gnd", R)           # one parameter, two fields
    return b


def test_parameters_drive_their_fields():
    c = _rlc(1e-3, 5.0).build()
    assert [(q.name, q.value, q.unit) for q in c.parameters()] == [
        ("Lbase", 1e-3, "H"), ("Rload", 5.0, "Ohm")]
    devs = {d["name"]: d for d in c.devices()}
    assert devs["L1"]["params"] == {"L": "Lbase"}
    assert devs["RL"]["params"] == {"R": "Rload"}
    assert devs["R2"]["params"] == {"R": "Rload"}
    assert devs["C1"]["params"] == {}
    assert c.get_value("L1.L") == 1e-3 and c.get_value("R2.R") == 5.0


def test_with_params_equals_rebuilding():
    receiver = _rlc(1e-3, 5.0).build()
    h0, s0 = receiver.value_hash(), receiver.structure_hash()
    derived = receiver.with_params({"Lbase": 2.2e-3}, Rload=8.0)
    rebuilt = _rlc(2.2e-3, 8.0).build()
    assert derived.value_hash() == rebuilt.value_hash() != h0
    assert derived.structure_hash() == s0
    assert receiver.value_hash() == h0 and receiver.get_value("L1.L") == 1e-3
    assert derived.get_value("R2.R") == 8.0
    np.testing.assert_array_equal(_run(derived), _run(rebuilt))


def test_binding_is_structure_a_literal_is_not():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vs", "a", "gnd", 12.0)
    b.add_resistor("RL", "a", "gnd", 5.0)
    literal = b.build()
    b2 = p.CircuitBuilder()
    R = b2.param("Rload", 5.0)
    b2.add_voltage_source("Vs", "a", "gnd", 12.0)
    b2.add_resistor("RL", "a", "gnd", R)
    bound = b2.build()
    assert bound.structure_hash() != literal.structure_hash()
    # with_value turns the field back into a literal: no longer driven.
    lit = bound.with_value("RL.R", 5.0)
    assert lit.devices()[1]["params"] == {}
    assert lit.with_params(Rload=9.0).get_value("RL.R") == 5.0


def test_parameter_refusals_name_parameter_device_and_field():
    c = _rlc(1e-3, 5.0).build()
    with pytest.raises(ValueError, match=r'no parameter named "Lx"; parameters: Lbase, Rload'):
        c.with_params(Lx=1.0)
    with pytest.raises(ValueError, match=r'parameter "Lbase" must be a finite number >= 1e-09 \(got 0\)'):
        c.with_params(Lbase=0.0)
    with pytest.raises(ValueError, match=r'parameter "Rload" = -1: field R of device "RL" must be a finite number > 0'):
        c.with_params(Rload=-1.0)
    assert c.get_value("RL.R") == 5.0
    b = p.CircuitBuilder()
    b.param("x", 1.0)
    with pytest.raises(ValueError, match='a parameter of that name already exists'):
        b.param("x", 2.0)
    g = b.param("g", 0.5)
    # i0 is an initial condition, not a value field (group 5).
    with pytest.raises(ValueError, match=r"add_inductor\('L'\): argument 'i0' cannot take the "
                                         r'parameter \'g\'.*has no value field "i0"'):
        b.add_inductor("L", "a", "gnd", 1e-3, g)


def test_update_methods_are_deprecated_and_name_with_value():
    b = _build("Resistor", 10.0)
    b.add_inductor("L", "a", "q", 1e-3)
    b.add_capacitor("C", "q", "gnd", 1e-6)
    for meth, dev, field, v in [("update_resistor_R", "X", "R", 2.0),
                                ("update_inductor_L", "L", "L", 2e-3),
                                ("update_capacitor_C", "C", "C", 2e-6),
                                ("update_voltage_source_V", "Vs", "V", 7.0)]:
        with pytest.warns(DeprecationWarning,
                          match=rf'with_value\("{dev}\.{field}", value\)'):
            getattr(b, meth)(dev, v)
        assert b.get_value(f"{dev}.{field}") == v


# ---------------------------------------------------------------------------
# Review fixes: a field update is checked as its add_* checks the device,
# and a builder whose Python side state a Circuit cannot hold is refused.
# ---------------------------------------------------------------------------
def test_with_value_applies_the_device_checks_of_add():
    nc = _build("NonlinearCapacitor", 1e-9).build()
    with pytest.raises(ValueError, match=r'device "X": NonlinearCapacitor: m must be in \[0, 1\)'):
        nc.with_value("X.m", 1.5)
    with pytest.raises(ValueError, match=r"its junction law does not read cgd_max"):
        nc.with_value("X.cgd_max", 1e-9)
    ig = _build("IgbtLevel1", 1.5).build()
    with pytest.raises(ValueError, match=r"k_tail must be < 1"):
        ig.with_value("X.k_tail", 1.0)
    with pytest.raises(ValueError, match=r"tau_tail and k_tail must be set together"):
        ig.with_value("X.tau_tail", 1e-6)
    pm = _build("PmsmMna", 0.5).build()
    with pytest.raises(ValueError, match=r"pole_pairs|PmsmMna"):
        pm.with_value("X_a.pole_pairs", 0.5)
    # refused derivations leave nothing changed
    assert nc.get_value("X.m") == 0.5 and ig.get_value("X.k_tail") == 0.0


def test_shockley_parameters_follow_the_device_fields():
    from pulsim._builder_ergonomics import shockley_params
    b = _build("ShockleyDiode", 1e-12)
    assert shockley_params(b, "X")["I_S"] == 1e-12
    b.set_value("X.I_S", 3e-12)
    assert shockley_params(b, "X")["I_S"] == 3e-12   # no stale side copy
    assert shockley_params(b, "R") is None


def test_build_refuses_c_blocks_and_simulate_of_the_builder_keeps_them():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vs", "a", "gnd", 1.0)
    b.add_resistor("R", "a", "o", 1.0)
    b.add_resistor("Ro", "o", "gnd", 1.0)
    calls = []

    def step(t, dt, inp, out, state):
        calls.append(t)
        out[0] = 2.0 * inp[0]

    b.add_resistor("Rc", "c", "gnd", 1.0)
    p.add_c_block(b, inputs=[("v", "o")], outputs=[("v", "c", "gnd")],
                  dt=1e-5, fn=step, name="ctl")
    with pytest.raises(p.UnsupportedError, match=r"C blocks \(ctl\).*simulate the builder itself"):
        b.build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p.simulate(b, 1e-4, dt=1e-6, engine="pwl")
    assert calls  # the block ran


def test_mmc_arms_need_the_builder():
    c = _build("Resistor", 10.0).build()
    with pytest.raises(ValueError, match="MMC arms are registered on a CircuitBuilder"):
        p.simulate(c, 1e-4, dt=1e-6, mmc_arms=[object()])


def test_an_unwrapped_add_refuses_a_parameter_loudly():
    # add_nonlinear_capacitor is overloaded and keeps its native form:
    # a Parameter is not a float, so the call fails instead of binding
    # nothing.
    b = p.CircuitBuilder()
    c0 = b.param("C0", 1e-9)
    with pytest.raises(TypeError):
        b.add_nonlinear_capacitor("X", "a", "gnd", c0, 1.0, 0.5)
