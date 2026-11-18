"""add-branch-form-ideal-switch D6: ``form=`` wherever a PWL switch or an
ungated PWL diode is created (add_switch, the MOSFET / IGBT helpers, the
topology helpers, YAML), default 'conductance' (unchanged), validated by
name. The branch form is the same element as add_ideal_switch /
add_diode(form='branch')."""
from __future__ import annotations

import math

import numpy as np
import pytest

import pulsim as p


def _comps(b):
    return {c["name"]: c["params"] for c in b.components()}


def _branch(b, name):
    return _comps(b)[name].get("branch_form", 0) == 1


# ---- add_switch ---------------------------------------------------------------
def test_add_switch_form():
    b = p.CircuitBuilder()
    b.add_switch("S0", "a", "b", 1e3, 1e-9)                      # default unchanged
    b.add_switch("S1", "a", "b", form="branch")                  # ideal
    b.add_switch("S2", "a", "b", 1e3, 1e-9, form="branch")       # R_on = 1/g_on
    b.add_switch("S3", "a", "b", form="branch", R_on=0.5, g_off=1e-6)
    c = _comps(b)
    assert not _branch(b, "S0") and c["S0"]["g_on"] == 1e3
    assert _branch(b, "S1") and c["S1"]["R_on"] == 0.0 and c["S1"]["g_off"] == 0.0
    assert c["S2"]["R_on"] == pytest.approx(1e-3) and c["S2"]["g_off"] == 1e-9
    assert c["S3"]["R_on"] == 0.5 and c["S3"]["g_off"] == 1e-6
    with pytest.raises(ValueError, match=r"add_switch\(\"X\"\): form='ideal'"):
        b.add_switch("X", "a", "b", 1e3, 1e-9, form="ideal")
    with pytest.raises(ValueError, match="conductance form takes g_on"):
        b.add_switch("X", "a", "b")
    with pytest.raises(ValueError, match="conductance form takes g_on"):
        b.add_switch("X", "a", "b", 1e3, 1e-9, R_on=1.0)


def test_add_switch_branch_is_add_ideal_switch():
    def run(add):
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "a", "gnd", 10.0)
        add(b)
        b.add_inductor("L", "k", "o", 1e-4)
        b.add_resistor("R", "o", "gnd", 2.0)
        m = p.SwitchStateMask(1)
        m.set(0, True)
        return np.asarray(p.simulate(b, t_end=2e-4, dt=1e-6, engine="pwl",
                                     switch_fn=lambda t: m).states)
    a = run(lambda b: b.add_switch("S", "a", "k", form="branch", R_on=0.01, g_off=1e-9))
    c = run(lambda b: b.add_ideal_switch("S", "a", "k", 0.01, 1e-9))
    assert np.array_equal(a, c)     # one code path: identical on this machine


# ---- device helpers --------------------------------------------------------------
@pytest.mark.parametrize("form", ["conductance", "branch"])
def test_mosfet_helpers_form(form):
    b = p.CircuitBuilder()
    b.add_mosfet("M1", "d", "s", form=form)
    b.add_mosfet("M2", "d", "s", body_diode=False, form=form)
    b.add_mosfet_with_body_diode("M3", "d", "s", form=form)
    br = form == "branch"
    for n in ("M1", "M1_body", "M2", "M3", "M3_body"):
        assert _branch(b, n) == br, n
    c = _comps(b)
    if br:
        assert c["M1"]["R_on"] == pytest.approx(1e-3)
        assert c["M3_body"]["R_on"] == pytest.approx(1e-3)
        assert c["M3_body"]["V_f"] == 0.7


def test_mosfet_branch_takes_ideal_values():
    b = p.CircuitBuilder()
    b.add_mosfet_with_body_diode("M", "d", "s", R_on=0.0, R_off=math.inf,
                                 g_on_diode=math.inf, g_off_diode=0.0, form="branch")
    c = _comps(b)
    assert c["M"]["R_on"] == 0.0 and c["M"]["g_off"] == 0.0
    assert c["M_body"]["R_on"] == 0.0
    with pytest.raises(ValueError, match="needs a finite on-conductance"):
        p.CircuitBuilder().add_mosfet("M", "d", "s", R_on=0.0)        # conductance
    with pytest.raises(ValueError, match="form='brnach'"):
        p.CircuitBuilder().add_mosfet("M", "d", "s", form="brnach")


def test_igbt_form():
    b = p.CircuitBuilder()
    b.add_igbt("Q", "c", "e", reverse_blocking=False, form="branch")
    assert _branch(b, "Q") and _comps(b)["Q"]["R_on"] == pytest.approx(10e-3)
    with pytest.raises(ValueError, match="gated PWL diode has no branch form"):
        b.add_igbt("Q2", "c", "e", reverse_blocking=True, form="branch")


def test_level1_helpers_form():
    b = p.CircuitBuilder()
    b.add_mosfet_level1("M", "d", "s", "g", K=1.0, V_T=2.0, form="branch")
    b.add_igbt_level1("Q", "c", "e", "g", with_fwd=True, form="branch")
    b.add_mosfet_level1("M0", "d", "s", "g", K=1.0, V_T=2.0)
    assert _branch(b, "M_body") and _branch(b, "Q_fwd") and not _branch(b, "M0_body")
    assert _comps(b)["M_body"]["R_on"] == pytest.approx(1e-3)
    # no diode to put in the branch form: refused, not ignored
    with pytest.raises(ValueError, match="with_fwd=false builds none"):
        b.add_igbt_level1("Q9", "c", "e", "g", form="branch")
    with pytest.raises(ValueError, match="with_body_diode=false builds none"):
        b.add_mosfet_level1("M9", "d", "s", "g", K=1.0, V_T=2.0,
                            with_body_diode=False, form="branch")


# ---- topology helpers ------------------------------------------------------------
def _vsi(**kw):
    b = p.CircuitBuilder()
    p.add_three_phase_vsi(b, "INV", vdc_pos="p", vdc_neg="n",
                          out_a="a", out_b="b", out_c="c", **kw)
    return b


def test_vsi_form_and_zero_resistance():
    b = _vsi(form="branch", R_on=0.0, R_d=0.0)
    c = _comps(b)
    assert c["INV__HSa"]["R_on"] == 0.0 and _branch(b, "INV__HSa")
    assert c["INV__Dha"]["R_on"] == 0.0 and _branch(b, "INV__Dha")
    # conductance: unchanged numbers for R_on > 0 ...
    c0 = _comps(_vsi())
    assert c0["INV__HSa"]["g_on"] == 1.0 / 1e-3 and c0["INV__Dha"]["g_on"] == 1.0 / 0.05
    # ... and R_on = 0 / R_d = 0 refused by name (it used to become 1 mOhm)
    with pytest.raises(ValueError, match="R_on = 0.0: the conductance form needs R_on > 0"):
        _vsi(R_on=0.0)
    with pytest.raises(ValueError, match="R_d = 0.0: the conductance form needs R_d > 0"):
        _vsi(R_d=0.0)
    with pytest.raises(ValueError, match="R_off = 0"):
        _vsi(R_off=0.0, form="branch")
    with pytest.raises(ValueError, match="form='x'"):
        _vsi(form="x")


@pytest.mark.parametrize("helper,sw,d", [
    (p.add_buck, "buck__Q", "buck__D"),
    (p.add_boost, "boost__Q", "boost__D"),
    (p.add_flyback, "flyback__Q", "flyback__D"),
])
def test_smps_helpers_form(helper, sw, d):
    b = p.CircuitBuilder()
    r = helper(b, form="branch", g_on=math.inf)
    names = (r.switch_name, r.diode_name)
    c = _comps(b)
    assert _branch(b, names[0]) and c[names[0]]["R_on"] == 0.0
    assert _branch(b, names[1]) and c[names[1]]["R_on"] == pytest.approx(1e-3)
    b0 = p.CircuitBuilder()
    helper(b0)
    assert not _branch(b0, names[0]) and not _branch(b0, names[1])


def test_bridge_rectifier_form():
    b = p.CircuitBuilder()
    r = p.add_bridge_rectifier(b, "BR", ac_a="x", ac_b="y", dc_pos="p", dc_neg="n",
                               g_on=math.inf, g_off=0.0, form="branch")
    assert all(_branch(b, n) and _comps(b)[n]["R_on"] == 0.0 for n in r.diode_names)


# ---- YAML and the circuit IR -------------------------------------------------------
def test_yaml_form_on_helpers():
    text = (
        "circuit:\n  devices:\n"
        "    - {type: mosfet, name: M1, drain: d, source: s, form: branch}\n"
        "    - {type: mosfet_with_body_diode, name: M2, drain: d, source: s, form: branch}\n"
        "    - {type: igbt, name: Q1, collector: d, emitter: s, reverse_blocking: false,"
        " form: branch}\n"
        "    - {type: mosfet_level1, name: M3, drain: d, source: s, gate: g, K: 1.0,"
        " V_T: 2.0, form: branch}\n"
        "    - {type: mosfet, name: M4, drain: d, source: s}\n")
    b = p.load_yaml_string(text)
    b = getattr(b, "builder", b)
    for n in ("M1", "M1_body", "M2", "M2_body", "Q1", "M3_body"):
        assert _branch(b, n), n
    assert not _branch(b, "M4") and not _branch(b, "M4_body")
    bad = text.replace("{type: mosfet, name: M4, drain: d, source: s}",
                       "{type: resistor, name: R, from: d, to: s, R: 1.0, form: branch}")
    with pytest.raises(Exception, match="form: 'branch' is not supported"):
        p.load_yaml_string(bad)


def test_ir_hash_tells_the_forms_apart():
    def h(form):
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "d", "gnd", 1.0)
        b.add_mosfet("M", "d", "s", form=form)
        b.add_switch("S", "s", "gnd", 1e3, 1e-9, form=form)
        b.add_resistor("R", "s", "gnd", 1.0)
        c = b.build()
        return c.structure_hash(), c.value_hash()
    assert h("conductance") != h("branch")
    assert h("branch") == h("branch")


# ---- DC point: a closed ideal switch in a voltage loop (DECISIONS #41) --------------
def _loop(**sw):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 12.0)
    b.add_inductor("L", "a", "k", 1e-4)
    b.add_switch("S", "k", "gnd", form="branch", **sw)
    b.add_resistor("R", "k", "gnd", 10.0)
    return b


def _dc(b, closed):
    m = p.SwitchStateMask(1)
    m.set(0, closed)
    return np.asarray(p.compute_dc_op(b, mask=m))


def test_dc_ideal_voltage_loop_is_refused_by_name():
    with pytest.raises(p.UnsupportedError) as ei:
        _dc(_loop(), closed=True)             # V - L - S(R_on = 0): 12 V around a short
    assert ei.value.feature == "ideal_voltage_loop"
    assert "S" in str(ei.value) and "V" in str(ei.value)
    # the same switch open, or with R_on > 0, has a DC point
    assert np.all(np.isfinite(_dc(_loop(), closed=False)))
    assert np.all(np.isfinite(_dc(_loop(R_on=0.5), closed=True)))
