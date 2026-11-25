"""A parameter drives a branch-form switch's g_on (as R_on = 1/g_on) and its
g_off (update-number-changing-release blocker 1.6). Checked against the
conductance-form twin built with the same values."""

import math

import numpy as np
import pytest

import pulsim as p

T = 1e-5


def _buck(form, g_on=1e3):
    b = p.CircuitBuilder()
    gon = b.param("Gon", g_on)
    goff = b.param("Goff", 1e-9)
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    if form == "branch":
        b.add_switch("S", "vin", "sw", gon, goff, form="branch")
    else:
        b.add_switch("S", "vin", "sw", gon, goff)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 5.0)
    return b


def _vout(b):
    sf = p.make_pwm_switch_fn(1 / T, 0.4, b.switch_index_of("S"),
                              b.graph.num_switches)
    return np.asarray(p.simulate(b, 30 * T, engine="trbdf2", switch_fn=sf).v("vout"))


def test_a_bound_g_on_sets_r_on_and_follows_set_param():
    b = _buck("branch")
    assert b.get_value("S.R_on") == pytest.approx(1e-3, rel=1e-15)
    b.set_param("Gon", 250.0)
    assert b.get_value("S.R_on") == pytest.approx(4e-3, rel=1e-15)
    b.set_param("Goff", 0.0)             # g_off = 0 is legal in branch form
    assert b.get_value("S.g_off") == 0.0
    b.set_param("Gon", 1e8)             # small R_on (below 1e-9 Ohm is refused)
    assert b.get_value("S.R_on") == pytest.approx(1e-8, rel=1e-15)
    with pytest.raises(ValueError, match="g_on"):
        b.set_param("Gon", -1.0)
    assert b.get_value("S.R_on") == pytest.approx(1e-8, rel=1e-15)  # all or nothing
    with pytest.raises(ValueError, match="finite"):   # a parameter is finite
        b.set_param("Gon", math.inf)
    b.set_value("S.R_on", 0.0)           # R_on = 0 exactly is the ideal switch
    assert b.get_value("S.R_on") == 0.0


def test_a_set_value_drops_the_binding_it_overrides():
    b = _buck("branch")
    b.set_value("S.R_on", 2e-3)
    b.set_param("Gon", 10.0)             # no longer drives R_on
    assert b.get_value("S.R_on") == 2e-3


@pytest.mark.parametrize("g_on", [1e2, 1e3, 1e4])
def test_a_sweep_over_g_on_equals_the_conductance_twin(g_on):
    b = _buck("branch")
    b.set_param("Gon", g_on)
    tw = _buck("conductance")
    tw.set_param("Gon", g_on)
    vb, vt = _vout(b), _vout(tw)
    assert np.abs(vb[-1] - vt[-1]) <= 1e-4 * abs(vt[-1])


def test_circuit_with_params_over_a_branch_form_g_on():
    c = _buck("branch").build()
    for g in (1e2, 1e4):
        assert c.with_params({"Gon": g}).get_value("S.R_on") == pytest.approx(
            1.0 / g, rel=1e-15)


def test_other_branch_form_fields_stay_refused_by_name():
    b = p.CircuitBuilder()
    b.param("X", 1.0)
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_switch("S", "a", "gnd", form="branch", R_on=1e-3, g_off=1e-9)
    with pytest.raises(ValueError, match="branch form"):
        b.bind_param("S.R_off", "X")
