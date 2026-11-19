"""add-branch-form-ideal-switch 6.4 / 6.5: losses, currents and commutations of ideal (R_on = 0)
branch-form elements. Their current is the solved unknown (result.i reads it); the summary
integrates v·i as the line between each step's end powers."""
import warnings

import numpy as np
import pytest

import pulsim as p
from pulsim import _diode_state

T = 10e-6


def _forced(V_f=0.7):
    b = p.CircuitBuilder()
    b.add_current_source("I", "a", "gnd", 10.0)          # 10 A into the anode
    b.add_diode("D", "a", "gnd", form="branch", R_on=0.0, g_off=1e-9, V_f=V_f)
    r = p.simulate(b, t_end=1e-4, dt=1e-6, engine="pwl",
                   switch_fn=lambda t: p.SwitchStateMask(b.graph.num_switches))
    return b, r


def test_forced_ten_amps_through_an_ideal_diode_drops_v_f_and_dissipates_v_f_i():
    b, r = _forced()
    assert np.asarray(r.v("a"))[-1] == pytest.approx(0.7, abs=1e-12)
    assert np.asarray(r.i("D"))[-1] == pytest.approx(10.0, abs=1e-9)      # the branch unknown
    with warnings.catch_warnings():
        warnings.simplefilter("error")                  # no "left out of the summary" warning
        s = p.device_loss_summary(b, r, t_from=1e-5)    # past the 0 -> 10 A first step
    (d,) = [e for e in s if e["name"] == "D"]
    assert d["form"] == "branch" and d["kind"] == "diode"
    assert d["P_avg"] == pytest.approx(7.0, rel=1e-9)
    assert d["i_avg"] == pytest.approx(10.0, rel=1e-9)
    assert d["duty_conducting"] == 1.0
    tr = _diode_state.diode_state_traces(b, r)
    (t,) = tr.values()
    assert np.all(np.isfinite(t.current)) and t.current[-1] == pytest.approx(10.0, abs=1e-9)
    assert t.conducting[1:].all()            # i_branch, not g_on·(v − V_f) = inf·0


def _ideal_buck(g_off=1e-12):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_ideal_switch("S", "vin", "sw", 0.0, g_off)
    b.add_diode("D", "gnd", "sw", form="branch", R_on=0.0, g_off=g_off)
    b.add_current_source("Iload", "gnd", "sw", 0.5)     # 0.5 A drawn from the switch node
    k, n = b.switch_index_of("S"), b.graph.num_switches

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, (t % T) < 0.4 * T)
        return m
    return b, p.simulate(b, t_end=10 * T, dt=T / 100, engine="pwl", switch_fn=fn)


def test_tabulated_turn_on_energy_of_an_ideal_switch():
    """E_on = 10 uJ·(i/1 A)·(v/24 V) books 5.0 uJ per turn-on at 0.5 A, 24 V. Measured 5.0 uJ
    within 4.8e-11 relative: the OFF leak g_off·24 V = 2.4e-11 A rides on the 0.5 A."""
    b, r = _ideal_buck()
    s = p.device_loss_summary(b, r, switch_specs={"S": {"E_on_ref": 10e-6, "E_off_ref": 0.0,
                                                         "V_ref": 24.0, "I_ref": 1.0}})
    sw = next(e for e in s if e["name"] == "S")
    d = next(e for e in s if e["name"] == "D")
    assert sw["n_turn_on_events"] == 9
    assert sw["E_sw_on_total"] / sw["n_turn_on_events"] == pytest.approx(5.0e-6, rel=1e-10)
    # Ideal elements: only the g_off leak dissipates (3.5e-10 / 2.3e-10 W measured); the
    # v-line x i-line product across each commutation step read 0.038 W.
    assert abs(sw["P_avg"]) < 1e-8 and abs(d["P_avg"]) < 1e-8
    # The gate's duty, 0.4 (the restart-less step read 0.399 / 0.601: the stored gate
    # state was one sample off at each edge; the release step samples it at the step start)
    assert sw["duty_closed"] == pytest.approx(0.4, abs=1e-12)
    assert d["duty_conducting"] == pytest.approx(0.6, abs=1e-12)


def test_commutation_events_include_the_ideal_diode_turn_off():
    b, r = _ideal_buck()
    bid = next(c["branch_id"] for c in b.components() if c.get("name") == "D")
    ev = [e for e in r.commutation_events if e.branch_id == bid]
    offs = [e for e in ev if not e.new_state]
    ons = [e for e in ev if e.new_state]
    assert len(ons) == 10 and len(offs) == 9           # ON at each gate-off, OFF at each gate-on
    for e in offs:                                      # at the gate's turn-on, inside one step
        assert (e.t_estimated % T) < T / 100 + 1e-15 or T - (e.t_estimated % T) < 1e-15


def test_static_path_records_the_ideal_diode_flips():
    """No L or C: the fixed engine runs its static (dt = 0) path. An ideal diode sits at
    v = V_th while ON, so the voltage-crossing test never fires; the record reads its state."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", v_dc=0.0, v_amplitude=10.0, frequency=1e3, phase=0.3)
    b.add_diode("D", "a", "k", form="branch", R_on=0.0, g_off=1e-9)
    b.add_resistor("R", "k", "gnd", 10.0)
    r = p.simulate(b, t_end=3e-3, dt=1e-6, engine="pwl",
                   switch_fn=lambda t: p.SwitchStateMask(b.graph.num_switches))
    bid = next(c["branch_id"] for c in b.components() if c.get("name") == "D")
    ev = [e for e in r.commutation_events if e.branch_id == bid]
    # The first step turns it ON (the start stamp is OFF); sin(2π·1e3·t + 0.3) then crosses
    # zero at 0.45225 + 0.5·k ms. Stamped at the end of the step that flipped (shortcut).
    assert [e.new_state for e in ev] == [True, False, True, False, True, False, True]
    assert ev[0].t_estimated == pytest.approx(1e-6, abs=1e-15)
    for k, e in enumerate(ev[1:]):
        lag = e.t_estimated - (0.45225e-3 + 0.5e-3 * k)
        assert 0.0 < lag <= 1e-6 + 1e-12


def test_ideal_element_specs_that_cannot_apply_warn():
    b, r = _forced()
    with pytest.warns(RuntimeWarning, match="conduction_specs for ideal"):
        p.device_loss_summary(b, r, conduction_specs={"D": {"V_f": 0.7}})
