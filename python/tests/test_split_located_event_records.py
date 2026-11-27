"""What the event split books at a located diode crossing (rule and split explicit).

F-Z-16: a diode turn-on the split locates (the cut inside a restart window) is a
continuity point: its ON and OFF laws meet at v = i = 0, so x(t_e) is consistent
in the new mode and nothing jumps. The restart's impulse probe read the new
mode's own fast transient as a jump: D1 (R_on 1 mOhm) clamping a 0.2 nF
junction carries i_L for tau = R_on*C(24 V) = 0.2 ps, i_L*tau = 2.40e-13 C, 10x
the probe's allowance i_L*snap*dt (2.4e-14 C at split_snap 1e-6); measured
before the fix: one capacitor_loop record of 2.4614e-13 C (2.3998e-13 at
split_snap 1e-8: the charge is i_L*tau, not the locator's tolerance)."""

from __future__ import annotations

import warnings

import pytest

import pulsim as p

SPLIT = dict(engine="pwl", discontinuity_restart=True, enable_substep_state_correction=True)


def _switched_junction(t_end, **kw):
    """A 1 nF nonlinear capacitor across a buck switch (release finding F-Z-16's
    cell): S turns off at 5 us (a kink: i_L charges Cj until D1 conducts at
    ~5.0067 us), on at 10 us (a real 1 ps discharge of Cj: a capacitor loop)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "out", 100e-6)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", 2.0)
    b.add_nonlinear_capacitor("Cj", "vin", "sw", 1e-9, 1.0)
    n, k = b.graph.num_switches, b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, (t % 1e-5) < 5e-6 - 1e-12)
        return m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=t_end, dt=2e-8, switch_fn=fn, **kw)


def test_a_located_turn_on_books_no_capacitor_loop():
    r = _switched_junction(9e-6, **SPLIT)
    (ev,) = r.commutation_events                       # D1's located turn-on
    assert ev.new_state and 5.0066e-6 < ev.t_estimated < 5.0068e-6
    assert list(r.reinit_records) == []


def test_a_switched_capacitor_loop_still_records_its_jump():
    """S closing on the charged junction at 10 us: the record and its energy are
    the split-off run's (the ledger: the 1.5e-16 J the probe booked at the
    turn-on is D1's resistive loss of the fast transient, now integrated)."""
    on = list(_switched_junction(1.2e-5, **SPLIT).reinit_records)
    off = list(_switched_junction(1.2e-5, engine="pwl", discontinuity_restart=True).reinit_records)
    assert [x.t for x in on] == [x.t for x in off] == [pytest.approx(1e-5)]
    assert on[0].kind == "capacitor_loop" and list(on[0].states) == ["Cj"]
    assert on[0].lambda_ == pytest.approx(off[0].lambda_, rel=1e-6)
    assert on[0].dW == pytest.approx(off[0].dW, rel=1e-6)


def _snubbed_dcm_boost(form=None):
    """F-Z-17's cell (test_switch_state_record's): the DCM boost with an RC snubber and
    10 nH in series with D1 (V_th 0.7 V); Q1 at 100 kHz, 50 %."""
    b = p.CircuitBuilder()
    fk = dict(form=form) if form else {}
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_inductor("L1", "vin", "sw", 100e-6)
    b.add_switch("Q1", "sw", "gnd", g_on=100.0, g_off=1e-9, **fk)
    b.add_diode("D1", "sw", "x", 1e3, 1e-9, V_th=0.7, **fk)
    b.add_inductor("Ld", "x", "out", 10e-9)
    b.add_resistor("Rs", "sw", "sm", 10.0)
    b.add_capacitor("Cs", "sm", "x", 1e-9)
    b.add_capacitor("C1", "out", "gnd", 47e-6)
    b.add_resistor("RL", "out", "gnd", 500.0)
    q = b.switch_index_of("Q1")

    def pwm(t):
        m = p.SwitchStateMask(b.graph.num_switches)
        m.set(q, (t % 1e-5) < 5e-6)
        return m
    return b, pwm


@pytest.mark.parametrize("form", [None, "branch"])
def test_a_located_turn_on_just_past_stage_1_keeps_the_diode_law(form):
    """F-Z-17: D1's turn-on 1.29 dt after Q1's turn-off (dt = T/200) falls inside the
    restart step's stage 1. The locator's bracket started at t1 = t + gamma*h with
    stage 1's backward-Euler v_D (0.65 V) while its SDIRK trial just past t1 already
    read 0.70-0.75 V: the cut landed at t1 and its left limit (an inner-event row)
    held D1 OFF above V_th, 18 rows by up to 0.0497 V. It now re-brackets on the
    trial's own value at t1, from the step start (max left-limit v_D 0.70055 V)."""
    from pulsim import _result_views as views
    from pulsim._diode_state import diode_state_traces
    b, pwm = _snubbed_dcm_boost(form)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, 2e-3, dt=5e-8, switch_fn=pwm, **SPLIT)
    (d,) = [t for t in diode_state_traces(b, views.with_inner_events(r)).values() if t.name == "D1"]
    assert d.n_rule_violations == 0, (d.violation_t_range, d.max_violation_V)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        (e,) = [x for x in p.device_loss_summary(b, r, diode_specs={"D1": {"Q_rr": 100e-9}})
                if x["name"] == "D1"]
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert e["n_turn_off_events"] == 200
