"""F-Z-14: a branch-form diode's commutation is booked where the conductance
form books it. Before, a branch-form flip was booked at the step end: the
start-up turn-on at t = dt (the conductance form: no event), and on a 1e-4 grid
the sine's zeros at 100 / 200 / 400 dt as 101 / 201 / 401 dt; the states of the
two forms agree to 3.6e-15 V. A flip at a gate edge keeps its own instant (the
edge under the restart rule, the step end without it)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p

A, F, R, T_END, SNAP = 10.0, 50.0, 10.0, 0.1, 1e-6
RULES = [(False, False), (True, False), (True, True)]   # (restart rule, split)


def _rect(dt, form, rule, split):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, A, F)
    b.add_diode("D", "in", "k", 1e6, 1e-12, form=form)
    b.add_resistor("R", "k", "gnd", R)
    n = b.graph.num_switches
    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build_lazy(dt)
    o = p.SimulationOptions(t_start=0.0, t_end=T_END, dt=dt)
    o.discontinuity_restart = rule
    o.enable_substep_state_correction = split
    o.split_snap = SNAP
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.run_transient(cache, b.graph, b.pool, o, lambda t: p.SwitchStateMask(n))


def _events(res):
    return [(e.t_estimated, int(e.branch_id), bool(e.new_state)) for e in res.commutation_events]


@pytest.mark.parametrize("rule,split", RULES)
@pytest.mark.parametrize("dt", [3.7e-5, 1e-4], ids=["off-grid", "on-grid"])
def test_both_forms_book_the_same_events(dt, rule, split):
    eb = _events(_rect(dt, "branch", rule, split))
    ec = _events(_rect(dt, "conductance", rule, split))
    assert [e[1:] for e in eb] == [e[1:] for e in ec]
    # measured: 0 (no split), 6.9e-15 s with the split; the gate is the locator's
    assert max(abs(a[0] - c[0]) for a, c in zip(eb, ec)) <= SNAP * dt


@pytest.mark.parametrize("dt", [3.7e-5, 1e-4], ids=["off-grid", "on-grid"])
def test_the_branch_form_books_each_zero_within_the_stated_tolerance(dt):
    te = np.array([e[0] for e in _events(_rect(dt, "branch", True, True))])
    zeros = np.arange(0.0, T_END + 1 / F, 0.5 / F)[1:]   # t = 0 is no event
    err = np.array([np.min(np.abs(zeros - t)) for t in te]) / dt
    assert len(err) >= 8
    assert err.max() <= 2 * SNAP, err


def _buck(form, rule, split):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_ideal_switch("S", "vin", "sw", 0.0, 0.0)
    b.add_diode("D", "gnd", "sw", 1e3, 1e-9, form=form)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6, 0.0)
    b.add_resistor("RL", "vout", "gnd", 5.0)
    k = b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(b.graph.num_switches)
        m.set(k, (t / 1e-5) % 1.0 < 0.3 - 1e-9)
        return m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=4e-4, dt=1e-7, engine="pwl", switch_fn=fn,
                          discontinuity_restart=rule, enable_substep_state_correction=split)


@pytest.mark.parametrize("rule,split", RULES)
def test_a_natural_turn_off_in_a_dcm_buck_is_booked_as_the_conductance_form(rule, split):
    eb, ec = _events(_buck("branch", rule, split)), _events(_buck("conductance", rule, split))
    assert [e[1:] for e in eb] == [e[1:] for e in ec]
    T, dt = 1e-5, 1e-7
    ph = np.array([e[0] for e in ec]) / T % 1.0
    natural = (np.abs(ph - 0.3) > 2 * dt / T) & (np.minimum(ph, 1 - ph) > 2 * dt / T)
    assert natural.sum() >= 5                        # the DCM turn-offs (measured 11)
    tb, tc = np.array([e[0] for e in eb]), np.array([e[0] for e in ec])
    assert np.abs(tb - tc)[natural].max() <= SNAP * dt
