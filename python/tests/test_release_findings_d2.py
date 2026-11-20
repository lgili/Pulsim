"""Release findings, group D2: TR-BDF2's SDIRK restart with stateful kinds
(openspec/changes/update-number-changing-release/fixes/group-d2.md)."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import pulsim as p

sys.path.insert(0, str(Path(__file__).parent))


def _flyback_landings(form_s, form_d, t_end):
    from test_release_findings_de import _flyback_branch
    b = _flyback_branch(form_s, form_d)
    q, n, period = b.switch_index_of("Q1"), b.graph.num_switches, 1e-5

    def sf(t):
        m = p.SwitchStateMask(n)
        m.set(q, (t % period) / period < 0.4)
        return m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=t_end, switch_fn=sf, engine="trbdf2", rtol=1e-6, atol=1e-9)
    t = np.asarray(r.times)
    mins = []
    for e in r.commutation_events:
        k = np.searchsorted(t, e.t_estimated + 1e-15)   # past the post-event sample
        d = np.diff(t[k:k + 40])
        mins.append(d[d > 0][1:].min())
    return np.asarray(mins)


@pytest.mark.parametrize("forms", [("conductance", "conductance"), ("branch", "conductance")])
def test_no_step_collapse_after_a_diode_landing_with_a_stateful_kind(forms):
    """F-3-4's cause: with a stateful kind (the saturable transformer) TR-BDF2
    took no SDIRK restart after a diode landing, so the next step read the
    pre-event derivative slot of the leakage inductor and was rejected down
    to 2e-13 .. 6e-14 s (26 of 99 landings below 1e-12 s on the conductance
    pair; branch/conductance aborted at the 46th). With the restart the
    smallest step after any landing is 1.7e-10 s."""
    mins = _flyback_landings(*forms, 4.6e-4)
    assert len(mins) > 40
    assert mins.min() > 1e-11


def test_dependent_states_next_to_a_stateful_kind_run_on_trbdf2():
    """The old `dependent_states` refusal (an ideal switch opening an
    inductor cut-set while ANY stateful kind is in the circuit, here an
    unrelated Lauritzen diode) runs: flux conservation across the jump,
    closed form i(t) = i+ exp(-R (t - t0) / (L + L2))."""
    L, L2, R, t0, V = 1e-3, 2e-3, 1.0, 2e-4, 10.0
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", V)
    b.add_ideal_switch("S", "vin", "a", 0.0, 0.0)
    b.add_resistor("R", "a", "m", R)
    b.add_inductor("Ls", "m", "gnd", L)
    b.add_inductor("L2", "a", "gnd", L2)
    b.add_voltage_source("V2", "v2", "gnd", 1.0)
    b.add_resistor("R2", "v2", "d", 1e3)
    b.add_lauritzen_diode("DL", "d", "gnd")

    def fn(t):
        m = p.SwitchStateMask(1)
        m.set(0, t < t0)
        return m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, 3e-4, engine="trbdf2", switch_fn=fn)
    t, i = np.asarray(r.times), np.asarray(r.i("Ls"))
    is0, i20 = V / R * (1 - np.exp(-R * t0 / L)), V * t0 / L2
    ip = (L * is0 - L2 * i20) / (L2 + L)
    m = t > t0 + 1e-9
    assert len(r.reinit_records) == 1
    assert np.abs(i[m] - ip * np.exp(-R * (t[m] - t0) / (L2 + L))).max() < 1e-7


@pytest.mark.parametrize("integrator", ["trbdf2", "esdirk43"])
def test_a_current_driven_nonlinear_capacitor_starts(integrator):
    """F-S-1: esdirk43 aborted at t = 0 (Newton at h_min) with a nonlinear
    capacitor (m = 0: a LINEAR charge law) fed by 10 A: the kinds' derivative
    slots start at zero, its explicit first stage read q' = 0 where the
    capacitor carries 10 A, and the q-row estimate was O(h) at any h. The
    run's first step is now an SDIRK restart step (no slot read). Closed
    form v = -I·R·(1 - exp(-t/(R·C))), tau = 1 ps."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 0.0)
    b.add_resistor("R", "a", "sw", 1e-3)
    b.add_nonlinear_capacitor("C", "sw", "gnd", 1e-9, 25.0, 0.0)
    b.add_current_source("I", "gnd", "sw", 10.0)
    kw = {"integrator": integrator} if integrator == "esdirk43" else {}
    r = p.simulate(b, t_end=3e-7, engine="trbdf2", **kw)
    t, v = np.asarray(r.times), np.asarray(r.v("sw"))
    assert v[-1] == pytest.approx(-0.01, rel=1e-6)
    m = t > 5e-11   # past the unresolved first picoseconds (the restart step is 3 tau)
    assert np.abs(v[m] + 0.01 * (1 - np.exp(-t[m] / 1e-12))).max() < 1e-8
