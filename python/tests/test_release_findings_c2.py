"""Release group C2: the mode of an edge under the restart rule, and the jumps the rule
still misses (openspec/changes/update-number-changing-release/fixes/group-c2.md)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p


def _igbt_gate(t_off, fall, **kw):
    """F-5-1's IGBT (100 A in 1 mH) with its free-wheeling diode, which cannot carry this
    current's direction: the gate falls 15 -> 0 V at t_off in `fall` (0: a source jump)."""
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("Vg", "g", "gnd", 15.0, 0.0, t_start=t_off, pulse_width=1.0,
                               period=10.0, rise_time=fall, fall_time=0.0)
    b.add_inductor("L", "sw", "gnd", 1e-3, i0=-100.0)
    b.add_igbt_level1("QL", "sw", "gnd", "g", 1.5, 0.05, 5.0, with_fwd=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=12e-9, dt=1e-9, engine="pwl", discontinuity_restart=True,
                          **kw)


def _assert_cut_recorded(r, t_rec, tol=1e-15):
    """Truth: no path for i_L < 0, so i_L -> 0 by an impulse of 1/2 L i^2 = 5 J."""
    t, i = np.asarray(r.times), np.asarray(r.i("L"))
    recs = list(r.reinit_records)
    assert len(recs) == 1 and list(recs[0].states) == ["L"], recs
    assert recs[0].t == pytest.approx(t_rec, abs=tol)
    assert recs[0].dW == pytest.approx(5.0, rel=1e-5)
    assert np.max(np.abs(i[t > t_rec + 1e-9])) < 1e-2, i


@pytest.mark.parametrize("t_off", [5.6e-9, 5.95e-9])
def test_a_gate_jump_inside_a_restart_step_is_landed_by_the_release_profile(monkeypatch, t_off):
    """A source jump in (t_from + gamma h, t] of a restart step: the release resolves the
    split ON with the rule (`_plan.DEFAULT_SUBSTEP_STATE_CORRECTION`), so the jump is a
    landing (a split event at t_off) and the restart starts AT it: recorded at t_off."""
    monkeypatch.setattr(p._plan, "DEFAULT_SUBSTEP_STATE_CORRECTION", True)
    _assert_cut_recorded(_igbt_gate(t_off, 0.0), t_off)


@pytest.mark.xfail(strict=True, reason=(
    "release group C2, task 3 (explicit split off): without the split a source jump in "
    "(t_from + gamma h, t] of a restart step is not a landing; the step is solved from t_from, "
    "the probe's moved reading stops at the jump's left limit and the frozen one never "
    "reaches it: no record, i_L -> +0.478 A held by the FWD (truth 0, 5 J). The release "
    "turns the split on with the rule (test above); this configuration opts out of landings"))
@pytest.mark.parametrize("t_off", [5.6e-9, 5.95e-9])
def test_a_gate_jump_inside_a_restart_step_without_the_split(t_off):
    _assert_cut_recorded(_igbt_gate(t_off, 0.0, enable_substep_state_correction=False), t_off)


@pytest.mark.xfail(strict=True, reason=(
    "release group C2, task 3 (open): a CONTINUOUS 1 ps gate fall in (t1, t] of a restart "
    "step (here window step 2 of the zero start) turns the IGBT level 1 off inside the step; "
    "a smooth Newton device's region change is not an event of the fixed engine, and the ramp's "
    "corners are not landings (only zero-rise jumps are split events): no record, "
    "i_L -> +0.478 A (truth 0, 5 J). Needs the source ramps' corners as split landings "
    "(a breakpoint class kept out of the restart trigger; moves every ramped source of the "
    "reference set) or a state event on the device's region change"))
@pytest.mark.parametrize("split", [False, True])
def test_a_continuous_gate_fall_inside_a_restart_step_is_an_impulse(split):
    _assert_cut_recorded(_igbt_gate(1.5e-9, 1e-12, enable_substep_state_correction=split),
                         1.5e-9, tol=1e-9)
