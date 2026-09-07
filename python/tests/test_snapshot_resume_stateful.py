"""Stateful devices survive `resume_from=` — an engine defect.

`SolverSnapshot` carried only the linear trapezoidal companion
history. The five stateful-device histories restarted from their init
value while `x` still described the old state, and the first Newton
step reconciled the two by jumping. Nothing was raised. Measured, a
two-segment resume against one continuous run on the fixed engine:

    saturable inductor      i(Ls) 171.85 A -> 27.14 A (8 mWb of flux lost)
    gapped-core inductor    99.93 A -> 0.018 A on the first resumed step
    charge-based Coss       109.85 V -> 0.28 V on the first resumed step
    Lauritzen diode, 1 MHz  reverse peak -2.21 A -> -1.15 A
    IGBT turn-off tail      17.72 A -> 0 on the first resumed step
    MNA PMSM                omega 3.87 vs -1.06 rad/s

Three more silent drops sat around it: the TR-BDF2 engine left
`final_snapshot` empty with `valid=False` (a fixed-engine resume from
it started at zero), the DSED engine and the closed-loop chain fast
path ignored the snapshot entirely, and an invalid snapshot resumed
from zero rather than being refused.
"""

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest

import pulsim as p


# --------------------------------------------------------------------------
# One circuit per stateful device kind.
# --------------------------------------------------------------------------

def _sat():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", v_dc=0.0, v_amplitude=50.0,
                              frequency=1e3, phase=0.0)
    b.add_saturable_inductor("Ls", "a", "gnd", 1e-3, 5.0, 5e-5)
    return b


def _gapped():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", v_dc=0.0, v_amplitude=20.0,
                              frequency=1e3, phase=0.0)
    b.add_resistor("R", "a", "m", 0.2)
    b.add_gapped_core_inductor("Lc", "m", "gnd", N=25, Ae=76e-6, le=72e-3,
                               lg=0.5e-3)
    return b


def _laur():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V1", "a", "gnd", v_dc=0.0, v_amplitude=5.0,
                              frequency=1e6, phase=0.0)
    b.add_resistor("R1", "a", "k", 1.0)
    b.add_lauritzen_diode("D", "k", "gnd", tau=1e-7, T_M=1e-8)
    return b


def _coss():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", v_dc=60.0, v_amplitude=50.0,
                              frequency=100e3, phase=0.0)
    b.add_resistor("R", "a", "c", 100.0)
    b.add_nonlinear_capacitor("Coss", "c", "gnd", 2e-9, 25.0, 0.5)
    return b


def _igbt():
    """A clamped inductive turn-off, split MID-TAIL. The load inductor
    carries i0=100 A, so this circuit also pins the ordering: builder
    ICs must NOT re-seed the tail charge after the snapshot restored
    it (that lost 17.7 A of tail current)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc", "gnd", 600.0)
    b.add_pulse_voltage_source("Vg", "g", "gnd", 0.0, 15.0, 0.0, 2e-6, 0.0,
                               1e-8, 1e-8)
    b.add_inductor("Lload", "dc", "c", 500e-6, i0=100.0)
    b.add_diode("Dfw", "c", "dc", 1e3, 1e-9, 0.7)
    b.add_igbt_level1("Q1", "c", "e", "g", 1.5, 0.05, 5.0, tau_tail=1e-6,
                      k_tail=0.30)
    b.add_resistor("Rsense", "e", "gnd", 1e-4)
    b.add_capacitor("Cs", "c", "gnd", 2e-9)
    return b


def _pmsm():
    b = p.CircuitBuilder()
    for k, node in enumerate(("ua", "ub", "uc")):
        b.add_sine_voltage_source(f"Vs_{'abc'[k]}", node, "gnd", v_dc=0.0,
                                  v_amplitude=12.0, frequency=30.0,
                                  phase=-2.0 * math.pi / 3.0 * k)
    b.add_pmsm_mna("M1", "ua", "ub", "uc", "nn", "w", "th", R_s=0.5,
                   L_d=1e-3, L_q=3e-3, psi_pm=0.05, pole_pairs=4, J=1e-3,
                   B=1e-4)
    return b


# name, builder, t_split, t_end, dt, snapshot field, reals per device
CASES = [
    ("saturable", _sat, 0.25e-3, 2e-3, 2e-7, "saturable_history", 7),
    ("gapped", _gapped, 0.25e-3, 2e-3, 2e-7, "saturable_history", 7),
    ("lauritzen", _laur, 0.25e-6, 2e-6, 1e-9, "lauritzen_history", 2),
    ("coss", _coss, 2.5e-6, 20e-6, 1e-9, "coss_history", 3),
    ("igbt_tail", _igbt, 2.5e-6, 6e-6, 1e-8, "igbt_tail_history", 2),
    ("pmsm", _pmsm, 0.01, 0.02, 1e-6, "pmsm_history", 9),
]


def _kw(make):
    n = make().graph.num_switches
    return dict(switch_fn=lambda _t: p.SwitchStateMask(n)) if n else {}


def _worst_relative(full, second, dt):
    tf = np.asarray(full.times)
    xf = np.asarray(full.states)[tf >= second.times[0] - 0.5 * dt]
    xs = np.asarray(second.states)
    n = min(len(xf), len(xs))
    assert n > 1
    err = float(np.abs(xf[:n] - xs[:n]).max())
    scale = float(np.abs(xf[:n]).max())
    return err / max(scale, 1e-30)


@pytest.mark.parametrize("name,make,t_split,t_end,dt,field,width", CASES,
                         ids=[c[0] for c in CASES])
def test_two_segment_resume_equals_continuous_run(name, make, t_split, t_end,
                                                  dt, field, width):
    kw = _kw(make)
    full = p.simulate(make(), t_end=t_end, dt=dt, engine="pwl", **kw)
    first = p.simulate(make(), t_end=t_split, dt=dt, engine="pwl", **kw)
    snap = first.final_snapshot
    assert len(getattr(snap, field)) == width, (name, field)
    second = p.simulate(make(), t_end=t_end, dt=dt, engine="pwl",
                        resume_from=snap, **kw)
    # The resumed run recomputes t = t_start + k*dt from a non-zero
    # t_start, so a sine source's argument differs in its last bits
    # (the same effect test_snapshot_resume.py documents for the
    # linear case) — machine precision, not equality.
    rel = _worst_relative(full, second, dt)
    assert rel <= 1e-9, (name, rel)


def test_the_flux_a_resume_used_to_lose_is_carried():
    """The headline number, on the device that showed it worst."""
    kw = _kw(_sat)
    full = p.simulate(_sat(), t_end=1.5e-3, dt=2e-7, engine="pwl", **kw)
    first = p.simulate(_sat(), t_end=0.25e-3, dt=2e-7, engine="pwl", **kw)
    second = p.simulate(_sat(), t_end=1.5e-3, dt=2e-7, engine="pwl",
                        resume_from=first.final_snapshot, **kw)
    i_cont = float(np.asarray(full.i("Ls"))[-1])
    i_split = float(np.asarray(second.i("Ls"))[-1])
    assert abs(i_cont) > 100.0                       # deep in saturation
    assert i_split == pytest.approx(i_cont, rel=1e-9), (i_cont, i_split)


def test_trbdf2_fills_a_snapshot_and_resumes_from_one():
    kw = _kw(_sat)
    first = p.simulate(_sat(), t_end=0.5e-3, engine="trbdf2", rtol=1e-8,
                       atol=1e-11, **kw)
    snap = first.final_snapshot
    assert snap.valid                                # was EMPTY, valid=False
    assert len(snap.saturable_history) == 7
    full = p.simulate(_sat(), t_end=1.2e-3, engine="trbdf2", rtol=1e-8,
                      atol=1e-11, **kw)
    second = p.simulate(_sat(), t_end=1.2e-3, engine="trbdf2", rtol=1e-8,
                        atol=1e-11, resume_from=snap, **kw)
    # Not bit-exact: the step controller's state (last accepted h, the
    # LTE history) is not in the snapshot, so the resumed segment
    # restarts from h_init and lands on a different time grid.
    i_cont = float(np.asarray(full.i("Ls"))[-1])
    i_split = float(np.asarray(second.i("Ls"))[-1])
    assert i_split == pytest.approx(i_cont, rel=1e-4), (i_cont, i_split)


def test_a_trbdf2_snapshot_resumes_on_the_fixed_engine():
    kw = _kw(_sat)
    var = p.simulate(_sat(), t_end=0.25e-3, engine="trbdf2", rtol=1e-10,
                     atol=1e-13, **kw)
    fixed = p.simulate(_sat(), t_end=1.5e-3, dt=2e-7, engine="pwl",
                       resume_from=var.final_snapshot, **kw)
    full = p.simulate(_sat(), t_end=1.5e-3, dt=2e-7, engine="pwl", **kw)
    assert float(np.asarray(fixed.i("Ls"))[-1]) == pytest.approx(
        float(np.asarray(full.i("Ls"))[-1]), rel=1e-5)


def test_an_old_layout_snapshot_is_refused_by_name_not_resumed_from_zero():
    """A snapshot from a Pulsim that did not carry saturable state has
    the field empty. That used to `return` silently from from_flat and
    resume with lambda_old = 0 while x still carried the current — the
    8 mWb seam. Now it is refused, and the message names the device
    kind and every builder spelling that shares the history."""
    first = p.simulate(_sat(), t_end=1e-4, dt=2e-7, engine="pwl")
    snap = first.final_snapshot
    assert len(snap.saturable_history) == 7
    snap.saturable_history = []                      # the old layout
    with pytest.raises(Exception, match="SaturableInductorHistory"):
        p.simulate(_sat(), t_end=2e-4, dt=2e-7, engine="pwl",
                   resume_from=snap)
    snap.saturable_history = [0.0] * 5               # the 5-real layout
    with pytest.raises(Exception, match="expected 7 values"):
        p.simulate(_sat(), t_end=2e-4, dt=2e-7, engine="pwl",
                   resume_from=snap)


def test_a_snapshot_from_a_different_circuit_is_refused():
    small = p.simulate(_sat(), t_end=1e-5, dt=1e-7, engine="pwl")
    two = _sat()
    two.add_saturable_inductor("L2", "a", "gnd", 1e-3, 5.0, 5e-5)
    with pytest.raises(Exception, match="different circuit"):
        p.simulate(two, t_end=2e-5, dt=1e-7, engine="pwl",
                   resume_from=small.final_snapshot)


def test_an_invalid_snapshot_is_refused_not_ignored():
    from pulsim._pulsim import SolverSnapshot
    with pytest.raises(ValueError, match="not valid"):
        p.simulate(_sat(), t_end=1e-3, dt=2e-7, engine="pwl",
                   resume_from=SolverSnapshot())


def test_dsed_refuses_resume_by_name_instead_of_dropping_it():
    first = p.simulate(_sat(), t_end=1e-3, dt=2e-7, engine="pwl")
    with pytest.raises(ValueError, match="resume_from"):
        p.simulate(_sat(), t_end=2e-3, engine="dsed",
                   resume_from=first.final_snapshot)


def test_a_snapshot_pickles_and_the_copy_resumes_identically():
    kw = _kw(_sat)
    first = p.simulate(_sat(), t_end=0.25e-3, dt=2e-7, engine="pwl", **kw)
    snap = first.final_snapshot
    back = pickle.loads(pickle.dumps(snap))
    assert back.valid and back.saturable_history == snap.saturable_history
    a = p.simulate(_sat(), t_end=1e-3, dt=2e-7, engine="pwl",
                   resume_from=snap, **kw)
    b = p.simulate(_sat(), t_end=1e-3, dt=2e-7, engine="pwl",
                   resume_from=back, **kw)
    assert float(np.asarray(a.i("Ls"))[-1]) == float(np.asarray(b.i("Ls"))[-1])


def test_steady_state_refuses_a_circuit_whose_map_is_not_affine():
    """Its one-period map is affine only for linear companions. With a
    saturable inductor it returned a point carrying +0.38 A of DC
    inductor current in its first period — a transient, as an orbit."""
    from pulsim.steady_state import steady_state
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", v_dc=0.0, v_amplitude=20.0,
                              frequency=1e3, phase=0.0)
    b.add_resistor("Rs", "a", "m", 0.5)
    b.add_saturable_inductor("Ls", "m", "o", 1e-3, 5.0, 5e-5)
    b.add_capacitor("C", "o", "gnd", 20e-6)
    b.add_resistor("Rl", "o", "gnd", 5.0)
    with pytest.raises(ValueError, match="saturable"):
        steady_state(b, period=1e-3, dt=2e-7)
