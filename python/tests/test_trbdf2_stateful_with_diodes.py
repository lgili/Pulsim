"""Stateful devices next to a diode, on the variable-step engine.

TR-BDF2 settles the initial diode bits with a zero-time probe solve
before its first step, and re-probes at every diode event and gate
landing. Those probes go through `trap_solve`, which handed the
LINEAR part its step size and left the five stateful devices reading
`(dev_h, coss_h)` — assigned only inside the step loop, so 0 in the
settle and STALE in the probes.

A saturable inductor or a PMSM divides by that step in its stamp. In
any circuit with a diode — a flyback, a forward, an LLC, the
rectifier this test builds — the settle divided by zero and the
process died with SIGSEGV (measured, exit 139). A Coss produced NaN
diode bits instead, silently, because the settle discards the
solve's result. The refusal that used to keep saturable inductors off
this engine had been hiding it; PR #132 lifted the refusal and this
is what was underneath.

`trap_solve` now sets the pair itself, at the one choke point every
trapezoidal solve passes through — the same shape as the fixed
engine's `refresh_dt` fix.

WHAT THESE TESTS DELIBERATELY AVOID. The variable-step engine has a
separate, pre-existing defect at a switched diode's TURN-OFF when an
inductor is in series with it: a plain LINEAR L–D–RC rectifier lands
at half the fixed engine's output with a 1.5 kV spike on the internal
node, and a flux device in the same place makes the diode chatter
hundreds of thousands of times. That is tracked by the strict xfail
at the bottom of this file. The regression tests above it therefore
stop before the first turn-off (0.4 ms of a 1 kHz drive): the settle
and the turn-on are what the fix here is about, and they are what
crashed.
"""

import math

import numpy as np
import pytest

import pulsim as p


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-30)


def _rectifier(kind):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", v_dc=0.0, v_amplitude=20.0,
                              frequency=1e3, phase=0.0)
    if kind == "gapped":
        b.add_gapped_core_inductor("Lc", "a", "m", N=25, Ae=76e-6, le=72e-3,
                                   lg=0.5e-3, B_sat=0.35)
    elif kind == "atan":
        b.add_saturable_inductor("Lc", "a", "m", 111e-6, 6.0, 1e-6)
    else:
        b.add_inductor("Lc", "a", "m", 111e-6)
    b.add_diode("D", "m", "o", 1e3, 1e-9, 0.7)
    b.add_resistor("R", "o", "gnd", 2.0)
    b.add_capacitor("C", "o", "gnd", 10e-6)
    return b


def _run(kind, engine, t_end):
    kw = (dict(dt=2e-7) if engine == "pwl"
          else dict(dt=4e-6, rtol=1e-6, atol=1e-9))
    res = p.simulate(_rectifier(kind), t_end=t_end, engine=engine, **kw)
    t = np.asarray(res.times)
    v_o = np.asarray(res.v("o"))
    i = np.asarray(res.i("Lc"))
    assert np.all(np.isfinite(v_o)) and np.all(np.isfinite(i))
    return t, v_o, i


@pytest.mark.parametrize("kind", ["atan", "gapped"])
def test_flux_device_beside_a_diode_settles_and_turns_on(kind):
    """Used to be exit 139. Through the settle and the first turn-on,
    up to but not including the first turn-off."""
    t_ref, v_ref, i_ref = _run(kind, "pwl", 0.4e-3)
    t_var, v_var, i_var = _run(kind, "trbdf2", 0.4e-3)
    assert i_ref.max() > 3.0                     # conducting hard
    if kind == "gapped":
        assert i_ref.max() > 8.0                 # and past the knee
    # Compare on the fixed grid: interpolate the variable trace.
    v_i = np.interp(t_ref, t_var, v_var)
    i_i = np.interp(t_ref, t_var, i_var)
    m = t_ref > 0.1e-3
    assert np.abs(v_i[m] - v_ref[m]).max() < 2e-2 * np.abs(v_ref).max()
    assert np.abs(i_i[m] - i_ref[m]).max() < 2e-2 * np.abs(i_ref).max()


def _pmsm_with_diode(engine):
    b = p.CircuitBuilder()
    for k, node in enumerate(("ua", "ub", "uc")):
        b.add_sine_voltage_source(f"Vs_{'abc'[k]}", node, "gnd",
                                  v_dc=0.0, v_amplitude=12.0,
                                  frequency=30.0,
                                  phase=-2.0 * math.pi / 3.0 * k)
    b.add_pmsm_mna("M1", "ua", "ub", "uc", "nn", "w", "th",
                   R_s=0.5, L_d=1e-3, L_q=3e-3, psi_pm=0.05,
                   pole_pairs=4, J=1e-3, B=1e-4)
    # An unrelated rectifier with NO series inductance, so the engine
    # has a diode to settle without the turn-off defect in play.
    b.add_sine_voltage_source("Vr", "r", "gnd", v_dc=0.0, v_amplitude=5.0,
                              frequency=1e3, phase=0.0)
    b.add_diode("Dr", "r", "rc", 1e3, 1e-9, 0.7)
    b.add_resistor("Rr", "rc", "gnd", 10.0)
    kw = (dict(dt=2e-7) if engine == "pwl"
          else dict(dt=4e-6, rtol=1e-6, atol=1e-9))
    res = p.simulate(b, t_end=0.02, engine=engine, **kw)
    w = np.asarray(res.v("w"))
    assert np.all(np.isfinite(w))
    return float(w[-1])


def test_pmsm_beside_a_diode_runs_on_trbdf2():
    """Same division, same crash; the PMSM stamp reads dev_h too."""
    ref = _pmsm_with_diode("pwl")
    got = _pmsm_with_diode("trbdf2")
    assert abs(ref) > 5.0
    # 3.7e-3 measured: the rectifier forces event probes the plain
    # PMSM run never sees, and each costs the adaptive engine a step.
    assert _rel(got, ref) < 8e-3, (got, ref)


def _dc(t, y, t0):
    """TIME-WEIGHTED average over [t0, t_end].

    Not `y[t > t0].mean()`. TR-BDF2's grid is adaptive and clusters
    hard around events — after a diode turn-off it takes tens of
    femtosecond-sized steps — so an unweighted mean over samples
    counts those instants as heavily as a microsecond of conduction.
    That metric, not the engine, is what said this rectifier "lands
    at half the fixed engine's output":

        kind     engine   sample mean   time-weighted
        linear   pwl           6.3295          6.3308
        linear   trbdf2        3.0408          6.3482
        atan     trbdf2        3.3101          6.3617
        gapped   trbdf2        9.7514          6.3494

    Sampled at fixed instants the two engines agree to four digits
    the whole way through (15.4227 vs 15.4219 at 0.2 ms, 3.1506 vs
    3.1506 at 0.53 ms, 0.0986 vs 0.0985 at 0.6 ms).
    """
    m = t >= t0
    tt, yy = t[m], y[m]
    return float(np.trapezoid(yy, tt) / (tt[-1] - tt[0]))


@pytest.mark.parametrize("kind", ["linear", "atan", "gapped"])
def test_switched_diode_turn_off_with_series_inductor_on_trbdf2(kind):
    """The engines agree on this rectifier. They always did — the
    metric was wrong. Also pinned: the flux devices no longer chatter
    (this once reported 839,262 diode events for `atan` and 93,998 for
    `gapped`; it is 6 for all three now, which is 3 cycles x on+off)."""
    t_ref, v_ref, _ = _run(kind, "pwl", 3e-3)
    t_var, v_var, _ = _run(kind, "trbdf2", 3e-3)
    dc_ref = _dc(t_ref, v_ref, 2e-3)
    dc_var = _dc(t_var, v_var, 2e-3)
    assert dc_ref > 5.0
    assert _rel(dc_var, dc_ref) < 2e-2, (dc_var, dc_ref)


@pytest.mark.xfail(strict=True,
                   reason="the femtosecond landing step after a diode "
                          "turn-off: the event is localised correctly (the "
                          "inductor current is 2 uA there) but the next step "
                          "is 3e-14 s, and the companion conductance 2L/h "
                          "turns those microamps into -1539 V on the internal "
                          "node of a 20 V circuit. It decays over the ~39 "
                          "femtosecond steps that follow, covering 4.4e-12 s "
                          "of physical time in total, and the OUTPUT is "
                          "untouched — but a reported kilovolt poisons any "
                          "max()-based check (overvoltage margin, insulation). "
                          "The fix is in the step controller: after landing an "
                          "event, resume from a physical step size rather than "
                          "the probe floor.")
def test_no_kilovolt_artefact_on_the_internal_node():
    _assert_internal_node_is_sane("linear")


@pytest.mark.parametrize("kind", ["atan", "gapped"])
def test_a_flux_device_in_the_same_place_has_no_artefact(kind):
    """The saturable and gapped cores do NOT spike (20.1 V and 20.6 V
    peak): their L collapses as the current leaves, so 2L/h at the
    landing step is small. Only the constant-L inductor turns the
    residual microamps into kilovolts. Pinned so the fix for the
    linear case is not written in a way that regresses these."""
    _assert_internal_node_is_sane(kind)


def _assert_internal_node_is_sane(kind):
    res = p.simulate(_rectifier(kind), t_end=3e-3, dt=4e-6, rtol=1e-6,
                     atol=1e-9, engine="trbdf2")
    v_m = np.abs(np.asarray(res.v("m")))
    # Nothing in this circuit can exceed the 20 V source by much.
    assert float(v_m.max()) < 40.0, float(v_m.max())
