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


@pytest.mark.xfail(strict=True,
                   reason="TR-BDF2 switched-diode turn-off with a series "
                          "inductor: a plain L-D-RC rectifier lands at half "
                          "the fixed engine's output with a kV spike on the "
                          "internal node; a flux device in the same place "
                          "chatters. Pre-existing; tracked separately.")
@pytest.mark.parametrize("kind", ["linear", "gapped"])
def test_switched_diode_turn_off_with_series_inductor_on_trbdf2(kind):
    t_ref, v_ref, _ = _run(kind, "pwl", 3e-3)
    t_var, v_var, _ = _run(kind, "trbdf2", 3e-3)
    dc_ref = float(v_ref[t_ref > 2e-3].mean())
    dc_var = float(v_var[t_var > 2e-3].mean())
    assert dc_ref > 5.0
    assert _rel(dc_var, dc_ref) < 2e-2, (dc_var, dc_ref)
