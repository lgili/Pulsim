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
"""

import numpy as np
import pytest

import pulsim as p


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-30)


def _sat_with_diode(engine):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 12.0)
    b.add_resistor("R", "a", "m", 1.0)
    b.add_gapped_core_inductor("Lc", "m", "d", N=25, Ae=76e-6, le=72e-3,
                               lg=0.5e-3, B_sat=0.35)
    b.add_diode("D", "d", "gnd", 1e3, 1e-9, 0.7)
    kw = (dict(dt=1e-6) if engine == "pwl"
          else dict(dt=1e-5, rtol=1e-6, atol=1e-9))
    res = p.simulate(b, t_end=2e-3, engine=engine, **kw)
    i = np.asarray(res.i("Lc"))
    assert np.all(np.isfinite(i))
    return float(i[-1]), float(np.abs(i).max())


def test_saturable_inductor_beside_a_diode_runs_on_trbdf2():
    """Used to be exit 139."""
    end_ref, peak_ref = _sat_with_diode("pwl")
    end_var, peak_var = _sat_with_diode("trbdf2")
    assert peak_ref > 6.0                      # past the core's knee
    assert _rel(end_var, end_ref) < 1e-3, (end_var, end_ref)
    assert _rel(peak_var, peak_ref) < 5e-3, (peak_var, peak_ref)


def _pmsm_with_diode(engine):
    import math
    b = p.CircuitBuilder()
    for k, node in enumerate(("ua", "ub", "uc")):
        b.add_sine_voltage_source(f"Vs_{'abc'[k]}", node, "gnd",
                                  v_dc=0.0, v_amplitude=12.0,
                                  frequency=30.0,
                                  phase=-2.0 * math.pi / 3.0 * k)
    b.add_pmsm_mna("M1", "ua", "ub", "uc", "nn", "w", "th",
                   R_s=0.5, L_d=1e-3, L_q=3e-3, psi_pm=0.05,
                   pole_pairs=4, J=1e-3, B=1e-4)
    # An unrelated rectifier, so the engine has a diode to settle.
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


@pytest.mark.parametrize("rtol", [1e-4, 1e-6])
def test_probe_solves_use_the_probe_step_not_a_stale_one(rtol):
    """The settle is not the only exposure: every diode event
    re-probes with a tiny step while `dev_h` still held the last
    STAGE step. With a saturable inductor that stamped 2λ/h against
    a linear part assembled at 2C/h_probe — a mixed-step solve, the
    exact class the saturable refusal used to name. A rectifier with
    the core in its loop crosses a diode event twice a cycle; the
    answer must not depend on how the controller happened to land."""
    def run(engine, **kw):
        b = p.CircuitBuilder()
        b.add_sine_voltage_source("V", "a", "gnd", v_dc=0.0, v_amplitude=20.0,
                                  frequency=1e3, phase=0.0)
        b.add_gapped_core_inductor("Lc", "a", "m", N=25, Ae=76e-6, le=72e-3,
                                   lg=0.5e-3, B_sat=0.35)
        b.add_diode("D", "m", "o", 1e3, 1e-9, 0.7)
        b.add_resistor("R", "o", "gnd", 2.0)
        b.add_capacitor("C", "o", "gnd", 10e-6)
        res = p.simulate(b, t_end=5e-3, engine=engine, **kw)
        t = np.asarray(res.times)
        v = np.asarray(res.v("o"))
        assert np.all(np.isfinite(v))
        # Mean over the last drive cycle, not the endpoint: R·C is
        # 20 µs against a 1 ms period, so v_out is a train of
        # half-sine pulses and the endpoint is a legitimate zero.
        m = t >= 4e-3
        return float(np.trapezoid(v[m], t[m]) / (t[m][-1] - t[m][0]))

    ref = run("pwl", dt=2e-7)
    got = run("trbdf2", dt=4e-6, rtol=rtol, atol=rtol * 1e-3)
    assert ref > 1.0
    assert _rel(got, ref) < 2e-2, (got, ref)
