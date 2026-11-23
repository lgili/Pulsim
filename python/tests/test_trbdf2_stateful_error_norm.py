"""TR-BDF2's error estimate sees the stateful kinds' internal states.

Before update-trbdf2-error-control DECISIONS 10 the estimate had no row for
a stateful kind's charge or flux: a circuit whose dynamics live only there
(a Lauritzen recovery, an IGBT tail, a nonlinear capacitor behind a
resistor) sat at its step ceiling, and tightening rtol changed nothing
(the Lauritzen recovery was 2e-1 off its closed form at every rtol). Each
q row now enters the norm with the scale atol·s + rtol·max(|q_n|, |q_1|),
s the device's own capacitance / inductance / time constant, on both
integrators (one implementation). Truths: the closed forms and SciPy runs
of test_esdirk43_stateful_kinds.py / test_esdirk43_engine.py.
"""
import numpy as np
import pytest

import pulsim as p

from . import test_esdirk43_engine as E
from . import test_esdirk43_stateful_kinds as K


def _run(build, t_end, rtol, **kw):
    return p.simulate(build(), t_end, engine="trbdf2", rtol=rtol, atol=rtol * 1e-3, **kw)


# (build, t_end, extra kwargs, error against the truth, rtol pair, factor, bound).
# Measured (TR-BDF2, atol = rtol*1e-3): Lauritzen 4.85e-5 -> 2.00e-6 at
# rtol 1e-5 -> 1e-7 (a 3e-6 floor below, the source edge's; esdirk43's is
# 3.4e-6), IGBT tail 1.46e-5 -> 1.30e-7 at 1e-5 -> 1e-8, nonlinear capacitor
# 6.79e-7 -> 3.11e-8 at 1e-8 -> 1e-10 (step-ceiling-bound above 1e-8).
# Before: 1.995e-1, 5.23e-3 and 8.62e-7 at every rtol. A second-order
# method under a third-order estimate falls as rtol^(2/3): factors 21 / 100
# over two / three decades, asserted with about 4x and 10x margin; bounds
# carry 5x.
CASES = {
    "lauritzen": (K._laur, K.TD + 5e-6, dict(dt=1e-6), K._laur_err, (1e-5, 1e-7), 5, 1e-5),
    "igbt_tail": (K._igbt, K.TDI + 10e-6, dict(dt=2e-6), K._igbt_err, (1e-5, 1e-8), 10, 7e-7),
    "nonlinear_capacitor": (E._cap_rc, E.TDC + 3e-6, {}, E._cap_err, (1e-8, 1e-10), 5, 1.6e-7),
}


@pytest.mark.filterwarnings("ignore:simulate")
@pytest.mark.parametrize("name", list(CASES))
def test_error_falls_with_rtol(name):
    build, t_end, kw, err, (lo, hi), factor, bound = CASES[name]
    e_lo = err(_run(build, t_end, lo, **kw))
    e_hi = err(_run(build, t_end, hi, **kw))
    assert e_hi < e_lo / factor, (e_lo, e_hi)
    assert e_hi < bound, e_hi


# ---- an inactive stateful device costs no steps ------------------------------
# DECISIONS 17 of add-high-order-stiff-integrator left a doubt: a q row that
# stays (almost) exactly 0 got the smallest possible scale and could force
# rejects. An IGBT whose gate is held at 0 V (its tail charge Q ~ 1e-28 C
# for the whole run) beside a switched RC: the same steps, within a few
# percent, as the tail-less IGBT (a static Newton device, no q row, the
# same nonlinear path).
def _rc_with_idle_igbt(tail):
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "in", "gnd", 0.0, 10.0, 1e-6, 4e-6, 10e-6, 1e-7, 1e-7)
    b.add_resistor("R", "in", "c", 100.0)
    b.add_capacitor("C", "c", "gnd", 1e-7)
    b.add_voltage_source("VB", "vb", "gnd", 50.0)
    b.add_resistor("RL", "vb", "col", 10.0)
    b.add_igbt_level1("Q", "col", "gnd", "gnd", **(dict(tau_tail=2e-6, k_tail=0.3) if tail else {}))
    return b


@pytest.mark.filterwarnings("ignore:simulate")
@pytest.mark.parametrize("integrator", [None, "esdirk43"], ids=["trbdf2", "esdirk43"])
@pytest.mark.parametrize("rtol", [1e-6, 1e-9])
def test_an_idle_stateful_device_takes_no_extra_steps(integrator, rtol):
    kw = dict(engine="trbdf2", rtol=rtol, atol=rtol * 1e-3)
    if integrator:
        kw["integrator"] = integrator
    with_q = p.simulate(_rc_with_idle_igbt(True), 60e-6, **kw)._trbdf2_stats
    without = p.simulate(_rc_with_idle_igbt(False), 60e-6, **kw)._trbdf2_stats
    assert with_q["n_forced_accepts"] == without["n_forced_accepts"]
    assert with_q["n_reject"] <= without["n_reject"] + 2
    assert abs(with_q["n_accept"] - without["n_accept"]) <= 0.05 * without["n_accept"], (
        with_q["n_accept"], without["n_accept"])


# ---- a rejected step leaves no device state behind (TR-BDF2) -----------------
# The q rows read the kinds' stage captures in the error phase. A resumed run
# estimates them from its first step (the snapshot carries real slots), so a
# resume with an oversized first step rejects ON the q rows; it must then match
# the resume started at the step it accepted.
@pytest.mark.filterwarnings("ignore:simulate")
@pytest.mark.parametrize("build,t_split,t_end", [
    (K._laur, K.TD + 1e-6, K.TD + 5e-6), (K._igbt, K.TDI + 2e-6, K.TDI + 10e-6),
    (K._pmsm_free, 2.5e-3, 5e-3)], ids=["lauritzen", "igbt", "pmsm"])
def test_rejected_steps_leave_no_device_state(build, t_split, t_end):
    kw = dict(engine="trbdf2", rtol=1e-8, atol=1e-11)
    snap = p.simulate(build(), t_split, **kw).final_snapshot
    kw["dt"] = (t_end - t_split) / 4   # the ceiling, so dt_init is not capped
    tried = p.simulate(build(), t_end, resume_from=snap, dt_init=kw["dt"], **kw)
    h1 = tried.times[1] - tried.times[0]
    clean = p.simulate(build(), t_end, resume_from=snap, dt_init=h1, **kw)
    assert tried._trbdf2_stats["n_reject"] >= clean._trbdf2_stats["n_reject"] + 2
    # Not bit for bit: h1 = t1 - t_split is rebuilt only to an ulp (t_split
    # != 0) and the controller carries that on. The yardstick is the same
    # clean run with its first step one ulp off: measured 4-5e-10 of scale
    # (PMSM), tried vs clean 1.0e-9. A rejected trial that left a stage
    # capture or K0/K1 behind moves the states by the step's error instead.
    ulp = p.simulate(build(), t_end, resume_from=snap, dt_init=h1 * (1 + 2e-16), **kw)
    xt, xc, xu = (np.asarray(r.states) for r in (tried, clean, ulp))
    assert xt.shape == xc.shape
    noise = np.max(np.abs(xu - xc)) if xu.shape == xc.shape else np.max(np.abs(xc))
    assert np.max(np.abs(xt - xc)) <= 10 * noise + 1e-14 * np.max(np.abs(xc))
