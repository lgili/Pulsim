"""ESDIRK43 on the stateful device kinds it used to refuse by name.

Each kind is held to a truth: a closed form where one exists (Lauritzen
reverse recovery, IGBT tail, PMSM locked rotor, DC machine), or an
independent reference (the pinned SciPy induction-machine run, TR-BDF2 at
rtol 1e-10 for the PMSM free acceleration). The order is measured on a
fixed ceiling (rtol = atol = 1: the estimate never rejects). A rejected
step leaves no device state behind: a run that tried and rejected a too
large first step is bit-identical to the run that started at the step it
accepted. The Jiles-Atherton core stays refused: its in-loop law is first
order in the step on both integrators (add-high-order-stiff-integrator
DECISIONS 17).
"""
import math

import numpy as np
import pytest

import pulsim as p

from . import test_dc_machine as DCM
from . import test_induction_machine_mna as IM


def _order(errs):
    return [math.log2(a / b) for a, b in zip(errs, errs[1:])]


# ---- Lauritzen reverse recovery -------------------------------------------
# Forward 1.7 V through 1 ohm, then -100 V at TD. With the junction voltage
# pinned reverse, q_M obeys dq/dt = -q/T_M - q/tau - I_S: the current is
# i(t) = -(q0/T_M) exp(-A t) - tau I_S / T_M, A = 1/T_M + 1/tau.
TAU, TM, IS, VT, TD = 1e-6, 1e-7, 1e-12, 0.025852, 20e-6


def _laur():
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "in", "gnd", 1.7, -100.0, TD, 1.0, 0.0, 0.0, 0.0)
    b.add_resistor("R", "in", "a", 1.0)
    b.add_lauritzen_diode("D", "a", "gnd", I_S=IS, tau=TAU, T_M=TM)
    return b


def _laur_q0():
    def res(v):
        return (1.7 - v) - (TAU / (TAU + TM) * IS * math.expm1(v / VT) + 1e-12 * v)
    lo, hi = 0.0, 1.7
    for _ in range(200):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if res(m) > 0 else (lo, m)
    v = 0.5 * (lo + hi)
    return (1.7 - v - 1e-12 * v) * TAU


def _laur_err(r):
    t = np.asarray(r.times)
    i = np.asarray(r.v("in")) - np.asarray(r.v("a"))
    q0 = _laur_q0()
    m = t > TD + 1e-12
    tp = t[m] - TD
    cf = -(q0 / TM) * np.exp(-(1 / TM + 1 / TAU) * tp) - TAU * IS / TM
    return np.max(np.abs(i[m] - cf)) / (q0 / TM)


@pytest.mark.filterwarnings("ignore:simulate")
def test_lauritzen_reverse_recovery_matches_the_closed_form():
    r = p.simulate(_laur(), TD + 5e-6, engine="trbdf2", integrator="esdirk43",
                   rtol=1e-8, atol=1e-11, dt=1e-6)
    # measured 3.2e-6 (a floor from the source edge's sliver)
    assert _laur_err(r) < 1e-5


def test_lauritzen_is_fourth_order():
    errs = [_laur_err(p.simulate(_laur(), TD + 3e-6, engine="trbdf2", integrator="esdirk43",
                                 dt=3e-6 / n, rtol=1.0, atol=1.0)) for n in (20, 40, 80)]
    # measured orders 3.97, 4.05 (TR-BDF2 on the same ceilings: 2.1)
    assert min(_order(errs)) > 3.5, errs


# ---- IGBT tail turn-off ---------------------------------------------------
# The gate falls at TDI; the tail current is k I0 exp(-t/tau_tail).
KT, TT, TDI = 0.3, 2e-6, 80e-6


def _igbt():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 100.0)
    b.add_pulse_voltage_source("G", "g", "gnd", 15.0, 0.0, TDI, 1.0, 0.0, 0.0, 0.0)
    b.add_resistor("R", "in", "a", 10.0)
    b.add_igbt_level1("Q", "a", "gnd", "g", tau_tail=TT, k_tail=KT)
    return b


def _igbt_err(r):
    t = np.asarray(r.times)
    i = (100.0 - np.asarray(r.v("a"))) / 10.0
    i0 = i[np.searchsorted(t, TDI) - 1]
    m = t > TDI + 1e-12
    return np.max(np.abs(i[m] - KT * i0 * np.exp(-(t[m] - TDI) / TT))) / i0


@pytest.mark.filterwarnings("ignore:simulate")
def test_igbt_tail_matches_the_closed_form():
    r = p.simulate(_igbt(), TDI + 10e-6, engine="trbdf2", integrator="esdirk43",
                   rtol=1e-8, atol=1e-11, dt=2e-6)
    assert _igbt_err(r) < 5e-7   # measured 1.4e-7 (TR-BDF2 5.2e-3)


def test_igbt_tail_is_fourth_order():
    errs = [_igbt_err(p.simulate(_igbt(), TDI + 6e-6, engine="trbdf2", integrator="esdirk43",
                                 dt=6e-6 / n, rtol=1.0, atol=1.0)) for n in (10, 20, 40)]
    assert min(_order(errs)) > 3.5, errs   # measured 3.88, 3.74


# ---- PMSM -------------------------------------------------------------------
RS, LD, LQ, PSI, PP = 0.5, 1e-3, 3e-3, 0.05, 4


def _pmsm_locked():
    # the rotor is held at theta = 0 (J huge); a d-axis voltage step gives
    # i_a = v/R (1 - exp(-R t / L_d))
    b = p.CircuitBuilder()
    for k, (node, vk) in enumerate(zip(("ua", "ub", "uc"), (6.0, -3.0, -3.0))):
        b.add_pulse_voltage_source(f"V{k}", node, "gnd", 0.0, vk, TD, 1.0, 0.0, 0.0, 0.0)
    b.add_pmsm_mna("M1", "ua", "ub", "uc", "nn", "w", "th", R_s=RS, L_d=LD, L_q=LQ,
                   psi_pm=PSI, pole_pairs=PP, J=1e6, B=0.0)
    return b


def _pmsm_err(r):
    t = np.asarray(r.times)
    cf = 6.0 / RS * (1 - np.exp(-RS * np.maximum(t - TD, 0) / LD))
    return np.max(np.abs(np.asarray(r.i("M1_a")) - cf)) / (6.0 / RS)


def _pmsm_free():
    b = p.CircuitBuilder()
    for k, node in enumerate(("ua", "ub", "uc")):
        b.add_sine_voltage_source(f"Vs{k}", node, "gnd", v_dc=0.0, v_amplitude=12.0,
                                  frequency=30.0, phase=-2 * math.pi / 3 * k)
    b.add_pmsm_mna("M1", "ua", "ub", "uc", "nn", "w", "th", R_s=RS, L_d=2e-3, L_q=3e-3,
                   psi_pm=PSI, pole_pairs=PP, J=1e-3, B=1e-4)
    return b


def test_pmsm_locked_rotor_matches_the_closed_form():
    r = p.simulate(_pmsm_locked(), TD + 5e-3, engine="trbdf2", integrator="esdirk43",
                   rtol=1e-9, atol=1e-12)
    assert _pmsm_err(r) < 1e-8   # measured 8.5e-10 (TR-BDF2 9.3e-8)


def test_pmsm_locked_rotor_is_fourth_order():
    errs = [_pmsm_err(p.simulate(_pmsm_locked(), TD + 4e-3, engine="trbdf2",
                                 integrator="esdirk43", dt=4e-3 / n, rtol=1.0, atol=1.0))
            for n in (10, 20)]
    # measured 4.8e-7, 3.1e-8 (3.96); n = 40 reaches the 4e-9 floor (3.09)
    assert min(_order(errs)) > 3.5, errs


def test_pmsm_free_acceleration_matches_tight_trbdf2():
    ref = p.simulate(_pmsm_free(), 0.1, engine="trbdf2", rtol=1e-10, atol=1e-13)
    r = p.simulate(_pmsm_free(), 0.1, engine="trbdf2", integrator="esdirk43",
                   rtol=1e-8, atol=1e-11)
    w_ref, w = ref.v("w")[-1], r.v("w")[-1]
    # measured 61.687253 vs 61.687307 (rel 9e-7); TR-BDF2 at 1e-10 is
    # itself within ~6e-7 of ESDIRK43 at 1e-10
    assert abs(w - w_ref) / abs(w_ref) < 5e-6
    t = np.asarray(r.times)
    np.testing.assert_allclose(np.interp(np.asarray(ref.times), t, r.i("M1_a")),
                               ref.i("M1_a"), rtol=0, atol=2e-3 * np.max(np.abs(ref.i("M1_a"))))


# ---- induction machine and DC machine --------------------------------------
def test_induction_machine_matches_the_scipy_reference():
    # under a 20 us ceiling (np.interp at T_REF needs it); measured ew
    # 5.3e-8, ei 4.5e-6 (TR-BDF2 at rtol 1e-8: 5.7e-6, 4.3e-6; at 1e-10:
    # 1.3e-6, 4.9e-6 - ei is the reference's own floor)
    r = p.simulate(IM._machine(p.CircuitBuilder(), J=IM.J_DOL), t_end=IM.T_REF[-1],
                   engine="trbdf2", integrator="esdirk43", rtol=1e-8, atol=1e-9, dt=2e-5)
    ew, ei = IM._dol_error(np.asarray(r.times), r.v("w"), r.i("M_a"))
    assert ew < 5e-7
    assert ei < 2e-5


def test_dc_machine_matches_the_closed_form():
    r = p.simulate(DCM._build(), t_end=DCM.T_END, engine="trbdf2", integrator="esdirk43",
                   rtol=1e-8, atol=1e-11)
    assert DCM._speed_error(r) < 1e-8   # measured 8.2e-10 (TR-BDF2 3.2e-6)


# ---- Jiles-Atherton ---------------------------------------------------------
def test_jiles_atherton_core_is_refused_by_name():
    from . import test_hysteretic_core_in_loop as H
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "a", "gnd", 0.0, 1.0, 50.0, math.pi / 2)
    b.add_resistor("Rs", "a", "n1", 0.1)
    H._core(b)
    with pytest.raises(Exception, match=r"Jiles-Atherton core 'L_core'.*first order"):
        p.simulate(b, t_end=1e-3, engine="trbdf2", integrator="esdirk43")


@pytest.mark.parametrize("build,t_end", [(_laur, TD + 5e-6), (_igbt, TDI + 10e-6)],
                         ids=["lauritzen", "igbt"])
def test_interface_path_agrees_with_legacy(build, t_end):
    kw = dict(engine="trbdf2", integrator="esdirk43", rtol=1e-8, atol=1e-11)
    a = p.simulate(build(), t_end, **kw)
    b = p.simulate(build(), t_end, device_path="interface", **kw)
    xa, xb = np.asarray(a.states), np.asarray(b.states)
    np.testing.assert_allclose(xb[-1], xa[-1], rtol=0, atol=1e-6 * np.max(np.abs(xa)))


# ---- a rejected step leaves no device state behind ---------------------------
@pytest.mark.parametrize("build,t_end", [(_laur, TD + 5e-6), (_igbt, TDI + 10e-6),
                                         (_pmsm_free, 5e-3)], ids=["lauritzen", "igbt", "pmsm"])
def test_rejected_steps_leave_no_device_state(build, t_end):
    kw = dict(engine="trbdf2", integrator="esdirk43", rtol=1e-8, atol=1e-11)
    tried = p.simulate(build(), t_end, dt_init=t_end / 4, **kw)
    h1 = tried.times[1] - tried.times[0]
    clean = p.simulate(build(), t_end, dt_init=h1, **kw)
    assert tried._trbdf2_stats["n_reject"] >= clean._trbdf2_stats["n_reject"] + 2
    np.testing.assert_array_equal(np.asarray(tried.times), np.asarray(clean.times))
    np.testing.assert_array_equal(np.asarray(tried.states), np.asarray(clean.states))
