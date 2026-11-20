"""Soft-switching validation (OpenSpec ``add-soft-switching-validation``).

A half-bridge leg with a nonlinear output capacitance across each switch,
C(v) = C0 / (1 + v/V0)^m (``add_nonlinear_capacitor``: charge form), and a
dead time between the two gates. Every truth below is independent of the
simulator: closed forms in charge and energy, mpmath quadrature and mpmath's
Taylor ODE integrator on the circuit's own two state equations. No number in
this file comes from a Pulsim run.

Notation: v = switch node, Q(v) = int_0^v C, W(v) = int_0^v u C(u) du (the
energy stored in one C_oss at v), C_tot(v) = C(v) + C(V_dc - v).

Case A, current-driven transition (an inductor large enough to be a current
source I pulling charge out of the node). The high side turns off at t_off
from v_0 = V_dc - I/g_on. Charge conservation at the node gives the
transition time to ANY level v in closed form, no ODE:
    t(v) = t_off + ([Q(v_0) - Q(v)] + [Q(V_dc - v) - Q(V_dc - v_0)]) / I.
The low side then turns on at zero voltage (its diode clamps the node at
-(V_f + I/g_on)); the high side's turn-on one half period later is hard and
dissipates, with the node going from v_a = -(V_f + I/g_on) to v_b = V_dc - I/g_on:
    E_imp = V_dc [Q(v_b) - Q(v_a)] - [W(V_dc - v_b) - W(V_dc - v_a)] - [W(v_b) - W(v_a)]
(the charge drawn from V_dc is the low capacitor's charge change; what is
not stored is dissipated), plus the conduction of the window around it.

Case B, resonant transition (an inductor L to a stiff node V_m). In the dead
time  v' = -i / C_tot(v),  L i' = v - V_m,  so  d(L i^2/2) = -C_tot(v)(v - V_m) dv.
ZVS (the node reaches 0 V) iff
    L i_off^2 / 2 >= int_0^{V_dc} C_tot(v) (V_m - v) dv.
(With V_m = V_dc/2 the right side is 0 by symmetry: any positive current
gives ZVS; V_m = 300 V of a 400 V bus needs 31.2 uJ.) Short of it the node
turns around at v_min:  L i_off^2 / 2 = int_{v_min}^{v_0} C_tot(v)(V_m - v) dv,
and the low side's turn-on at v_1 dissipates
    E_imp = W(v_1) + V_dc [Q(V_dc) - Q(V_dc - v_1)] - [W(V_dc) - W(V_dc - v_1)].
The current at the turn-off follows from the ramped sources (both sources
ramp over T_R from 0 V, so v - V_m = (V_dc - V_m)/V_dc * V_dc(t) while the
high side is on):  i_off = i_0 + (V_dc - V_m)(t_off - T_R/2) / L,  to
R_on/L * int i dt ~ 3e-5 A (2e-5 relative), below every tolerance here.

Engines and tolerances (each through explicit options, never the defaults):
* ``pwl`` (dt = 1 ns, discontinuity_restart False): first order at a gate
  edge (AI_GUIDE trap 9, a stale trapezoidal history): held to ONE STEP on a
  transition time (a half-step shift is that order's signature) and to the
  shift times the slope (<= 2.9 V/ns here) on a voltage;
* ``pwl_restart`` (dt = 1 ns, discontinuity_restart True), ``trbdf2`` and
  ``esdirk43`` (``engine='trbdf2', integrator='esdirk43'``): second order or
  error-controlled; held to 0.05 ns on a 31-180 ns transition and 0.5 V
  (1.25e-3 of the bus) on a voltage; the trough of case B (v' = 0) is flat,
  so v_min and the voltage at the turn-on are held to the same 0.5 V;
* ``dsed``: refuses a Newton-stateful device by name (asserted).

Known open items this file cites instead of re-discovering them:
* with a nonlinear C_oss the fixed engine books (almost) none of E_oss at a
  hard turn-on: the 1 mOhm x 1 nF collapse (1 ps) is not resolved by a 1 ns
  step (open);
* trbdf2 at b9e46778 books more than the hard turn-on energy with a
  nonlinear C_oss (fixed on ``fix/core-trbdf2-state-error-norm``, not merged).
New findings are strict xfails ``F-S-<n>``, listed in the change's FINDINGS.md.
"""
from __future__ import annotations

import functools
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import pulsim as p
from pulsim.losses import device_loss_summary

mp = pytest.importorskip("mpmath")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from kpi import compute_zvs_fraction  # noqa: E402 - pure python

VDC, C0, V0, M = 400.0, 1e-9, 25.0, 0.5
GON, GOFF, VF = 1e3, 1e-9, 0.7
TR = 100e-9                     # source ramp: a consistent start at 0 V
FSW, DT = 1e6, 1e-9
I_A = 10.0                      # case A
L_B, VM_B = 10e-6, 300.0        # case B
DEAD_A, DEAD_PART, DEAD_FULL = 100e-9, 178e-9, 120e-9   # whole ns: on the pwl grid

ENGINES = {
    "pwl": dict(engine="pwl", dt=DT, discontinuity_restart=False),
    "pwl_restart": dict(engine="pwl", dt=DT, discontinuity_restart=True),
    "trbdf2": dict(engine="trbdf2"),
    "esdirk43": dict(engine="trbdf2", integrator="esdirk43"),
}
FIRST_ORDER = {"pwl"}
T_TOL = {"pwl": DT}             # one step (trap 9); else 0.05 ns
V_TOL = {"pwl": 3.0}            # 0.5 ns x 2.9 V/ns, x2; else 0.5 V

mp.mp.dps = 25


# ------------------------------------------------------------------ truths --
def _C(v):
    return mp.mpf(C0) / (1 + mp.mpf(v) / V0) ** M


def _Q(v):
    return mp.quad(_C, [0, v])


def _W(v):
    return mp.quad(lambda u: u * _C(u), [0, v])


def _Ctot(v):
    return _C(v) + _C(VDC - v)


def _t_cross_A(level):
    v0 = VDC - I_A / GON
    t_off = 0.5 / FSW - DEAD_A
    return float(t_off + ((_Q(v0) - _Q(level)) + (_Q(VDC - level) - _Q(VDC - v0))) / I_A)


def _e_hard_A(t_pre, t_post):
    va, vb = -(VF + I_A / GON), VDC - I_A / GON
    imp = VDC * (_Q(vb) - _Q(va)) - (_W(VDC - vb) - _W(VDC - va)) - (_W(vb) - _W(va))
    cond = (VF * I_A + I_A ** 2 / GON) * t_pre + I_A ** 2 / GON * t_post
    return float(imp + cond)


def _e_req_B():
    return mp.quad(lambda v: _Ctot(v) * (VM_B - v), [0, VDC])


def _i_crit_B():
    return float(mp.sqrt(2 * _e_req_B() / L_B))


def _case_B(frac, dead):
    """(i0, i_off, t_off, solution of the dead-time ODE from t_off)."""
    i_off = frac * _i_crit_B()
    t_off = 0.5 / FSW - dead
    i0 = i_off - (VDC - VM_B) * (t_off - TR / 2) / L_B
    v0 = mp.mpf(VDC) - mp.mpf(i_off) / GON
    sol = mp.odefun(lambda t, y: [-y[1] / _Ctot(y[0]), (y[0] - VM_B) / L_B],
                    0, [v0, mp.mpf(i_off)])
    return i0, i_off, t_off, sol


# ------------------------------------------------------------ simulations --
def _builder(case, i0=0.0):
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("Vdc", "vdc", "gnd", 0.0, VDC, 0.0, 1.0, 0.0, TR, 0.0)
    b.add_switch("Sh", "vdc", "sw", GON, GOFF)
    b.add_switch("Sl", "sw", "gnd", GON, GOFF)
    b.add_diode("Dh", "sw", "vdc", GON, GOFF, V_f=VF)
    b.add_diode("Dl", "gnd", "sw", GON, GOFF, V_f=VF)
    b.add_nonlinear_capacitor("Ch", "vdc", "sw", C0, V0, M)
    b.add_nonlinear_capacitor("Cl", "sw", "gnd", C0, V0, M)
    if case == "A":   # pulls I_A out of the switch node
        b.add_pulse_current_source("I", "gnd", "sw", 0.0, I_A, 0.0, 1.0, 0.0, TR, 0.0)
    else:
        b.add_pulse_voltage_source("Vm", "vm", "gnd", 0.0, VM_B, 0.0, 1.0, 0.0, TR, 0.0)
        b.add_inductor("L", "sw", "vm", L_B, i0=i0)
    return b


def _gates(b, dead):
    return p.make_dead_time_pwm_pair_fn(FSW, 0.5, b.switch_index_of("Sh"),
                                        b.switch_index_of("Sl"), b.graph.num_switches, dead)


@functools.lru_cache(maxsize=None)
def _run(engine, case, frac=None):
    if case == "A":
        b, dead, t_end = _builder("A"), DEAD_A, 2.2 / FSW
    else:
        dead = DEAD_PART if frac < 1 else DEAD_FULL
        b, t_end = _builder("B", _case_B(frac, dead)[0]), 0.5 / FSW + 60e-9
    fn = _gates(b, dead)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = p.simulate(b, t_end=t_end, switch_fn=fn, **ENGINES[engine])
        except p.SimulationAborted as e:
            if engine == "esdirk43" and "t = 0.000000" in str(e):
                pytest.xfail(f"finding F-S-1 (esdirk43 cannot start this circuit): {str(e)[:100]}")
            raise
    return b, fn, r


def _cross(t, v, level, t0, t1):
    k = np.nonzero((t >= t0) & (t <= t1))[0]
    for a, c in zip(k[:-1], k[1:]):
        if (v[a] - level) * (v[c] - level) <= 0 and v[a] != v[c]:
            return t[a] + (level - v[a]) * (t[c] - t[a]) / (v[c] - v[a])
    raise AssertionError(f"no crossing of {level} V in [{t0}, {t1}]")


def _window_loss(b, fn, r, t0, t1):
    rows = device_loss_summary(b, r, switch_fn=fn, t_from=t0, t_to=t1)
    return sum(e["E_total"] for e in rows if e["name"] in ("Sh", "Sl", "Dh", "Dl"))


# ------------------------------------------------------------------ tests --
def test_the_exact_engine_refuses_a_nonlinear_coss_by_name():
    b = _builder("A")
    with pytest.raises(p.PulsimError, match="NonlinearCapacitor"):
        p.simulate(b, t_end=1e-6, switch_fn=_gates(b, DEAD_A), engine="dsed")


def test_the_spec_constants_rederived():
    # add_nonlinear_capacitor's docstring: C0 = 2 nF, V0 = 25 V, m = 0.5 at
    # 400 V, Q(V)/V against C(V) differ by 1.61x
    q = 2e-9 * 25 / 0.5 * (np.sqrt(1 + 400 / 25) - 1)
    c = 2e-9 / np.sqrt(1 + 400 / 25)
    assert q / 400 / c == pytest.approx(1.61, abs=0.005)
    # case B's ZVS energy and critical current
    assert float(_e_req_B()) == pytest.approx(3.1231e-5, rel=1e-4)
    assert _i_crit_B() == pytest.approx(2.4992, rel=1e-4)


@pytest.fixture(params=list(ENGINES))
def engine(request):
    return request.param


@pytest.mark.parametrize("level", [300.0, 200.0, 100.0])
def test_current_driven_transition_time_is_the_charge_closed_form(engine, level):
    _, _, r = _run(engine, "A")
    t, v = np.asarray(r.times), np.asarray(r.v("sw"))
    t_off = 0.5 / FSW - DEAD_A
    got = _cross(t, v, level, t_off - 1e-9, t_off + DEAD_A)
    assert abs(got - _t_cross_A(level)) <= T_TOL.get(engine, 0.05e-9)


def test_zvs_fraction_soft_low_side_hard_high_side(engine):
    b, fn, r = _run(engine, "A")
    t, v = np.asarray(r.times), np.asarray(r.v("sw"))
    gate = [fn(x) for x in t]
    hs, ls = b.switch_index_of("Sh"), b.switch_index_of("Sl")
    lo = compute_zvs_fraction([g.get(ls) for g in gate], list(v), threshold_v=5.0)
    hi = compute_zvs_fraction([g.get(hs) for g in gate], list(VDC - v), threshold_v=5.0)
    assert (lo["zvs_fraction"], lo["total_turn_on_events"]) == (1.0, 2.0)
    assert (hi["zvs_fraction"], hi["total_turn_on_events"]) == (0.0, 2.0)


KNOWN_LOSS = {
    "pwl": "known open item: the fixed engine books almost none of E_oss (unresolved RC collapse)",
    "pwl_restart": "known open item: the fixed engine books almost none of E_oss (unresolved RC collapse)",
    "trbdf2": "known open item: trbdf2 overbooks a hard turn-on with nonlinear C_oss "
              "(fixed on fix/core-trbdf2-state-error-norm, not merged)",
}


@pytest.mark.parametrize("case", ["A", "B"])
def test_hard_turn_on_energy_is_booked(engine, case, request):
    if engine in KNOWN_LOSS:
        request.applymarker(pytest.mark.xfail(strict=True, reason=KNOWN_LOSS[engine]))
    tp, tq = 2e-9, 20e-9
    if case == "A":
        b, fn, r = _run(engine, "A")
        t_on, truth = 1.0 / FSW, _e_hard_A(tp, tq)
    else:
        b, fn, r = _run(engine, "B", 0.7)
        _, _, t_off, sol = _case_B(0.7, DEAD_PART)
        v1 = sol(mp.mpf(DEAD_PART))[0]
        truth = float(_W(v1) + VDC * (_Q(VDC) - _Q(VDC - v1)) - (_W(VDC) - _W(VDC - v1)))
        t_on = t_off + DEAD_PART
    got = _window_loss(b, fn, r, t_on - tp, t_on + tq)
    assert got == pytest.approx(truth, rel=0.05)


def test_partial_zvs_trough_is_the_energy_closed_form(engine):
    _, _, r = _run(engine, "B", 0.7)
    i0, i_off, t_off, sol = _case_B(0.7, DEAD_PART)
    v0 = mp.mpf(VDC) - mp.mpf(i_off) / GON
    vmin = float(mp.findroot(
        lambda x: mp.quad(lambda v: _Ctot(v) * (VM_B - v), [x, v0]) - L_B * mp.mpf(i_off) ** 2 / 2, 60))
    t, v = np.asarray(r.times), np.asarray(r.v("sw"))
    win = (t > t_off) & (t < t_off + DEAD_PART)
    tol = V_TOL.get(engine, 0.5) if engine in FIRST_ORDER else 0.5
    assert vmin > 30.0                       # ZVS lost by design (0.7 x the critical current)
    assert v[win].min() == pytest.approx(vmin, abs=tol)
    # the node at the low side's (hard) turn-on, from the ODE
    # (one step before the edge: the fixed engine without restart reads the
    # gate at the step's END, so its sample AT the edge is already switched)
    t_on = t_off + DEAD_PART
    want = float(sol(mp.mpf(DEAD_PART - DT))[0])
    assert np.interp(t_on - DT, t, v) == pytest.approx(want, abs=tol)


@pytest.mark.parametrize("frac", [0.7, 1.5])
def test_resonant_trajectory_is_the_ode_truth(engine, frac):
    _, _, r = _run(engine, "B", frac)
    dead = DEAD_PART if frac < 1 else DEAD_FULL
    _, _, t_off, sol = _case_B(frac, dead)
    t, v = np.asarray(r.times), np.asarray(r.v("sw"))
    tol = V_TOL.get(engine, 0.5)
    for dt_ in (20e-9, 40e-9, 60e-9):
        want = float(sol(mp.mpf(dt_))[0])
        assert np.interp(t_off + dt_, t, v) == pytest.approx(want, abs=tol), dt_


def test_full_zvs_low_side_turns_on_at_its_diode_clamp(engine, request):
    if engine == "pwl":
        # the clamping diode's current alternates ~5 A / 0 A step by step
        # around i_L = 2.5 A (a trapezoidal ring of the 1 ps clamp under a
        # 1 ns step): the stale-history ringing of AI_GUIDE trap 9
        request.applymarker(pytest.mark.xfail(strict=True, reason="known: trap 9 ringing (restart off)"))
    b, fn, r = _run(engine, "B", 1.5)
    _, _, t_off, sol = _case_B(1.5, DEAD_FULL)
    t, v = np.asarray(r.times), np.asarray(r.v("sw"))
    # the node reaches 100 V at the ODE's time
    t100 = float(mp.findroot(lambda s: sol(s)[0] - 100, (mp.mpf(40e-9), mp.mpf(80e-9)), solver='anderson'))
    assert abs(_cross(t, v, 100.0, t_off, t_off + DEAD_FULL) - (t_off + t100)) <= T_TOL.get(engine, 0.05e-9)
    gate = [fn(x).get(b.switch_index_of("Sl")) for x in t]
    k = compute_zvs_fraction(gate, list(v), threshold_v=5.0)
    assert (k["zvs_fraction"], k["total_turn_on_events"]) == (1.0, 1.0)
    # clamped by the low diode before the gate: -(V_f + i_L/g_on), i_L in (0, i_off)
    v_on = np.interp(t_off + DEAD_FULL - DT, t, v)
    assert -(VF + 4.0 / GON) < v_on < -VF


@pytest.mark.xfail(strict=True, reason="finding F-S-1: esdirk43 aborts at t = 0 (Newton at h_min) "
                                       "with a nonlinear capacitor driven by a current source")
def test_esdirk43_starts_a_current_driven_nonlinear_capacitor():
    """Minimal: R + charge-form capacitor (m = 0, i.e. a LINEAR law) + 10 A
    current source. ``add_capacitor`` in its place runs; trbdf2 runs both.
    At b9e46778: Newton fails at t = 0, h_min, ||residual|| 7.6e-6 at node sw."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 0.0)
    b.add_resistor("R", "a", "sw", 1e-3)
    b.add_nonlinear_capacitor("C", "sw", "gnd", 1e-9, 25.0, 0.0)
    b.add_current_source("I", "gnd", "sw", 10.0)
    r = p.simulate(b, t_end=3e-7, engine="trbdf2", integrator="esdirk43")
    assert r.v("sw")[-1] == pytest.approx(-0.01, rel=1e-6)


# ------------------------------------------------- KPI helpers, synthetic --
def test_kpi_zvs_and_zcs_count_hand_built_edges():
    from kpi import compute_zcs_fraction
    gate = [0, 0, 1, 1, 0, 0, 1, 1, 0]
    v = [400, 0.3, 0.1, 0.1, 0.1, 380, 390, 0.1, 0.1]   # 1st turn-on soft, 2nd hard
    i = [0, 0, 5, 0.05, 0, 0, 5, 7, 0]                   # 1st turn-off soft, 2nd hard
    z = compute_zvs_fraction([bool(g) for g in gate], v, threshold_v=1.0)
    assert z == {"zvs_fraction": 0.5, "total_turn_on_events": 2.0}
    c = compute_zcs_fraction([bool(g) for g in gate], i, threshold_a=0.1)
    assert c == {"zcs_fraction": 0.5, "total_turn_off_events": 2.0}


def test_kpi_switching_loss_is_an_energy():
    """v falls 10 -> 0 V while i rises 0 -> 5 A over 1 us after the edge:
    E = int_0^1us 10(1 - s) 5 s ds = 50/6 uJ (trapezoid on 10 001 samples:
    error (h^2/12) * int|p''| = 8e-8 relative)."""
    from kpi import compute_switching_loss_per_event
    t = np.linspace(0.0, 3e-6, 30001)
    s = np.clip((t - 1e-6) / 1e-6, 0.0, 1.0)
    v = 10.0 * (1 - s)
    i = 5.0 * s
    gate = [bool(x >= 1e-6) for x in t]
    out = compute_switching_loss_per_event(gate, list(v), list(i), list(t), window_s=1.5e-6)
    assert out["switching_event_count"] == 1.0
    assert out["switching_energy_j"] == pytest.approx(50 / 6 * 1e-6, rel=1e-6)
    assert out["switching_loss_w_avg"] == pytest.approx(50 / 6 * 1e-6 / 3e-6, rel=1e-6)
    with pytest.raises(ValueError):
        compute_switching_loss_per_event(gate, list(v), list(i), list(t), window_s=0.0)
