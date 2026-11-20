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

Engines and tolerances (each engine selected through explicit options; the
variable-step ones at their default rtol / atol):
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
        r = p.simulate(b, t_end=t_end, switch_fn=fn, **ENGINES[engine])
    return b, fn, r


F_S_6 = ("finding F-S-6: discontinuity_restart refuses by name an event that moves a "
         "nonlinear capacitor (the low-side turn-on at 0.5 us moves Ch); ran at b9e46778, "
         "refused since cedac36b")


def _line_marks(request, engine, case):
    """F-S-6, found when this suite was merged onto the integration line
    (batch 14t). True when a mark was applied. (F-S-5, the trbdf2 / esdirk43
    abort on case A, is fixed: its marks were removed in batch 14u.)"""
    if engine == "pwl_restart":
        request.applymarker(pytest.mark.xfail(strict=True, raises=p.UnsupportedError, reason=F_S_6))
        return True
    return False


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
def test_current_driven_transition_time_is_the_charge_closed_form(engine, level, request):
    _line_marks(request, engine, "A")
    _, _, r = _run(engine, "A")
    t, v = np.asarray(r.times), np.asarray(r.v("sw"))
    t_off = 0.5 / FSW - DEAD_A
    got = _cross(t, v, level, t_off - 1e-9, t_off + DEAD_A)
    assert abs(got - _t_cross_A(level)) <= T_TOL.get(engine, 0.05e-9)


def test_zvs_fraction_soft_low_side_hard_high_side(engine, request):
    _line_marks(request, engine, "A")
    b, fn, r = _run(engine, "A")
    t, v = np.asarray(r.times), np.asarray(r.v("sw"))
    gate = [fn(x) for x in t]
    hs, ls = b.switch_index_of("Sh"), b.switch_index_of("Sl")
    lo = compute_zvs_fraction([g.get(ls) for g in gate], list(v), threshold_v=5.0)
    hi = compute_zvs_fraction([g.get(hs) for g in gate], list(VDC - v), threshold_v=5.0)
    assert (lo["zvs_fraction"], lo["total_turn_on_events"]) == (1.0, 2.0)
    assert (hi["zvs_fraction"], hi["total_turn_on_events"]) == (0.0, 2.0)


# trbdf2's overbooking (1381.7 uJ case A at b9e46778) is gone with the
# state error norm: case B reads 1.6891 uJ (truth 1.6878); case A (F-S-5
# fixed) reads trbdf2 62.8490 / esdirk43 62.9259 uJ (truth 62.8335).
# pwl_restart (1.44 uJ case A at b9e46778) is refused now: F-S-6.
KNOWN_LOSS = {
    "pwl": "known open item: the fixed engine books almost none of E_oss (unresolved RC collapse)",
}


@pytest.mark.parametrize("case", ["A", "B"])
def test_hard_turn_on_energy_is_booked(engine, case, request):
    if not _line_marks(request, engine, case) and engine in KNOWN_LOSS:
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


def test_partial_zvs_trough_is_the_energy_closed_form(engine, request):
    _line_marks(request, engine, "B")
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
def test_resonant_trajectory_is_the_ode_truth(engine, frac, request):
    _line_marks(request, engine, "B")
    _, _, r = _run(engine, "B", frac)
    dead = DEAD_PART if frac < 1 else DEAD_FULL
    _, _, t_off, sol = _case_B(frac, dead)
    t, v = np.asarray(r.times), np.asarray(r.v("sw"))
    tol = V_TOL.get(engine, 0.5)
    for dt_ in (20e-9, 40e-9, 60e-9):
        want = float(sol(mp.mpf(dt_))[0])
        assert np.interp(t_off + dt_, t, v) == pytest.approx(want, abs=tol), dt_


def test_full_zvs_low_side_turns_on_at_its_diode_clamp(engine, request):
    _line_marks(request, engine, "B")
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


def test_esdirk43_starts_a_current_driven_nonlinear_capacitor():
    """Minimal: R + charge-form capacitor (m = 0, i.e. a LINEAR law) + 10 A
    current source. ``add_capacitor`` in its place runs; trbdf2 runs both.
    Finding F-S-1: at b9e46778 Newton failed at t = 0, h_min, ||residual||
    7.6e-6 at node sw; closed on the integration line (batch 14t)."""
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


# ------------------------------------------- LLC half-bridge, exact truth --
LLC_YAML = (Path(__file__).resolve().parents[2] / "benchmarks" / "circuits" / "switching"
            / "isolated" / "llc_half_bridge_zvs.yaml")
LLC_ENGINES = {**ENGINES, "dsed": dict(engine="dsed")}
LLC_OFF = sorted([k * 8e-6 + 3.7e-6 for k in range(5)] + [k * 8e-6 + 7.7e-6 for k in range(5)])


@pytest.fixture(scope="module")
def llc_truth():
    """The netlist solved by ``benchmarks/tools/exact_truth.py`` (mpmath, 30
    digits; it never imports pulsim): per turn-off edge t_e, the tank current
    i_Lr(t_e), the time the node crosses 200 V and the body diode's turn-on
    (the end of the transition, v_DS = 0: ZVS)."""
    pytest.importorskip("yaml")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "tools"))
    import exact_truth as et
    sol = et.solve(str(LLC_YAML), dps=30)

    def v(s):
        return sol.sample(["V(mid)"], [s])["V(mid)"][0]
    rows = []
    for te in LLC_OFF:
        tr = min(e["t"] for e in sol.events if "root" in e["cause"] and e["t"] > te)
        t200 = mp.findroot(lambda s: v(s) - 200, (mp.mpf(te), tr), solver="anderson")
        i = sol.sample(["I(Lr)"], [mp.mpf(te)])["I(Lr)"][0]
        rows.append((te, float(t200), float(i), float(tr)))
    return sol, rows


@functools.lru_cache(maxsize=None)
def _llc_run(engine):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "tools"))
    import v1_reader
    case = v1_reader.load(LLC_YAML)
    kw = dict(LLC_ENGINES[engine])
    eng = kw.pop("engine")
    kw.pop("dt", None)          # the netlist's dt (10 ns) is the fixed engine's step
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return case.run(eng, **kw)


LLC_KNOWN = "known: trap 9 — without restart the fixed engine rings (body diode / switch 14 A / 33 A " \
            "step by step) and dumps C_oss in one step: the transition collapses"


@pytest.fixture(params=list(LLC_ENGINES))
def llc_engine(request):
    return request.param


def test_llc_truth_is_zvs_on_every_edge(llc_truth):
    """The design (dead time 300 ns, 2 C_oss V_dc = 160 nC, |i_Lr| >= 1.46 A
    at turn-off: <= 110 ns at constant current) reaches v_DS = 0 before
    every gate: the spec's >= 95 % holds on the truth itself (10 of 10)."""
    _, rows = llc_truth
    assert all(tr - te < 300e-9 for te, _, _, tr in rows)
    assert min(abs(i) for _, _, i, _ in rows) > 1.4


def test_llc_dead_time_commutation_against_exact_truth(llc_truth, llc_engine, request):
    """i_Lr at each turn-off within 2.5e-4 A (1e-4 of the 2.5 A peak; the
    worst seen 3.4e-5, pwl+restart) and every engine sample inside each
    transition within 0.5 V of the truth (1.25e-3 of the bus). Both bounds
    hold the trbdf2 family to its default rtol with margin."""
    if llc_engine == "pwl":
        request.applymarker(pytest.mark.xfail(strict=True, reason=LLC_KNOWN))
    sol, rows = llc_truth
    r = _llc_run(llc_engine)
    t, v, il = np.asarray(r.times), np.asarray(r.v("mid")), np.asarray(r.i("Lr"))
    for te, _, i, tr in rows:
        assert np.interp(te, t, il) == pytest.approx(i, abs=2.5e-4), te
        k = np.nonzero((t > te) & (t < tr))[0]
        assert len(k) >= 1
        want = np.array([float(x) for x in sol.sample(["V(mid)"], [mp.mpf(float(s)) for s in t[k]])["V(mid)"]])
        assert np.max(np.abs(v[k] - want)) <= 0.5, te


def test_llc_zvs_fraction_kpi(llc_engine):
    r = _llc_run(llc_engine)
    t, v = np.asarray(r.times), np.asarray(r.v("mid"))
    T = 8e-6
    on_hi = [((x % T) < 3.7e-6) for x in t]
    on_lo = [(4e-6 <= (x % T) < 7.7e-6) for x in t]
    hi = compute_zvs_fraction(on_hi, list(400.0 - v), threshold_v=5.0)
    lo = compute_zvs_fraction(on_lo, list(v), threshold_v=5.0)
    # 8, 16, 24, 32 us (+ 40 us when the grid ends on it)
    assert hi["zvs_fraction"] == 1.0 and hi["total_turn_on_events"] >= 4.0
    assert lo["zvs_fraction"] == 1.0 and lo["total_turn_on_events"] == 5.0


# ------------------------------- the exact-truth tool itself (F-S-3, fixed) --
TOOLS = Path(__file__).resolve().parents[2] / "benchmarks" / "tools"
PSFB_YAML = LLC_YAML.with_name("psfb_full_bridge_zvs.yaml")


def _et():
    pytest.importorskip("yaml")
    sys.path.insert(0, str(TOOLS))
    import exact_truth as et
    return et


def _diode_law_holds(sol, rel=1e-9):
    """The truth obeys its own diode law (V_f = V_th = 0, conductance form)
    at the midpoint of every segment: an OFF diode is not forward biased, an
    ON one carries forward current (i = g_on v >= 0). Before F-S-3's fix the
    LLC below kept its rectifier OFF with 17.7 V forward for 2.5 us."""
    ck = sol.ck
    for seg in sol.segs:
        x = seg.dyn.X * seg.y(seg.tau / 2)
        tol = rel * max(abs(v) for v in x)
        for dv, on in zip(ck.devs, seg.mode):
            if dv["kind"] == "D":
                v = sum(s * x[ck._vi(n)] for n, s in ((dv["a"], 1), (dv["k"], -1)) if n != "gnd")
                assert (v >= -tol) if on else (v <= tol), (dv["name"], on, float(seg.t0), float(v))


def _expm_rel(sol, A, cols, x_of, seg_t0, tau_end):
    """max relative distance between the tool and x(tau) = expm(A tau) x0
    (x0 = the tool's right limit at seg_t0) at 1/4, 1/2, 3/4, 0.999 of
    [seg_t0, seg_t0 + tau_end]; x_of maps the tool's columns to the state."""
    s = sol.sample(cols, [seg_t0, seg_t0])
    x0 = mp.matrix(x_of({c: s[c][1] for c in cols}) + [1])
    worst = mp.mpf(0)
    for f in ("0.25", "0.5", "0.75", "0.999"):
        tau = mp.mpf(f) * tau_end
        want = mp.expm(A * tau) * x0
        got = x_of({c: v[0] for c, v in sol.sample(cols, [seg_t0 + tau]).items()})
        worst = max([worst] + [abs(g - want[i]) / max(abs(want[i]), 1) for i, g in enumerate(got)])
    return worst, x0


def test_exact_tool_settles_the_llc_with_a_transformer_rectifier():
    """F-S-3. The netlist of llc_half_bridge_zvs.yaml with R_ac replaced by an
    ideal 4:1 transformer, a diode bridge, C_o = 10 uF (48 V) and R_o = 10 Ohm.
    Before the fix the tool raised "no consistent diode mode" at the first
    gate-off (3.7 us) at dps 30 -- after a silently wrong 3.7 us: z, the
    slow coordinates, were columns of Eh^(index+1), aligned to the slowest
    mode to ~1e-27, so a 400 V state was a difference of ~1e18 numbers and
    every root test's noise bound read 3e4 V.

    Independent check, by hand: in the dead time (switches and every diode
    OFF, each a g_off = 1e-9 conductance) the secondary loads the winding with
    g_off (v_s1 + v_s2 = v_o, i_w = g_off n v_p), i.e. the primary with
    G_p = n^2 g_off, n = 1/4; with v_c = v_mid - v_a (C_r):
        (C_hi + C_lo) v_mid' = 2 g_off (V_dc - v_mid) - 2 g_off v_mid - i_r
        C_r v_c' = i_r,  L_r i_r' = v_mid - v_c - v_p,  L_m i_m' = v_p,
        v_p = (i_r - i_m) / G_p,  C_o v_o' = -(g_off + 1/R_o) v_o.
    x(tau) = expm(A tau) x0 from the tool's state at the gate edge, held to
    1e-9 relative; the rectifier turns on (D2 and D3 together) when
    n v_p = -v_o, held to 1e-9 of the interval."""
    import yaml
    et = _et()
    d = yaml.safe_load(LLC_YAML.read_text(encoding="utf-8"))
    d["simulation"]["tstop"] = 3.75e-6
    d["components"] = [c for c in d["components"] if c["name"] != "Rac"] + [
        {"type": "transformer", "name": "T1", "nodes": ["p", 0, "s1", "s2"], "turns_ratio": 4.0},
        *({"type": "diode", "name": n, "nodes": nd, "g_on": 1000.0, "g_off": 1e-9}
          for n, nd in (("D1", ["s1", "out"]), ("D2", ["s2", "out"]), ("D3", [0, "s1"]), ("D4", [0, "s2"]))),
        {"type": "capacitor", "name": "Co", "nodes": ["out", 0], "value": 10e-6, "ic": 48.0},
        {"type": "resistor", "name": "Ro", "nodes": ["out", 0], "value": 10.0}]
    sol = et.Solver(et.Circuit(d), 30).run()          # the predecessor's failing call (dps 30)
    _diode_law_holds(sol)
    first = next(e for e in sol.events if e["cause"].endswith("root"))
    assert first["cause"] in ("D1 root", "D4 root") and first["t"] < 1e-9   # the rectifier conducts at once
    te = min((e["t"] for e in sol.events if e["cause"] == "schedule"), key=lambda t: abs(t - mp.mpf("3.7e-6")))
    seg = next(s for s in sol.segs if s.t0 == te)
    assert not any(seg.mode)                           # dead time, every diode OFF
    g, Ch, Cl, Cr, Lr, Lm, n, Co, Ro, V = (mp.mpf(x) for x in (1e-9, 200e-12, 200e-12, 24e-9, 60e-6,
                                                                300e-6, 0.25, 10e-6, 10.0, 400))
    Gp, Cm = n ** 2 * g, Ch + Cl
    A = mp.matrix([[-4 * g / Cm, 0, -1 / Cm, 0, 0, 2 * g * V / Cm],
                   [0, 0, 1 / Cr, 0, 0, 0],
                   [1 / Lr, -1 / Lr, -1 / (Lr * Gp), 1 / (Lr * Gp), 0, 0],
                   [0, 0, 1 / (Lm * Gp), -1 / (Lm * Gp), 0, 0],
                   [0, 0, 0, 0, -(g + 1 / Ro) / Co, 0],
                   [0] * 6])
    cols = ["V(mid)", "V(a)", "I(Lr)", "I(Lm)", "V(out)"]
    worst, x0 = _expm_rel(sol, A, cols, lambda s: [s["V(mid)"], s["V(mid)"] - s["V(a)"], s["I(Lr)"],
                                                   s["I(Lm)"], s["V(out)"]], te, seg.tau)
    assert worst < 1e-9
    assert any(e["t"] == te + seg.tau and e["cause"] in ("D2 root", "D3 root") for e in sol.events)

    def u_plus_vo(tau):
        y = mp.expm(A * tau) * x0
        return n * (y[2] - y[3]) / Gp + y[4]
    t_on = mp.findroot(u_plus_vo, (seg.tau * mp.mpf("0.99"), seg.tau * mp.mpf("1.01")), solver="anderson")
    assert abs(t_on - seg.tau) < 1e-9 * seg.tau


@pytest.fixture(scope="module")
def psfb_truth():
    """psfb_full_bridge_zvs.yaml solved by the exact tool (dps 30, ~7 s)."""
    return _et().solve(str(PSFB_YAML), dps=30)


def test_exact_tool_psfb_lagging_leg_transition_closed_form(psfb_truth):
    """The lagging leg's first turn-off (S3 at 2.92 us) on the tool against
    a hand closed form. The segment from the rectifier's freewheel onset
    (D5..D8 all ON, the secondary shorted) to D4's turn-on: S1 and D1 ON
    (g_on = 1e3 each, vin - a), every other bridge device OFF (g_off). With
    all four rectifier diodes at G = g_on: v_s1 + v_s2 = -i_o/G, v_r = -i_o/G,
    the winding sees G, the primary G_s = n^2 G (n = 1/4):
        2C v_a' = 2G (V_dc - v_a) - 2 g_off v_a - i_lk
        2C v_b' = 2 g_off (V_dc - v_b) - 2 g_off v_b + i_lk
        L_lk i_lk' = v_a - v_b - i_lk / G_s
        L_o i_o' = -i_o / G - v_o,   C_o v_o' = i_o - v_o / R_o.
    Held to 1e-9 relative (1 V / 1 A floors); D4 turns on at v_b = 0 within
    1e-9 of the 57 ns interval. The tool's diode law holds on every segment."""
    sol = psfb_truth
    _diode_law_holds(sol)
    ck = sol.ck
    want = {"S1", "D1", "D5", "D6", "D7", "D8"}
    seg = next(s for s in sol.segs if s.t0 > mp.mpf("2.9e-6")
               and {dv["name"] for dv, b in zip(ck.devs, s.mode) if b} == want)
    assert 50e-9 < seg.tau < 60e-9
    g, G, C, L, Lo, Co, Ro, n, V = (mp.mpf(x) for x in (1e-9, 1000, 200e-12, 10e-6, 20e-6, 20e-6, 2.0, 0.25, 400))
    Gs = n ** 2 * G
    A = mp.matrix([[-(2 * G + 2 * g) / (2 * C), 0, -1 / (2 * C), 0, 0, 2 * G * V / (2 * C)],
                   [0, -4 * g / (2 * C), 1 / (2 * C), 0, 0, 2 * g * V / (2 * C)],
                   [1 / L, -1 / L, -1 / (L * Gs), 0, 0, 0],
                   [0, 0, 0, -1 / (Lo * G), -1 / Lo, 0],
                   [0, 0, 0, 1 / Co, -1 / (Co * Ro), 0],
                   [0] * 6])
    cols = ["V(a)", "V(b)", "I(Llk)", "I(Lo)", "V(out)"]
    worst, x0 = _expm_rel(sol, A, cols, lambda s: [s[c] for c in cols], seg.t0, seg.tau)
    assert worst < 1e-9
    t_on = mp.findroot(lambda tau: (mp.expm(A * tau) * x0)[1], (seg.tau * mp.mpf("0.99"), seg.tau * mp.mpf("1.01")),
                       solver="anderson")
    assert abs(t_on - seg.tau) < 1e-9 * seg.tau
    assert any(e["t"] == seg.t0 + seg.tau and e["cause"] == "D4 root" for e in sol.events)


# -------------------------------------- PSFB full bridge, exact truth (3.x) --
PSFB_T, PSFB_DEAD = 10e-6, 80e-9
# (switch, its node, the body diode that clamps the node after its turn-off, first turn-off)
PSFB_LEGS = (("S1", "a", "D2", 4.92e-6), ("S2", "a", "D1", 9.92e-6),
             ("S3", "b", "D4", 2.92e-6), ("S4", "b", "D3", 7.92e-6))
PSFB_ON = {"S1": 0.0, "S2": 5e-6, "S3": 8e-6, "S4": 3e-6}        # gate-on instants mod T
PSFB_KNOWN = "known: trap 9 -- without restart the fixed engine reads the gate at the step's end " \
             "(one 10 ns step early at every edge) and dumps C_oss in about one step"


def _psfb_edges(sol):
    """(node, t_off, i_Llk(t_off), t_clamp): every gate-off in [0, 30 us] and
    the instant the opposite body diode turns on (v_DS = 0 on the switch the
    dead time ends on)."""
    rows = []
    for _, node, dio, t1 in PSFB_LEGS:
        for k in range(3):
            te = min((e["t"] for e in sol.events if e["cause"] == "schedule"),
                     key=lambda t: abs(t - mp.mpf(t1 + k * PSFB_T)))
            tc = min(e["t"] for e in sol.events if e["cause"] == f"{dio} root" and e["t"] > te)
            rows.append((node, te, sol.sample(["I(Llk)"], [te])["I(Llk)"][0], tc))
    return rows


def test_psfb_truth_is_zvs_on_every_edge(psfb_truth):
    """Leading leg (a) on the reflected load current: ~32-35 ns; lagging leg
    (b) on the L_lk energy alone (rectifier shorted): ~47-57 ns; both under
    the 80 ns dead time, and the body diode still conducts at the gate (the
    node sits within 0.01 V of its rail 1 ns before the switch turns on)."""
    rows = _psfb_edges(psfb_truth)
    assert len(rows) == 12
    for node, te, i, tc in rows:
        assert tc - te < PSFB_DEAD and abs(i) > 3.0, (node, float(te))
        v = psfb_truth.sample([f"V({node})"], [te + mp.mpf(PSFB_DEAD - 1e-9)])[f"V({node})"][0]
        assert min(abs(v), abs(v - 400)) < 0.01, (node, float(te), float(v))


PSFB_ENGINES = {**LLC_ENGINES}


@functools.lru_cache(maxsize=None)
def _psfb_run(engine):
    sys.path.insert(0, str(TOOLS))
    import v1_reader
    kw = dict(PSFB_ENGINES[engine])
    eng = kw.pop("engine")
    kw.pop("dt", None)          # the netlist's dt (10 ns) is the fixed engine's step
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return v1_reader.load(PSFB_YAML).run(eng, **kw)


@pytest.fixture(params=list(PSFB_ENGINES))
def psfb_engine(request):
    return request.param


def test_psfb_dead_time_commutation_against_exact_truth(psfb_truth, psfb_engine, request):
    """i_Llk at each of the 12 gate-off edges within 5e-3 A (1e-3 of its
    ~5 A peak) and every engine sample inside each transition within 0.5 V
    of the truth (1.25e-3 of the bus, the LLC's bound). Seen: pwl+restart
    6.7e-4 A / 0.46 V (second order on its fixed 10 ns grid: deterministic,
    so the thin 8 % margin is not a platform risk), trbdf2 1.7e-5 A / 0.026 V,
    esdirk43 6.0e-6 A / 0.010 V, dsed 1.8e-7 A / 2e-5 V; pwl without
    restart 0.11 A / 245 V (trap 9)."""
    if psfb_engine == "pwl":
        request.applymarker(pytest.mark.xfail(strict=True, reason=PSFB_KNOWN))
    sol = psfb_truth
    r = _psfb_run(psfb_engine)
    t, il = np.asarray(r.times), np.asarray(r.i("Llk"))
    for node, te, i, tc in _psfb_edges(sol):
        assert np.interp(float(te), t, il) == pytest.approx(float(i), abs=5e-3), float(te)
        k = np.nonzero((t > te) & (t < tc))[0]
        assert len(k) >= 1
        want = np.array([float(x) for x in sol.sample([f"V({node})"], [mp.mpf(float(s)) for s in t[k]])[f"V({node})"]])
        assert np.max(np.abs(np.asarray(r.v(node))[k] - want)) <= 0.5, float(te)


def test_psfb_time_weighted_means_against_exact_truth(psfb_truth, psfb_engine, request):
    """V(out) and I(Lo) averaged over the last period, [20, 30] us, time
    weighted (pulsim.time_average: trbdf2 / dsed grids cluster at events),
    within 1e-3 of the truth's exact window mean (the scoreboard's pass
    grade). Seen: pwl+restart 2.8e-5 / 1.3e-4, trbdf2 1.2e-6 / 4.1e-6,
    esdirk43 3.6e-7 / 2.8e-6, dsed 1.2e-8 / 4.1e-7; pwl 1.2e-3 / 2.4e-3
    (one step early at each of the 8 edges of the window: trap 9)."""
    if psfb_engine == "pwl":
        request.applymarker(pytest.mark.xfail(strict=True, reason=PSFB_KNOWN))
    w = psfb_truth.window(["V(out)", "I(Lo)"], mp.mpf("20e-6"), mp.mpf("30e-6"))
    r = _psfb_run(psfb_engine)
    t = np.asarray(r.times)
    got = {"V(out)": p.time_average(t, np.asarray(r.v("out")), t_from=20e-6, t_to=30e-6),
           "I(Lo)": p.time_average(t, np.asarray(r.i("Lo")), t_from=20e-6, t_to=30e-6)}
    for c, v in got.items():
        assert v == pytest.approx(float(w[c]["mean"]), rel=1e-3), c


def test_psfb_zvs_fraction_kpi(psfb_engine, request):
    """Every turn-on of all four switches at |v_DS| < 5 V (one sample
    before the gate): leading and lagging leg 100 %, as in the truth."""
    if psfb_engine == "pwl":
        # S3 at 8 us: v_DS -0.004 / 8.53 / -0.001 V on three steps -- the
        # body-diode clamp rings step by step (trap 9): 2 of 3 counted ZVS
        request.applymarker(pytest.mark.xfail(strict=True, reason=PSFB_KNOWN))
    r = _psfb_run(psfb_engine)
    t, va, vb = np.asarray(r.times), np.asarray(r.v("a")), np.asarray(r.v("b"))
    vds = {"S1": 400.0 - va, "S2": va, "S3": 400.0 - vb, "S4": vb}
    for sw, t_on in PSFB_ON.items():
        gate = [((x - t_on) % PSFB_T) < PSFB_T / 2 - PSFB_DEAD for x in t]
        k = compute_zvs_fraction(gate, list(vds[sw]), threshold_v=5.0)
        assert k["zvs_fraction"] == 1.0 and k["total_turn_on_events"] >= 2.0, (sw, k)


def test_exact_engine_keeps_the_output_inductor_current_at_a_rectifier_start():
    """Minimal (11 elements, from the PSFB's power-transfer start): 400 V ->
    L_lk (10 uH, 0 A) -> ideal 4:1 primary (p, b); b to ground through 1 mOhm
    (S4 ON) || its body diode || 200 pF; secondary full bridge into L_o (20 uH,
    24 A) to ground. Consistent: the four rectifier diodes share L_o (12 A
    each), L_lk ramps at 40 A/us. At b9e46778 ``engine='dsed'`` starts L_o at
    -2.8e-9 A and reaches i_Llk(200 ns) = 0.2424 A (truth 6.0605 A); without
    the 200 pF or the body diode it agrees; with a diode b -> 400 V added it
    raises DiodeModeInconsistentError (the body diode ON at -1.2e-4 A at
    0.2 ns: the predecessor's PSFB failure). Finding F-S-4, closed on the
    integration line (batch 14t): 24 A kept, 6.06050914 A, with the b -> 400 V
    diode as well."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Va", "a", "gnd", 400.0)
    b.add_resistor("S4", "b", "gnd", 1e-3)
    b.add_diode("D4", "gnd", "b", 1000.0, 1e-9)
    b.add_capacitor("C4", "b", "gnd", 200e-12)
    b.add_inductor("Llk", "a", "p", 10e-6, 0.0)
    b.add_ideal_transformer("T1", "p", "b", "s1", "s2", 0.25)
    for name, an, ca in (("D5", "s1", "r"), ("D6", "s2", "r"), ("D7", "gnd", "s1"), ("D8", "gnd", "s2")):
        b.add_diode(name, an, ca, 1000.0, 1e-9)
    b.add_inductor("Lo", "r", "gnd", 20e-6, 24.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=2e-7, dt=1e-9, engine="dsed")
    # truth (exact tool, same netlist): i_Lo(0) = 24 A, i_Llk(200 ns) = 6.06050914 A
    assert r.i("Lo")[0] == pytest.approx(24.0, abs=1e-6)
    assert np.interp(2e-7, np.asarray(r.times), np.asarray(r.i("Llk"))) == pytest.approx(6.06050914, rel=1e-6)
