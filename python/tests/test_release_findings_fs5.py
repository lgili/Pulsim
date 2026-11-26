"""Finding F-S-5: the variable-step engine aborted on a half bridge whose
nonlinear output capacitances make a current-driven (zero-voltage) transition.

Case A of ``test_soft_switching_validation.py`` (feat/core-integration-14t),
copied here with its closed forms: 400 V bus, 10 A pulled out of the switch
node, C_oss(v) = C0 / (1 + v/V0)^m (C0 1 nF, V0 25 V, m 0.5) on each switch.
It ran at b9e46778; trbdf2 aborted at 1 us from be5454f3 and both integrators
at t = 0 from cedac36b (Newton at h_min). Notes:
openspec/changes/update-number-changing-release/fixes/f-s-5.md.
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
FSW = 1e6
I_A = 10.0
DEAD_A = 100e-9
T_TOL = 0.05e-9

mp.mp.dps = 25


# ------------------------------------------------------------------ truths --
def _C(v):
    return mp.mpf(C0) / (1 + mp.mpf(v) / V0) ** M


def _Q(v):
    return mp.quad(_C, [0, v])


def _W(v):
    return mp.quad(lambda u: u * _C(u), [0, v])


def _t_cross_A(level):
    v0 = VDC - I_A / GON
    t_off = 0.5 / FSW - DEAD_A
    return float(t_off + ((_Q(v0) - _Q(level)) + (_Q(VDC - level) - _Q(VDC - v0))) / I_A)


def _e_hard_A(t_pre, t_post):
    va, vb = -(VF + I_A / GON), VDC - I_A / GON
    imp = VDC * (_Q(vb) - _Q(va)) - (_W(VDC - vb) - _W(VDC - va)) - (_W(vb) - _W(va))
    cond = (VF * I_A + I_A ** 2 / GON) * t_pre + I_A ** 2 / GON * t_post
    return float(imp + cond)


# ------------------------------------------------------------ simulations --
def _builder():
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("Vdc", "vdc", "gnd", 0.0, VDC, 0.0, 1.0, 0.0, TR, 0.0)
    b.add_switch("Sh", "vdc", "sw", GON, GOFF)
    b.add_switch("Sl", "sw", "gnd", GON, GOFF)
    b.add_diode("Dh", "sw", "vdc", GON, GOFF, V_f=VF)
    b.add_diode("Dl", "gnd", "sw", GON, GOFF, V_f=VF)
    b.add_nonlinear_capacitor("Ch", "vdc", "sw", C0, V0, M)
    b.add_nonlinear_capacitor("Cl", "sw", "gnd", C0, V0, M)
    b.add_pulse_current_source("I", "gnd", "sw", 0.0, I_A, 0.0, 1.0, 0.0, TR, 0.0)
    return b


def _gates(b):
    return p.make_dead_time_pwm_pair_fn(FSW, 0.5, b.switch_index_of("Sh"),
                                        b.switch_index_of("Sl"), b.graph.num_switches, DEAD_A)


ENGINES = {
    "trbdf2": dict(engine="trbdf2"),
    "esdirk43": dict(engine="trbdf2", integrator="esdirk43"),
}


@functools.lru_cache(maxsize=None)
def _run(engine, t_end=2.2 / FSW, **extra):
    b = _builder()
    fn = _gates(b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=t_end, switch_fn=fn, **ENGINES[engine], **extra)
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


@pytest.fixture(params=list(ENGINES))
def engine(request):
    return request.param


# ---------------------------------------------- the ten cells of case A --
@pytest.mark.parametrize("level", [300.0, 200.0, 100.0])
def test_current_driven_transition_time_is_the_charge_closed_form(engine, level):
    _, _, r = _run(engine)
    t, v = np.asarray(r.times), np.asarray(r.v("sw"))
    t_off = 0.5 / FSW - DEAD_A
    got = _cross(t, v, level, t_off - 1e-9, t_off + DEAD_A)
    assert abs(got - _t_cross_A(level)) <= T_TOL


def test_zvs_fraction_soft_low_side_hard_high_side(engine):
    b, fn, r = _run(engine)
    t, v = np.asarray(r.times), np.asarray(r.v("sw"))
    gate = [fn(x) for x in t]
    hs, ls = b.switch_index_of("Sh"), b.switch_index_of("Sl")
    lo = compute_zvs_fraction([g.get(ls) for g in gate], list(v), threshold_v=5.0)
    hi = compute_zvs_fraction([g.get(hs) for g in gate], list(VDC - v), threshold_v=5.0)
    assert (lo["zvs_fraction"], lo["total_turn_on_events"]) == (1.0, 2.0)
    assert (hi["zvs_fraction"], hi["total_turn_on_events"]) == (0.0, 2.0)


def test_hard_turn_on_energy_is_booked(engine):
    tp, tq = 2e-9, 20e-9
    b, fn, r = _run(engine)
    got = _window_loss(b, fn, r, 1.0 / FSW - tp, 1.0 / FSW + tq)
    assert got == pytest.approx(_e_hard_A(tp, tq), rel=0.05)


# ------------------------------------------------- the same run, other knobs --
@pytest.mark.parametrize("extra", [dict(rtol=1e-4), dict(rtol=1e-8), dict(dt=1e-10)],
                         ids=["rtol1e-4", "rtol1e-8", "dt1e-10"])
def test_case_a_transition_under_other_controls(engine, extra):
    _, _, r = _run(engine, **extra)
    t, v = np.asarray(r.times), np.asarray(r.v("sw"))
    t_off = 0.5 / FSW - DEAD_A
    got = _cross(t, v, 200.0, t_off - 1e-9, t_off + DEAD_A)
    assert abs(got - _t_cross_A(200.0)) <= T_TOL


def test_case_a_short_run_follows_the_bus(engine):
    """50 ns: inside the source ramp, the high side ON (1 mOhm across Ch):
    v(sw) = vdc - 1 mOhm x (the switch current), within 0.1 V of the ramp."""
    _, _, r = _run(engine, t_end=50e-9)
    t, v = np.asarray(r.times), np.asarray(r.v("sw"))
    assert t[-1] == pytest.approx(50e-9)
    assert np.max(np.abs(v - VDC * t / TR)) < 0.1


# ------------------------------------- the reduced case of the t = 0 abort --
@pytest.mark.parametrize("m", [0.0, 0.5])
def test_a_charge_law_capacitor_charged_from_0_v_by_a_ramp_runs(engine, m):
    """400 V ramp behind 1 mOhm into ONE charge-law capacitor from 0 V. The
    first (SDIRK) step's estimate takes it to ~2e-15 s, where the law's
    pow(1 + u, 1 - m) - 1 held Q to C0·V0·eps only: a 1e-8 A Newton floor,
    aborted at h_min on cedac36b. m = 0 is linear: closed form
    v = S·(t - tau·(1 - exp(-t/tau))), tau = R·C0 = 1 ps."""
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("Vdc", "vdc", "gnd", 0.0, VDC, 0.0, 1.0, 0.0, TR, 0.0)
    b.add_resistor("R", "vdc", "sw", 1e-3)
    b.add_nonlinear_capacitor("C", "sw", "gnd", C0, V0, m)
    r = p.simulate(b, t_end=50e-9, **ENGINES[engine])
    t, v = np.asarray(r.times), np.asarray(r.v("sw"))
    assert t[-1] == pytest.approx(50e-9)
    if m == 0.0:
        s, tau = VDC / TR, 1e-3 * C0
        assert np.max(np.abs(v - s * (t - tau * (1 - np.exp(-t / tau))))) < 1e-6 * VDC
    else:   # the node follows the bus within the RC's lag, R·C(v)·S <= 4 mV
        assert np.max(np.abs(v - VDC * t / TR)) < 5e-3


# ------------------------------------- run length as a property (F-S-5 session 2) --
def test_case_a_runs_at_every_run_length(engine):
    """The default h_min is span·1e-11, so the companion floor eps·Q/(γh)
    at a ps edge depends on t_end: Newton cycled there (period 6, 7, 10) and
    the two-step stagnation test declined it on 9 of 30 lengths. 40 lengths
    from a fixed seed in [0.2, 6] us; the full 200 are in the notes."""
    lengths = np.random.default_rng(20261120).uniform(0.2e-6, 6e-6, 40)
    bad = []
    for te in lengths:
        try:
            _run.__wrapped__(engine, t_end=float(te))
        except p.SimulationAborted:
            bad.append(te)
    assert bad == []


# ------------- saturable cut-set opened by a switch (reviewer's case, session 2) --
def _cut_set():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "v", "gnd", 10.0)
    b.add_resistor("R", "v", "a", 1.0)
    b.add_saturable_inductor("L1", "a", "b", 1e-3, 5.0, 1e-4)
    b.add_inductor("L2", "b", "gnd", 1e-3)
    b.add_switch("S", "b", "gnd", 1e3, 1e-9)
    n, k = b.graph.num_switches, b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, t < 2e-4)
        return m
    return b, fn


@pytest.mark.parametrize("cfg", [
    ("esdirk43", {}), ("esdirk43", dict(rtol=1e-8)), ("trbdf2", {}),
    # Session 3: both aborted at 200 us under the single window of 16 ("growing"
    # at omega 0.12 eps); the split windows (DECISIONS 11 amendment) run them.
    ("trbdf2", dict(rtol=1e-8)), ("trbdf2", dict(rtol=1e-9)),
], ids=["esdirk43", "esdirk43-rtol1e-8", "trbdf2", "trbdf2-rtol1e-8", "trbdf2-rtol1e-9"])
def test_saturable_cut_set_opened_by_a_switch(cfg):
    """L1 (atan law) + L2 forced into series at 200 us. Flux conservation:
    lam1(i1-) + L2 i2- = lam1(i+) + L2 i+ with i1- = 1.882674, i2- = 1.9e-4
    gives i+ = 0.908907 A (mpmath); 100 ns later the current has moved by
    < 4e-4 A (di/dt <= 10 V / 2.7 mH). esdirk43 aborted here since cedac36b."""
    eng, kw = cfg
    b, fn = _cut_set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=4e-4, switch_fn=fn, **ENGINES[eng], **kw)
    t = np.asarray(r.times)
    i = np.asarray(r.v("v")) - np.asarray(r.v("a"))
    assert abs(np.interp(2.001e-4, t, i) - 0.908907) < 1e-3


# ------------- reference circuits that lean on the roundoff acceptance (session 3) --
# Every benchmark circuit with more than 50 roundoff accepts on trbdf2 defaults
# (ll13: 359, ll08: 117; no other circuit has more than 50). ll13 ran at
# 14b0c81a and aborted at 41 us under the single window of 16. Accuracy: the
# states at smooth instants against engine='pwl' + discontinuity_restart at a
# small dt (measured: ll13 2.0e-5 of range, ll08 2.3e-6 = the reference's own
# error); the bound is 50x looser.
_HEAVY = {
    "ll13": ("circuits/switching/isolated/ll13_active_clamp_forward.yaml", 2e-9, 4e-6,
             (1e-6, 3e-6), ["Lout", "Lm_core", "Lleak_prim", "Lsec"], ["vout", "clamp_node"]),
    "ll08": ("circuits/switching/buck/ll08_buck_mosfet.yaml", 1e-8, 5e-6,
             (1e-6, 3e-6), ["L1"], ["out"]),
}


# ll13 at trbdf2 rtol 1e-7 and esdirk43 rtol 1e-6 aborted at h_min on 14f (and its
# default on the gate-edge merge): a floating-point fixed point at the dead-time
# spike (3.8e8 V, ||dx|| below one ulp) declined against the start-state row
# scale. DECISIONS 12 of add-newton-roundoff-acceptance; measured 9.4e-6 /
# 1.0e-4 of range against the reference.
@pytest.mark.parametrize(("name", "kw"), [
    ("ll08", {}), ("ll13", {}), ("ll13", {"rtol": 1e-7}),
    ("ll13", {"rtol": 1e-6, "integrator": "esdirk43"})])
def test_roundoff_heavy_reference_circuit_runs_on_trbdf2(name, kw):
    rel, dt_ref, period, offs, inductors, nodes = _HEAVY[name]
    bench = Path(__file__).resolve().parents[2] / "benchmarks"
    sys.path.insert(0, str(bench / "tools"))
    import v1_reader  # noqa: PLC0415
    case = v1_reader.load(bench / rel)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = case.run("trbdf2", **kw)
        ref = case.run("pwl", dt=dt_ref, discontinuity_restart=True)
    t, tr = np.asarray(r.times), np.asarray(ref.times)
    assert t[-1] == pytest.approx(case.t_end, rel=1e-12)
    ts = np.array([n * period + o for n in range(int(case.t_end / period)) for o in offs])
    sig = [(lambda x, k=k: np.asarray(x.i(k))) for k in inductors]
    sig += [(lambda x, k=k: np.asarray(x.v(k))) for k in nodes]
    for f in sig:
        y = f(ref)
        err = np.max(np.abs(np.interp(ts, t, f(r)) - np.interp(ts, tr, y))) / np.ptp(y)
        assert err < 1e-3


# ------------- case A at a tight tolerance: the chatter floor (gate-edge line) --
# Closed form 62.8335 uJ. At rtol 1e-8 the first steps after the 1 us hard
# turn-on (Sh into a conducting Dl across C_oss, tau ~1 ps) exhaust the chatter
# guard on Dl; h is then floored at h_chatter = 2.2e-13 s for 1e3 h_chatter,
# and the transient is crossed by forced accepts (174 on trbdf2; 14f: 4).
# Measured on the gate-edge merge: trbdf2 63.0154 uJ (+0.29 %; 14f 62.8368),
# esdirk43 63.0065 uJ (+0.27 %; 14f 63.0082: the same floor, already there).
# rtol 1e-7, 1e-9 and 1e-10 read the same 63.0154 / 63.008. The defaults
# moved toward the truth (trbdf2 62.8490 -> 62.8386).
# A REGRESSION against 2.1.0 (origin/main ec08e98e, same script): trbdf2 rtol
# 1e-7 / 1e-8 / 1e-9 read 62.8372 / 62.8368 / 62.8367 there (4 forced
# accepts, no chatter break); esdirk43 62.8414 / 63.0082 / 62.8368. Seen on
# the line: the edge's post-edge samples keep Dl ON (mask 1001) for two
# samples, then Dl chatters (2 breaks) and h sits at h_chatter (0.22 ps).
@pytest.mark.xfail(strict=True, reason="open (regression vs 2.1.0): chatter-floor forced "
                   "accepts through the C_oss transient at rtol <= 1e-7 (+0.29 % trbdf2, "
                   "+0.27 % esdirk43; 2.1.0 +0.005 %)")
@pytest.mark.parametrize("eng", list(ENGINES))
def test_hard_turn_on_energy_at_a_tight_tolerance(eng):
    b, fn, r = _run(eng, rtol=1e-8)
    e = _window_loss(b, fn, r, 1e-6 - 2e-9, 1e-6 + 20e-9)
    assert e == pytest.approx(_e_hard_A(2e-9, 20e-9), rel=1e-3)
