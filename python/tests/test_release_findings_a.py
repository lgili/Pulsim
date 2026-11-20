"""Release findings, group A: impulsive events under the restart rule.

Each test reproduces a finding of the number-changing release on the PR line with the
release behaviour turned on EXPLICITLY (``discontinuity_restart=True`` on the fixed
engine; ``engine='dsed'`` for the exact engine) and checks it against an independent
truth (flux / charge conservation, the exact engine, a fine fixed step).
Notes and traces: openspec/changes/update-number-changing-release/fixes/group-a.md.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p

RESTART = {"engine": "pwl", "discontinuity_restart": True}


# ---- F-5-1: a pathless inductor under the restart rule ---------------------------------------

def _freewheel(**kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vg", "g", "gnd", 15.0)
    b.add_inductor("L", "sw", "gnd", 1e-3, i0=100.0)
    b.add_igbt_level1("QL", "sw", "gnd", "g", 1.5, 0.05, 5.0, with_fwd=False)
    return p.simulate(b, t_end=1e-6, dt=1e-9, **kw)


@pytest.mark.xfail(strict=True, reason=(
    "release finding F-5-1: traced - stage 1 of the restart's SDIRK(2,2) relaxes i_L 100 -> 0 "
    "A (v_sw = -3.41e8 V); stage 2 starts from x0* = x0 + ((1-g)/g)(x1-x0) = -141.42 A, where "
    "the SMOOTH IGBT's forward channel conducts, so stage 2 keeps it (-141.42 A, v_sw 8.57 V). "
    "The stage-2 mode check that fixes F-3-3 reads diode bits; a Newton device has none. "
    "Needs a region test for Newton devices (or the BE re-take on a stage-1 impulse) and the "
    "stage-1 voltage reported to the voltage-sanity guard"))
def test_f51_pathless_inductor_never_gains_energy_and_is_named():
    """No conduction path: the 100 A is an impulse (the IGBT blocks reverse current). The
    restart must not turn it into -141 A (energy 5 J -> 10 J, no source in the loop), and the
    voltage-sanity guard must name the open path, as it does restart-less."""
    with pytest.warns(UserWarning, match="conduction path opens"):
        res = _freewheel(**RESTART)
    i = np.asarray(res.i("L"))
    # stored energy never exceeds the initial 5 J (flux cannot be created)
    assert np.max(np.abs(i)) <= 100.0 * (1 + 1e-12), np.max(np.abs(i))
    # the cut-set leaves no path: the flux is gone after the impulse
    assert np.max(np.abs(i[1:])) < 1e-3, i[:4]


# ---- F-3-3: the flyback's MOSFET turned off into its body diode ------------------------------

L_M = 111.4e-6
N_RATIO = 13 / 25


def _flyback(R_L=1.0, L_leak_p=5e-6):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 48.0)
    b.add_mosfet("Q1", "sw", "gnd", R_on=1e-2, R_off=1e9)
    L_p = L_leak_p + L_M
    L_s = N_RATIO * N_RATIO * L_M
    b.add_transformer("T1", "vin", "sw", "sec_neg", "sec_anode", L_p=L_p, L_s=L_s,
                      k=N_RATIO * L_M / np.sqrt(L_p * L_s))
    b.add_diode("D1", "sec_anode", "vout", 1e3, 1e-9, V_th=0.7)
    b.add_capacitor("Cout", "vout", "sec_neg", 100e-6)
    b.add_resistor("R_L", "vout", "sec_neg", R_L)
    b.add_resistor("Rgnd", "sec_neg", "gnd", 1e-6)
    return b


def _run_flyback(t_end=1e-3, **kw):
    b = _flyback()
    pwm = p.make_pwm_switch_fn(frequency=100e3, duty=0.5, switch_idx=0,
                               num_switches=b.graph.num_switches, phase=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, t_end=t_end, switch_fn=pwm, **kw)
    t = np.asarray(res.times)
    m = t > 0.8 * t_end
    return np.asarray(res.i("T1.p"))[m]


@pytest.mark.xfail(strict=True, reason=(
    "release finding F-3-3: traced - at the turn-off stage 1 of the restart's SDIRK(2,2) "
    "relaxes i_p to -7e-6 A (v_sw 3567 V, D1 on); stage 2 solves from x0* = -sqrt(2) x0 and "
    "turns the body diode ON (off at x0 and in stage 1), keeping the reversed current. "
    "run_restart's stage-2 check excludes a fresh stage-2 turn-on. Including it (BE re-take "
    "from x1) fixes this (peak 16.38199 vs exact 16.381990, 2nd order) but re-breaks the zero "
    "start of a CCM buck (test_buck_ccm_stays_second_order_to_t_over_1600: BE kept, a "
    "dt-independent 7e-10 A). Needs a test that tells the pseudo-state's turn-on from a real "
    "one, e.g. include the fresh turn-on only when stage 1 itself commutated (iters_1 > 1, as "
    "at this turn-off) - unmeasured"))
def test_f33_flyback_turn_off_does_not_enter_the_body_diode():
    """Truth: the exact engine (LCP mode selection) takes the primary to ~0 at turn-off (the
    leakage current has no path but R_off = 1 GOhm) with a 16.382 A peak. The restart's
    first stage takes it there too; the body diode must not pick up -sqrt(2) times it."""
    ip_exact = _run_flyback(engine="dsed", mode_selection="auto")
    ip = _run_flyback(dt=1e-8, **RESTART)
    assert ip_exact.min() > -1e-6 and ip_exact.max() == pytest.approx(16.382, abs=1e-3)
    assert ip.min() > -1e-3, ip.min()
    assert ip.max() == pytest.approx(ip_exact.max(), rel=2e-3)


# ---- F-5-10: the exact engine relaxes an interrupted inductor current silently ---------------

def test_f510_exact_engine_names_the_interrupted_current():
    """The switch opens 2.4 A of inductor current into g_off = 1 nS. The exact engine resolves
    that mode exactly (tau = L g_off = 1 ps): its right-limit state at each turn-off holds
    v_sw = i_L / g_off = 2.4e9 V, and the sampled trace (48 V, 4.8e-8 A after) is right. The
    energy 1/2 L i^2 = 2.88 mJ is dissipated in g_off inside that picosecond, so nothing is
    dropped; but the 2.4 GV spike lives ONLY in the right-limit samples, which the
    voltage-sanity guard did not read: no warning. The guard must name it as it does on
    the other engines."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 48.0)
    b.add_inductor("L", "in", "sw", 1e-3)
    b.add_switch("S", "sw", "gnd", 1e3, 1e-9)

    def sf(t):
        m = p.SwitchStateMask(1)
        m.set(0, (t % 100e-6) < 50e-6)
        return m

    with pytest.warns(UserWarning, match="largest voltage"):
        res = p.simulate(b, t_end=3e-4, switch_fn=sf, engine="dsed")
    # the closed form at the turn-off: v_sw(0+) = i_L(0-) / g_off, i_L continuous
    rl = np.asarray(res.right_limit_index)
    t = np.asarray(res.times)
    k = int(rl[np.argmin(np.abs(t[rl] - 50e-6))])
    i_cut = float(np.asarray(res.i("L"))[k])
    v_rl = float(np.asarray(res.right_limit_states)[list(rl).index(k), 1])
    assert v_rl == pytest.approx(i_cut / 1e-9, rel=1e-9)
    assert float(np.asarray(res.i("L"))[k + 1]) == pytest.approx(48.0 * 1e-9, rel=1e-6)


# ---- F-3-7: saturable-primary PSFB aborts under the restart rule ------------------------------

def _psfb(primary):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vbus", "vbus", "gnd", 100.0)
    for name, frm, to in [("HS_A", "vbus", "mid_a"), ("LS_A", "mid_a", "gnd"),
                          ("HS_B", "vbus", "mid_b"), ("LS_B", "mid_b", "gnd")]:
        b.add_switch(name, frm, to, g_on=1e3, g_off=1e-9)
    for name, a, c in [("D_HS_A", "mid_a", "vbus"), ("D_LS_A", "gnd", "mid_a"),
                       ("D_HS_B", "mid_b", "vbus"), ("D_LS_B", "gnd", "mid_b")]:
        b.add_diode(name, a, c, 1e3, 1e-9)
    if primary == "saturable":
        b.add_saturable_inductor("L_leak", "mid_a", "pri_pos", 37.7e-6, 30.0, 1e-6)
    else:
        b.add_inductor("L_leak", "mid_a", "pri_pos", 37.7e-6)
    b.add_transformer("T1", p_from="pri_pos", p_to="mid_b", s_from="sec_pos", s_to="sec_neg",
                      L_p=40e-6, L_s=10e-6, k=0.99)
    for name, a, c in [("D1", "sec_pos", "rect_pos"), ("D2", "sec_neg", "rect_pos"),
                       ("D3", "rect_neg", "sec_pos"), ("D4", "rect_neg", "sec_neg")]:
        b.add_diode(name, a, c, 1e3, 1e-9)
    b.add_inductor("L_out", "rect_pos", "vout", 100e-6)
    b.add_capacitor("C_out", "vout", "rect_neg", 47e-6)
    b.add_resistor("R_L", "vout", "rect_neg", 10.0)
    return b


def _psfb_vout(primary, **kw):
    b = _psfb(primary)
    sw = p.make_phase_shift_full_bridge_fn(
        switching_frequency=100e3, phase_shift=1.0, leg_a_hs_idx=0, leg_a_ls_idx=1,
        leg_b_hs_idx=2, leg_b_ls_idx=3, num_switches=b.graph.num_switches, dead_time=100e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, t_end=0.2e-3, switch_fn=sw, **kw)
    return float(np.asarray(res.v("vout"))[-1])


@pytest.mark.xfail(strict=True, reason=(
    "release finding F-3-7: traced - NOT the stage-2 pseudo-state: the restart step at "
    "t = 21.5 us (a gate edge, rectifier bits 00001001) fails in STAGE 1's Newton solve at "
    "every halving down to dt/64 (diode-mask chatter inside Newton with the saturable "
    "primary, ||dx|| 1.1e4 V at vout); the restart-less trapezoid takes the same step. Needs "
    "the Newton loop's diode re-decision fixed for the backward-Euler stage"))
def test_f37_saturable_psfb_runs_under_the_restart_rule():
    """Below its knee the saturable primary is the linear one: the restart rule must run it
    and land with the linear primary (restart-less: 6.913 V vs 6.857 V linear)."""
    v_lin = _psfb_vout("linear", dt=50e-9, **RESTART)
    v_sat = _psfb_vout("saturable", dt=50e-9, **RESTART)
    assert v_sat == pytest.approx(v_lin, rel=0.01), (v_sat, v_lin)


# ---- F-3-1: restart + split not rounding-stable at a diode turn-on ----------------------------

@pytest.mark.xfail(strict=True, reason=(
    "release finding F-3-1: traced - the floating bridge's diode PAIR (D2, D3) turns on "
    "together, but the split locates the event at the FIRST single diode's interpolated "
    "crossing; which diode crosses first, and whether the second lands within snap of it, "
    "is a tie decided by leakage and rounding (k=0: t_d = 33.70395 ms, one diode, the other "
    "flips in stage 1; k=2: t_d = 33.70405 ms, both) -> 14.299 vs 14.192 A, 2.15e-4 V. Not "
    "an impulse. Needs the split to locate the commutation of the decided post-event mode"))
def test_f31_split_turn_on_is_rounding_stable():
    """The floating single-phase bridge (V_f = 0.7) with C and R moved by 1..5 ulp: the
    waveform must move by rounding only (restart-less: 2.3e-11 V)."""
    from tests.oracles import lcp_pulsim_bridge as B
    f = B.FAMILIES["bridge1_vf07"]

    def run(k):
        parts = []
        for kind, *a in f.parts:
            if kind in ("C", "R") and k:
                a = list(a)
                for _ in range(k):
                    a[3] = float(np.nextafter(a[3], np.inf))
            parts.append((kind, *a))
        b = B.Family("bridge1_vf07", parts, f.run).builder(0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = p.simulate(b, t_end=0.04, dt=1e-5, enable_substep_state_correction=True,
                           **RESTART)
        return np.asarray(r.v("p")) - np.asarray(r.v("n"))

    w0 = run(0)
    for k in (1, 2, 3, 4, 5):
        assert np.max(np.abs(run(k) - w0)) <= 1e-9, k
