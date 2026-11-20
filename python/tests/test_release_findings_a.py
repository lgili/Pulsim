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


def test_f51_pathless_inductor_never_gains_energy_and_is_named():
    """No conduction path (the IGBT blocks reverse current): the 100 A is an impulse.
    Truth: flux conservation in the cut leaves i_L = 0 and removes 1/2 L i0^2 = 5 J.
    Before the fix the restart's stage 2 solved from the pseudo-state x0* = -sqrt(2) x0
    and the IGBT's forward channel kept -141.42 A (energy 5 J -> 10 J, no source in the
    loop). The consistent start x0+ = 2 x1 - S(x1) takes the jump first and records it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _freewheel(**RESTART)
    i = np.asarray(res.i("L"))
    # stored energy never exceeds the initial 5 J (flux cannot be created)
    assert np.max(np.abs(i)) <= 100.0 * (1 + 1e-12), np.max(np.abs(i))
    # the cut-set leaves no path: the flux is gone after the impulse
    assert np.max(np.abs(i[1:])) < 1e-3, i[:4]
    recs = list(res.reinit_records)
    assert len(recs) == 1, recs
    r = recs[0]
    assert r.t == 0.0 and r.kind == "cut_set" and r.states == ["L"]
    assert r.cause == "QL" and "QL" in r.devices
    assert r.dW == pytest.approx(5.0, rel=1e-5)
    assert r.lambda_ == pytest.approx(-0.1, rel=1e-5)   # L * (0 - 100 A) V*s
    with pytest.raises(Exception, match="on_impulse='error'.*'QL'"):
        _freewheel(on_impulse="error", **RESTART)
    with pytest.warns(UserWarning, match="impulsive re-initialisation"):
        _freewheel(on_impulse="warn", **RESTART)


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


def test_f33_flyback_turn_off_does_not_enter_the_body_diode():
    """Truth: the exact engine (LCP mode selection) takes the primary to ~0 at turn-off (the
    leakage current has no path but R_off = 1 GOhm) with a 16.382 A peak. The restart's
    first stage takes it there too; the body diode must not pick up -sqrt(2) times it."""
    ip_exact = _run_flyback(engine="dsed", mode_selection="auto")
    ip = _run_flyback(dt=1e-8, **RESTART)
    assert ip_exact.min() > -1e-6 and ip_exact.max() == pytest.approx(16.382, abs=1e-3)
    assert ip.min() > -1e-3, ip.min()
    assert ip.max() == pytest.approx(ip_exact.max(), rel=2e-3)


def test_f33_flyback_peak_converges_at_second_order():
    """The restart from the consistent state x0+ is second order through the
    impulsive turn-off: peak primary current against the exact engine's
    16.381990162 A (measured 5.1e-6, 1.2e-6 A at dt 10, 5 ns)."""
    peak_exact = 16.381990162
    e = [abs(_run_flyback(dt=dt, **RESTART).max() - peak_exact) for dt in (1e-8, 5e-9)]
    assert e[0] < 1e-5 and e[0] / e[1] > 3.5, e


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


def _psfb_vout(primary, on_grid=False, **kw):
    """The output voltage v(vout) - v(rect_neg). The secondary floats: its common mode is
    set by the diodes' g_off leaks alone (mV, rounding), not a circuit quantity.
    ``on_grid``: every gate edge moved to the grid point the restart rule without the
    split takes it at (the gate read at t_k - dt/100 acts over (t_{k-1}, t_k])."""
    b = _psfb(primary)
    sw = p.make_phase_shift_full_bridge_fn(
        switching_frequency=100e3, phase_shift=1.0, leg_a_hs_idx=0, leg_a_ls_idx=1,
        leg_b_hs_idx=2, leg_b_ls_idx=3, num_switches=b.graph.num_switches, dead_time=100e-9)
    if on_grid:
        dt, sw0 = kw["dt"], sw

        def sw(t):   # an opaque function: the split bisects its changes onto the grid
            return sw0(dt * np.ceil(t / dt - 1e-9) - dt / 100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, t_end=0.2e-3, switch_fn=sw, **kw)
    return float(np.asarray(res.v("vout"))[-1] - np.asarray(res.v("rect_neg"))[-1])


# mpmath truth (benchmarks/tools/exact_truth.py of feat/core-soft-switching-validation, 30
# digits) of the LINEAR primary at the switch_fn's exact gate edges (leg B at T/2pi: off the
# 50 ns grid, so the split's landings), the secondary tied to ground by 1 GOhm: v_out(200 us).
F37_LINEAR_TRUTH = 6.856714473154
# The saturable primary, split, its own limit (dt 50 / 25 / 12.5 / 6.25 ns: 6.8655416568,
# ...6940, ...7045, ...7064; the linear twin's errors against the truth at the same steps
# -7.4e-8, -2.4e-8, -9.7e-9, -2.0e-9). Grounding the secondary (1 kOhm, short) gives the same
# value to 10 digits.
F37_SATURABLE_LIMIT = 6.86554171


@pytest.mark.parametrize("selector", ["legacy", "auto"])
def test_f37_saturable_psfb_runs_under_the_restart_rule(selector):
    """F-3-7: restart stage 1 at a gate edge was solved by Newton in the step-start diode
    mask (D_LS_A OFF): the saturable L_leak cut into g_off, mid_a at -1.26e4 V, the floating
    secondary's iterate bouncing volts at that scale: abort. The edge's mode is now decided
    before the Newton solve (on the stage's linearization at the pre-edge state). 'auto'
    runs a Newton circuit on the settle; 'lcp' refuses it by name."""
    split = dict(RESTART, mode_selection=selector, enable_substep_state_correction=True)
    v_lin = _psfb_vout("linear", dt=50e-9, **split)
    assert v_lin == pytest.approx(F37_LINEAR_TRUTH, abs=2e-7)
    v_sat = _psfb_vout("saturable", dt=50e-9, **split)
    assert v_sat == pytest.approx(F37_SATURABLE_LIMIT, abs=2e-7)
    # without the split leg B's edges act at the grid (another circuit): its reference is
    # the split run of that circuit (the switch_fn's edges moved to the grid points the
    # plain run takes them at; measured 6e-5 apart on the saturable - linear difference,
    # 3e-6 V on v_out). The old pin (the split's own difference, rel 0.05) held only
    # while the plain run took an off-grid edge on the ceil side (0.008827 vs 0.008123).
    plain = dict(RESTART, mode_selection=selector, enable_substep_state_correction=False)
    d = _psfb_vout("saturable", dt=50e-9, **plain) - _psfb_vout("linear", dt=50e-9, **plain)
    d_ref = (_psfb_vout("saturable", True, dt=50e-9, **split)
             - _psfb_vout("linear", True, dt=50e-9, **split))
    assert d == pytest.approx(d_ref, rel=1e-3)


def test_f37_lcp_refuses_the_saturable_psfb_by_name():
    with pytest.raises(p.UnsupportedError, match="Newton device"):
        _psfb_vout("saturable", dt=50e-9, mode_selection="lcp", **RESTART)


# ---- F-3-1: restart + split not rounding-stable at a diode turn-on ----------------------------

def test_f31_split_turn_on_is_rounding_stable():
    """The floating single-phase bridge (V_f = 0.7) with C and R moved by 1..5 ulp: the
    waveform must move by rounding only (restart-less: 2.3e-11 V). The series diode pair
    (D2, D3) conducts together; the split located the event at ONE diode's crossing, a tie
    of leakage and rounding (3.1e-4 V). It now locates the decided mode's commutation on
    the pair's summed signal: 33.7040 ms, the exact engine's 33.704043-33.704054 ms."""
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


# ---- the impulse rule: one criterion, its edges ----------------------------------------------

def _rectifier_from_rest(**kw):
    """A switched-diode half-wave rectifier from rest: C = 100 uF through g_on = 1e3 S
    (tau = 0.1 us << dt) follows the 400 V sine from t = 0. A fast mode TRACKING its
    forcing, not a jump: the probe (two BE stages, sources advanced) extrapolates the
    stages' line back to x0 itself."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "ac", "gnd", 0.0, 400.0, 60.0)
    b.add_diode("D", "ac", "vout", 1e3, 1e-9)
    b.add_resistor("R", "vout", "gnd", 50.0)
    b.add_capacitor("C", "vout", "gnd", 100e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=0.02, dt=1e-4, **kw)


def test_fast_mode_that_tracks_its_source_is_not_an_impulse():
    res = _rectifier_from_rest(**RESTART)
    assert list(res.reinit_records) == []


def _pathless_saturable(**kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S", "in", "a", 1e3, 1e-9)
    b.add_saturable_inductor("Lsat", "a", "gnd", 1e-3, 30.0, 1e-4)
    n, k = b.graph.num_switches, b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, t < 5e-6)
        return m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=1e-5, dt=1e-7, switch_fn=fn, **kw)


def test_a_saturable_core_cut_by_its_switch_is_refused_by_name():
    """The consistent start moves linear C / L histories only; a saturable core's current
    cut into g_off (5e-2 A -> leakage) is refused, naming the core, not integrated from the
    reversed pseudo-state."""
    with pytest.raises(p.UnsupportedError, match="saturable_inductor 'Lsat'") as ei:
        _pathless_saturable(**RESTART)
    assert ei.value.device == "Lsat" and ei.value.feature == "impulse"
    _pathless_saturable(engine="pwl")    # restart-less: the rule is not on


def _switched_junction(t_end, **kw):
    """A 1 nF nonlinear capacitor across a buck switch (R_on 1 mOhm): the turn-on at 10 us
    discharges it in tau = 1 ps << dt = 20 ns (a real impulse); the turn-off at 5 us
    recharges it through the inductor current (a kink, not a jump)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "out", 100e-6)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", 2.0)
    b.add_nonlinear_capacitor("Cj", "vin", "sw", 1e-9, 1.0)
    n, k = b.graph.num_switches, b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, (t % 1e-5) < 5e-6 - 1e-12)
        return m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=t_end, dt=2e-8, switch_fn=fn, **kw)


def test_a_nonlinear_capacitor_shorted_by_its_switch_is_carried_in_charge():
    """The rule reads a nonlinear capacitor's rate before the event from its law
    (i_prev / C(v_prev)): the turn-off kink at 5 us is no record; the 1 ps discharge at
    the 10 us turn-on (refused by name before F-S-6) is carried in charge. Truth: Cj goes
    from 24 V + i_L/g_on (D1 freewheeling) to i_L/g_on (S on), i_L continuous; the moved
    charge is Q(v+) - Q(v-) and the energy the law's int (Q(v) - Q(v-)) dv, with
    Q(v) = 2 C0 V0 (sqrt(1 + v/V0) - 1) (m = 0.5)."""
    assert list(_switched_junction(9e-6, **RESTART).reinit_records) == []
    r = _switched_junction(1.2e-5, **RESTART)
    recs = list(r.reinit_records)
    assert len(recs) == 1 and recs[0].t == pytest.approx(1e-5) and list(recs[0].states) == ["Cj"]
    t = np.asarray(r.times)
    k = int(np.searchsorted(t, 1e-5 - 1e-12))
    i_l = float(np.asarray(r.i("L"))[k])           # the slow state, 1e-3 V of offset
    vm, vp = 24.0 + i_l / 1e3, i_l / 1e3

    def q(v):
        return 2e-9 * (np.sqrt(1.0 + v) - 1.0)

    def iq(v):                                       # int Q dv
        return 2e-9 * ((2.0 / 3.0) * (1.0 + v) ** 1.5 - v)
    e = iq(vp) - iq(vm) - q(vm) * (vp - vm)
    assert recs[0].lambda_ == pytest.approx(q(vp) - q(vm), rel=1e-6)
    assert recs[0].dW == pytest.approx(e, rel=1e-6)
    v_cj = np.asarray(r.v("vin")) - np.asarray(r.v("sw"))
    assert v_cj[k + 1] == pytest.approx(vp, abs=1e-5)
    _switched_junction(1.2e-5, engine="pwl")


def _cut_smaller_than_the_old_rate(i0):
    """1 uH charged at 100 V from i0 until S opens at 0.1 us; the body diode Db (a -> vin) is
    reverse biased by the cut, so no path is left: the truth is i_L -> 0."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 100.0)
    b.add_switch("S", "vin", "a", 1e3, 1e-9)
    b.add_diode("Db", "a", "vin", 1e3, 1e-9)
    b.add_inductor("L", "a", "gnd", 1e-6, i0=i0)
    n, k = b.graph.num_switches, b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, t < 1e-7 - 1e-15)
        return m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=1e-6, dt=1e-7, switch_fn=fn, **RESTART)


def test_a_cut_smaller_than_h_times_the_old_rate_is_an_impulse():
    """The old mode's rate h*V/L = 10 A must not mask the cut: the 8 A cut came back as
    -8.38 A through Db with no record (reviewer A1, third session). The jump test reads no
    rate before the event: d = -8 A, d' = 5e-7 A. Truth: no path, i_L -> 0 by an impulse
    of 1/2*L*i^2 (i = 7.9997 A at the cut)."""
    r = _cut_smaller_than_the_old_rate(-2.0)
    i = np.asarray(r.i("L"))
    assert len(r.reinit_records) == 1 and list(r.reinit_records[0].states) == ["L"]
    assert abs(r.reinit_records[0].dW - 0.5 * 1e-6 * 8.0**2) < 1e-3 * 0.5 * 1e-6 * 64
    assert np.max(np.abs(i[2:])) < 1e-3


def _two_rectifiers(which, dt=1e-4):
    """Two INDEPENDENT half-wave rectifiers, the second's sine 0.1 dt late: their turn-offs
    fall in the same grid step."""
    import math
    b = p.CircuitBuilder()
    for k, dly in (("1", 0.0), ("2", 0.1 * dt)):
        if k not in which:
            continue
        b.add_sine_voltage_source("V" + k, "s" + k, "gnd", 0.0, 100.0, 50.0,
                                  -2 * math.pi * 50 * dly)
        b.add_diode("D" + k, "s" + k, "o" + k, 1e3, 1e-9)
        b.add_capacitor("C" + k, "o" + k, "gnd", 1e-4)
        b.add_resistor("R" + k, "o" + k, "gnd", 100.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=0.04, dt=dt, enable_substep_state_correction=True,
                          **RESTART)


def test_split_locates_independent_diodes_on_their_own_signals():
    """Only diodes at one FLOATING island share a summed signal (the floating bridge of
    F-3-1). Two grounded rectifiers crossing in one step keep their own crossings: the
    first is located where it is alone, and its waveform is the lone run's bit for bit
    (the summed signal put both at a common time between them)."""
    alone, both = _two_rectifiers("1"), _two_rectifiers("12")
    d1 = [e.t_estimated for e in alone.commutation_events]
    b1 = [e.t_estimated for e in both.commutation_events if e.branch_id == 1]
    assert len(d1) == len(b1) >= 3
    np.testing.assert_allclose(b1, d1, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(both.v("o1")), np.asarray(alone.v("o1")))


def _rectifier_and_unrelated_pulse(pulse):
    """The rectifier from rest, plus (pulse=True) a 1 V pulse into its own resistor with its
    edge at 0.4 dt: inside the probe's (t1, t2] (t1 = 0.29 dt, t2 = 0.59 dt)."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "ac", "gnd", 0.0, 400.0, 60.0)
    b.add_diode("D", "ac", "vout", 1e3, 1e-9)
    b.add_resistor("R", "vout", "gnd", 50.0)
    b.add_capacitor("C", "vout", "gnd", 100e-6)
    if pulse:
        b.add_pulse_voltage_source("Vp", "p", "gnd", 0.0, 1.0, t_start=0.4e-4,
                                   pulse_width=1.0, period=10.0, rise_time=0.0, fall_time=0.0)
        b.add_resistor("Rp", "p", "gnd", 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=0.02, dt=1e-4, **RESTART)


def test_a_source_edge_inside_the_probe_is_read_at_its_left_limit():
    """Reviewer A3 (third session): the edge left only the frozen reading, which reads the
    capacitor tracking its sine as a 4.39 V jump (0.96 mJ record at t = 0). The moved
    reading takes the edge's left limit: no record, and the rectifier is the pulse-free
    run's (the pulse drives nothing of it)."""
    with_pulse, alone = _rectifier_and_unrelated_pulse(True), _rectifier_and_unrelated_pulse(False)
    assert list(with_pulse.reinit_records) == []
    np.testing.assert_allclose(np.asarray(with_pulse.v("vout")), np.asarray(alone.v("vout")),
                               rtol=0, atol=1e-9)


def test_the_dcm_buck_has_no_impulse():
    """The DCM buck has no impulsive event: its diode turns the inductor off at ZERO current
    and its 47 uF output capacitor is in series with the inductor at every edge. Two false
    records the probe produced: the capacitor at its extremum at a gate edge (d = 6.4e-7 V
    against a rate after the event of ~0; the curvature term d' = 6.3e-7 V says it is a
    trajectory), and the inductor at each diode turn-off (0.4-1.5 mA: the motion before a
    stop inside stage 1, an unlocated state event: allowance gamma*h*|di/dt(0-)|)."""
    from analytic import BY_ID   # benchmarks/ (conftest puts it on the path)
    c = BY_ID["buck_dcm"]
    b, fn = c.build(1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=c.t_end, dt=c.dt, switch_fn=fn, **RESTART)
    assert list(r.reinit_records) == []


def _freewheel_into(i0, split, L=1e-6, V=100.0, dt=1e-7):
    """L freewheels through D into V from i0: di/dt = -V/L until i = 0 at t_c = i0*L/V,
    where D turns off at ZERO current (no impulse; reviewer, fourth session)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vo", "gnd", V)
    b.add_diode("D", "gnd", "a", 1e3, 1e-9)
    b.add_inductor("L", "a", "vo", L, i0=i0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=2e-6, dt=dt, enable_substep_state_correction=split,
                          **RESTART)


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("i0", [12.0, 15.0, 22.0, 51.5, 55.0])
def test_a_diode_turning_an_inductor_off_at_zero_current_is_not_an_impulse(i0, split):
    """Truth (closed form, R_on = 1 mOhm of the ON diode): i(t) = (i0 + V/R) e^(-Rt/L) - V/R
    until it reaches 0, then 0. The stop inside stage 1 of a window step (t_c = 1.2, 1.5,
    2.2 dt) read as a jump of up to 3.1 uJ; it is the motion gamma*h*V/L of an unlocated
    crossing, not a jump."""
    L, V, R = 1e-6, 100.0, 1e-3
    r = _freewheel_into(i0, split)
    assert list(r.reinit_records) == []
    t, i = np.asarray(r.times), np.asarray(r.i("L"))
    ic = np.maximum((i0 + V / R) * np.exp(-R * t / L) - V / R, 0.0)
    t_c = -L / R * np.log((V / R) / (i0 + V / R))
    far = np.abs(t - t_c) > 1e-7   # the step that holds the crossing is first order
    assert np.max(np.abs(i - ic)[far]) < 1e-6 * i0, np.max(np.abs(i - ic)[far])


@pytest.mark.xfail(strict=True, reason=(
    "release group A, open: the stop lies inside stage 1 of the FIRST step (t_c = 0.25 dt); "
    "the allowance's rate before the event is the history slot, which is 0 at a start from "
    "initial conditions (no trajectory before t = 0), so the stop reads as a jump of "
    "1/2 L i0^2 = 3.1 uJ. The pre-flip stage-1 solve is not that rate at a time event (it "
    "holds the event's own cut: F-3-3). Needs the start's rate in its initial mode."))
def test_a_stop_within_gamma_h_of_a_start_from_initial_conditions_is_not_an_impulse():
    assert list(_freewheel_into(2.5, False).reinit_records) == []


def _buck_freewheel(saturable, split, edge_at=None):
    """S charges 1 uH from 100 V into Vo = 49.08 V until 0.4 us; D then freewheels it to 0 at
    t_c = 0.815 us (reviewer, fourth session). The saturable core has L_sat = L0: linear.
    edge_at: an UNRELATED switch S2 (own source, own load) closes then, a time event whose
    restart step holds the stop (reviewer, fifth session)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 100.0)
    b.add_voltage_source("Vo", "vo", "gnd", 49.08)
    b.add_switch("S", "vin", "a", 1e3, 1e-9)
    b.add_diode("D", "gnd", "a", 1e3, 1e-9)
    if saturable:
        b.add_saturable_inductor("L", "a", "vo", 1e-6, 1e3, 1e-6)
    else:
        b.add_inductor("L", "a", "vo", 1e-6)
    b.add_voltage_source("V2", "v2", "gnd", 10.0)
    b.add_switch("S2", "v2", "q", 1e3, 1e-9)
    b.add_resistor("R2", "q", "gnd", 10.0)
    n, k, k2 = b.graph.num_switches, b.switch_index_of("S"), b.switch_index_of("S2")

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, t < 0.4e-6 - 1e-15)
        m.set(k2, edge_at is not None and t >= edge_at - 1e-15)
        return m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=1.2e-6, dt=1e-7, switch_fn=fn,
                          enable_substep_state_correction=split, **RESTART)


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("saturable", [False, True])
@pytest.mark.parametrize("edge_at", [None, 0.8e-6])
def test_a_dcm_freewheel_is_neither_recorded_nor_refused(saturable, split, edge_at):
    """Truth (closed form, R_on = 1 mOhm neglected: < 1e-3 relative): i rises at
    (100 - 49.08)/L to 20.368 A at 0.4 us, falls at 49.08/L to 0 at 0.815 us and stays
    there. The saturable core (linear here) was refused by name at the turn-off."""
    r = _buck_freewheel(saturable, split, edge_at)
    assert list(r.reinit_records) == []
    t, i = np.asarray(r.times), np.asarray(r.i("L"))
    ip = 50.92e6 * 0.4e-6
    ic = np.where(t <= 0.4e-6, 50.92e6 * t, np.maximum(ip - 49.08e6 * (t - 0.4e-6), 0.0))
    far = np.abs(t - (0.4e-6 + ip / 49.08e6)) > 1e-7
    np.testing.assert_allclose(i[far], ic[far], rtol=0, atol=2e-3 * ip)


def _igbt_gate_fall(t_off, fall, split=False, dt=1e-9):
    """F-5-1's IGBT (100 A in 1 mH) with its free-wheeling diode, which cannot carry this
    current's direction: the gate falls 15 -> 0 V at t_off in `fall` (a continuous source,
    no jump to trigger a restart)."""
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("Vg", "g", "gnd", 15.0, 0.0, t_start=t_off, pulse_width=1.0,
                               period=10.0, rise_time=fall, fall_time=0.0)
    b.add_inductor("L", "sw", "gnd", 1e-3, i0=-100.0)
    b.add_igbt_level1("QL", "sw", "gnd", "g", 1.5, 0.05, 5.0, with_fwd=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=12e-9, dt=dt, enable_substep_state_correction=split,
                          **RESTART)


@pytest.mark.parametrize("split", [False, True])
def test_a_gate_fall_inside_a_restart_window_step_is_an_impulse(split):
    """The 1 ps gate fall lands at 1.02 ns, inside window step 2 of the first step's
    restart (no trigger, no diode bit): it was not probed and i_L went -100 -> +140.6 A
    through the FWD. Truth: no path for i_L < 0, so i_L -> 0 by an impulse of
    1/2 L i^2 = 5 J (the IGBT's drop moves i by < 1e-5 A before the fall)."""
    r = _igbt_gate_fall(1.02e-9, 1e-12, split)
    t, i = np.asarray(r.times), np.asarray(r.i("L"))
    recs = list(r.reinit_records)
    assert len(recs) == 1 and list(recs[0].states) == ["L"], recs
    assert 1e-9 <= recs[0].t <= 1.02e-9
    assert recs[0].dW == pytest.approx(5.0, rel=1e-5)
    assert np.max(np.abs(i[t > 2e-9])) < 1e-2, i


def _snubber(Rs, Cs=1e-9, dt=1e-7, Rs2=None):
    """100 V, S vin -> a (R_on 1 mOhm), load 10 Ohm a -> gnd, snubber Rs + Cs across S; S
    off at 0.5 us, on again at 1.0 us (a gate edge on the grid: a TIME event)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 100.0)
    b.add_switch("S", "vin", "a", 1e3, 1e-9)
    b.add_resistor("Rl", "a", "gnd", 10.0)
    b.add_resistor("Rs", "vin", "m", Rs)
    b.add_capacitor("Cs", "m", "a", Cs)
    if Rs2 is not None:   # a second snubber across S
        b.add_resistor("Rs2", "vin", "m2", Rs2)
        b.add_capacitor("Cs2", "m2", "a", Cs)
    n, k = b.graph.num_switches, b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, not (5e-7 - 1e-15 <= t < 1e-6 - 1e-15))
        return m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=2e-6, dt=dt, switch_fn=fn, **RESTART)


def test_a_snubber_faster_than_the_grid_records_the_whole_jump():
    """At the 1 us turn-on Cs (charged to J above its new equilibrium) discharges through
    Rs + R_on||Rl, tau' = 1.001 ns << dt = 100 ns: a jump on the grid's scale. Closed form
    on that one linear mode: the probe's d = -J (1 - a)^2 and d' = a d, a = 1/(1 + gamma
    h/tau'); the rule takes the whole jump J = d/(1 - a)^2, so the record is the energy
    the discharge dissipates in Rs, 1/2 Cs J^2 (5.04 uJ), and Cs starts the step at its
    equilibrium. (Recording d alone gave (1-a)^4 of it, 4.40 uJ, and left the rest to
    stage 2's pseudo-state.)"""
    Cs, Ron, Rl, V = 1e-9, 1e-3, 10.0, 100.0
    r = _snubber(1.0)
    t, v = np.asarray(r.times), np.asarray(r.v("m")) - np.asarray(r.v("a"))
    k = int(np.argmin(np.abs(t - 1e-6)))
    v_on = V * Ron / (Ron + Rl)   # the ON equilibrium
    J = v[k] - v_on
    recs = [x for x in r.reinit_records if x.t == pytest.approx(1e-6)]
    assert len(recs) == 1 and list(recs[0].states) == ["Cs"], list(r.reinit_records)
    assert recs[0].lambda_ / Cs == pytest.approx(-J, rel=1e-6)
    assert recs[0].dW == pytest.approx(0.5 * Cs * J**2, rel=1e-6)
    assert abs(v[k + 1] - v_on) < 1e-6 * J


def test_a_snubber_the_grid_resolves_is_not_an_impulse():
    """tau = Rs Cs = dt: a mode the grid resolves (r = gamma h / tau = 0.29 < 1/gamma)."""
    assert list(_snubber(100.0).reinit_records) == []


def test_two_snubbers_each_take_their_whole_jump():
    """A second snubber (Rs2 = 5 Ohm, tau' = 5 ns, a = 0.146) beside the first: each state
    takes its own J = d/(1 - a)^2. One a for the event (the dominant state's) recorded
    80 % of the energy and left Cs2 at -3.4 V after the step (truth 0.01 V)."""
    Cs, Ron, Rl, V = 1e-9, 1e-3, 10.0, 100.0
    r = _snubber(1.0, Rs2=5.0)
    t = np.asarray(r.times)
    k = int(np.argmin(np.abs(t - 1e-6)))
    v_on = V * Ron / (Ron + Rl)
    J = [np.asarray(r.v(m))[k] - np.asarray(r.v("a"))[k] - v_on for m in ("m", "m2")]
    recs = [x for x in r.reinit_records if x.t == pytest.approx(1e-6)]
    assert len(recs) == 1 and sorted(recs[0].states) == ["Cs", "Cs2"], list(r.reinit_records)
    assert recs[0].dW == pytest.approx(0.5 * Cs * (J[0] ** 2 + J[1] ** 2), rel=1e-4)
    for m, j in zip(("m", "m2"), J):
        v = np.asarray(r.v(m)) - np.asarray(r.v("a"))
        assert abs(v[k + 1] - v_on) < 1e-4 * j, (m, v[k + 1])
