"""The fixed engine across discontinuities: the restart rule.

`engine='pwl'` is a trapezoidal companion. After a mask change, a
source jump or an inconsistent start its derivative slots (a
capacitor's i_prev, an inductor's v_prev) describe a circuit that no
longer exists, and the trapezoid averages that stale derivative with
the new one: every edge lands half a step off (FIRST order) and every
algebraic quantity alternates at the Nyquist frequency, undamped.

`simulate(discontinuity_restart=True)` (add-discontinuity-restart-rule)
takes the step that contains a discontinuity and the next two as
Alexander SDIRK(2,2) steps (gamma = 1 - 1/sqrt2: L-stable, second
order; each stage a backward-Euler-like solve on the factor of
2*gamma*dt), samples the gate mask at the step START and reads a
source's left limit on a jump at the step end. It replaced a pair of
backward-Euler half-steps that drained resonant modes by (wh)^2/2 per
event (the tank and LC gates below). Each test below pins BOTH sides: the
engine without the rule (the measured "before", default off) and with
it (the target, against an exact reference). Removing the rule fails
every `on` assertion.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import brentq

import pulsim as p

ON = dict(discontinuity_restart=True)
GON, GOFF = 1e3, 1e-9


def _mask_fn(b, name, on):
    n, k = b.graph.num_switches, b.switch_index_of(name)

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, bool(on(t)))
        return m
    return fn


def _factors(errs):
    return [abs(a / b) for a, b in zip(errs, errs[1:])]


# --- RL behind a switch that closes on a grid point -------------------------
V, R, L = 10.0, 1.0, 1e-3
TAU = L / (R + 1 / GON)


def _rl_edge(dt, **kw):
    te = round(1e-3 / dt) * dt
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", V)
    b.add_switch("S", "in", "a", GON, GOFF)
    b.add_resistor("R", "a", "m", R)
    b.add_inductor("L", "m", "gnd", L)
    # right-continuous: closed for t >= te
    res = p.simulate(b, t_end=te + 0.6e-3 + 1e-12, dt=dt, engine="pwl",
                     switch_fn=_mask_fn(b, "S", lambda t: t >= te - 1e-15), **kw)
    t, i = np.asarray(res.times), np.asarray(res.i("L"))
    j = int(np.argmin(abs(t - (te + 0.5e-3))))
    ex = V / (R + 1 / GON) * (1 - math.exp(-(t[j] - te) / TAU))
    slope = V / (R + 1 / GON) / TAU * math.exp(-(t[j] - te) / TAU)
    return res, te, i[j] - ex, (i[j] - ex) / slope / dt


def test_gate_edge_on_the_grid_is_second_order():
    on = [_rl_edge(dt, **ON)[2] for dt in (4e-6, 2e-6, 1e-6)]
    assert abs(on[1]) <= 2e-6, on                 # engine: +1.012e-06
    assert all(f >= 3.7 for f in _factors(on)), on
    # Without the rule: the edge lands half a step off, first order.
    _, _, err, shift = _rl_edge(2e-6)
    assert err == pytest.approx(6.057e-3, rel=1e-3)
    assert shift == pytest.approx(0.500, abs=2e-3)


def test_a_right_continuous_edge_takes_effect_at_its_grid_point():
    res, te, err, shift = _rl_edge(2e-6, **ON)
    t = np.asarray(res.times)
    sw = np.asarray(res.switch_states)[:, 0].astype(bool)
    # The step (te, te + dt] is the first one stamped closed (the gate
    # is sampled at the step START, switch_fn(te)).
    first_closed = int(np.argmax(sw[1:])) + 1
    assert t[first_closed] == pytest.approx(te + 2e-6, abs=1e-12)
    assert abs(shift) < 0.01


# --- buck converter against the exact piecewise-LTI solution ----------------
VIN, LB, CB, TB, T0 = 24.0, 100e-6, 47e-6, 1e-5, 2e-6


def _prop(A, B, x, dt):
    M = np.zeros((3, 3))
    M[:2, :2], M[:2, 2] = A, B
    E = expm(M * dt)
    return E[:2, :2] @ x + E[:2, 2]


def _buck_exact(Rl, D, nper):
    """Exact state (i_L, v_C) of the same PWL circuit: each linear
    segment propagated by expm, the diode event located by brentq."""
    def AB(gs, gd):
        k = 1 / (gs + gd)
        return (np.array([[-k / LB, -1 / LB], [1 / CB, -1 / (Rl * CB)]]),
                np.array([gs * VIN * k / LB, 0.0]))

    def seg(x, rem, gs):
        gd = GON if -((gs * VIN - x[0]) / (gs + GOFF)) >= 0 else GOFF
        A, B = AB(gs, gd)
        f = lambda h: -((gs * VIN - _prop(A, B, x, h)[0]) / (gs + gd))  # noqa: E731
        if f(1e-15) * f(rem) < 0:
            h = brentq(f, 1e-15, rem, xtol=1e-18, rtol=1e-15)
            A2, B2 = AB(gs, GOFF if gd == GON else GON)
            return _prop(A2, B2, _prop(A, B, x, h), rem - h)
        return _prop(A, B, x, rem)
    x = np.zeros(2)
    for _ in range(nper):
        x = seg(x, D * TB, GON)
        x = seg(x, (1 - D) * TB, GOFF)
    return x


def _buck(Rl, D, N, nper=40, **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", VIN)
    b.add_switch("S", "vin", "sw", GON, GOFF)
    b.add_diode("D1", "gnd", "sw", GON, GOFF)
    b.add_inductor("L", "sw", "vout", LB)
    b.add_capacitor("C", "vout", "gnd", CB)
    b.add_resistor("RL", "vout", "gnd", Rl)

    def on(t):
        ph = (t - T0) / TB
        return t >= T0 - 1e-13 and ph - math.floor(ph + 1e-9) < D - 1e-9
    return p.simulate(b, t_end=T0 + nper * TB + 1e-12, dt=TB / N, engine="pwl",
                      switch_fn=_mask_fn(b, "S", on), **kw)


def _buck_err(Rl, D, N, **kw):
    res = _buck(Rl, D, N, **kw)
    t = np.asarray(res.times)
    j = int(np.argmin(abs(t - (T0 + 40 * TB))))
    return float(np.asarray(res.i("L"))[j] - _buck_exact(Rl, round(D * N) / N, 40)[0])


def test_buck_ccm_converges_at_second_order():
    # (T/400 and T/800 reach the reference's own rounding, ~1e-9.)
    on = [_buck_err(1.0, 0.5, N, **ON) for N in (50, 100, 200)]
    assert abs(on[0]) <= 2.5e-7, on               # engine: +1.17e-07
    assert all(f >= 3.8 for f in _factors(on)), on
    off = [_buck_err(1.0, 0.5, N) for N in (50, 100)]
    assert off[0] == pytest.approx(1.18e-2, rel=0.02)   # first order
    assert 1.9 < off[0] / off[1] < 2.1


def test_freewheel_diode_picks_up_the_current_in_the_first_stage():
    """The SDIRK step is not atomic for the diode event iteration: each
    stage re-decides. Decided once per step instead (stage 1 with the
    old bits), the turn-off current has no path in stage 1 and the buck
    collapses (measured: v_out 0.83 V instead of 13.3 V in DCM)."""
    res = _buck(1.0, 0.5, 100, **ON)
    v_end = float(np.asarray(res.v("vout"))[-1])
    assert v_end == pytest.approx(_buck_exact(1.0, 0.5, 40)[1], abs=1e-4)
    assert v_end == pytest.approx(12.229, abs=2e-3)


def _dcm_idle(N, **kw):
    res = _buck(50.0, 0.3, N, **kw)
    t = np.asarray(res.times)
    vsw, vo = np.asarray(res.v("sw")), np.asarray(res.v("vout"))
    m = t >= t[-1] - TB + 1e-12
    tt = (t[m] - t[m][0]) / TB
    idle = (tt > 0.75) & (tt < 0.98)
    return vsw[m][idle], float(vo[m][idle].mean())


@pytest.mark.parametrize("N", [100, 400])
def test_no_alternation_in_the_idle_interval_of_a_dcm_buck(N):
    vsw, vo = _dcm_idle(N, **ON)
    assert np.abs(vsw - vo).max() <= 0.01 * vo        # engine: 0.05 %
    vsw_off, vo_off = _dcm_idle(N)
    assert vsw_off.min() < 1.0 and vsw_off.max() > 25.0   # 0.38 / 26.27 V


def test_dcm_buck_output_is_second_order():
    errs = []
    for N in (50, 100, 200):
        res = _buck(50.0, 0.3, N, **ON)
        t = np.asarray(res.times)
        j = int(np.argmin(abs(t - (T0 + 40 * TB))))
        errs.append(float(np.asarray(res.v("vout"))[j])
                    - _buck_exact(50.0, round(0.3 * N) / N, 40)[1])
    assert abs(errs[0]) < 6e-5, errs                  # engine: +2.93e-05
    assert all(f >= 3.8 for f in _factors(errs)), errs


# --- hard-switched capacitor --------------------------------------------------
def _caploop(dt, **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 100.0)
    b.add_resistor("R", "in", "sw", 10.0)
    b.add_switch("S", "sw", "gnd", GON, GOFF)
    b.add_capacitor("C", "sw", "gnd", 1e-9)
    on = lambda t: (t / TB) - math.floor(t / TB + 1e-9) >= 0.5 - 1e-9  # noqa: E731
    res = p.simulate(b, t_end=20 * TB + 1e-12, dt=dt, engine="pwl",
                     switch_fn=_mask_fn(b, "S", on), **kw)
    t = np.asarray(res.times)
    i_s = np.asarray(res.v("sw")) * GON
    ph = (t / TB) % 1.0
    sel = (ph > 0.52) & (ph < 0.98) & (t > 10 * TB)
    P = np.trapezoid(np.where(sel, i_s ** 2 / GON, 0.0), t) / (t[-1] - 10 * TB)
    return i_s[sel], float(P)


def test_hard_switched_capacitor_current_does_not_ring():
    exact = 100.0 / (10.0 + 1 / GON)                       # 9.999 A
    p_exact = exact ** 2 / GON * 0.46                     # this window: 0.04599 W
    i_on, P = _caploop(1e-8, **ON)
    assert np.abs(i_on - exact).max() <= 0.01, (i_on.min(), i_on.max())
    assert P == pytest.approx(p_exact, rel=0.02)
    i_off, P_off = _caploop(1e-8)
    assert i_off.min() < -9.0 and i_off.max() > 29.0      # -9.8 / +29.8 A
    assert P_off > 4 * p_exact                            # 0.197 W


def test_hard_switched_capacitor_at_a_finer_step():
    """The unresolved 1 ps mode is damped by the three L-stable steps
    (the BE pair it replaced left 0.18 A of it ringing at 1 ns)."""
    exact = 100.0 / (10.0 + 1 / GON)
    i_on, P = _caploop(1e-9, **ON)
    assert np.abs(i_on - exact).max() <= 0.02, (i_on.min(), i_on.max())   # 6e-3 A
    assert P == pytest.approx(exact ** 2 / GON * 0.46, rel=0.02)          # 0.0460 W
    i_off, _ = _caploop(1e-9)
    assert np.abs(i_off - exact).max() > 50.0             # -79 / +99 A


# --- source jump on the grid ---------------------------------------------------
def _rc_pulse(dt, te=2e-6, tau=1e-6, **kw):
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "in", "gnd", 0.0, 10.0, te, 50e-6, 0.0, 0.0, 0.0)
    b.add_resistor("R", "in", "a", 1e3)
    b.add_capacitor("C", "a", "gnd", tau / 1e3)
    res = p.simulate(b, t_end=te + 4 * tau + 1e-12, dt=dt, engine="pwl", **kw)
    t, v = np.asarray(res.times), np.asarray(res.v("a"))
    ex = np.where(t > te, 10 * (1 - np.exp(-(t - te) / tau)), 0.0)
    return float(np.abs(v - ex).max())


def test_source_step_on_the_grid_is_second_order():
    on = [_rc_pulse(1e-6 / n, **ON) for n in (10, 20, 40)]
    assert on[0] <= 5e-3, on                          # engine: 2.63e-03
    assert all(f >= 3.5 for f in _factors(on)), on
    assert _rc_pulse(1e-7) == pytest.approx(0.476, rel=0.01)


def test_a_pwm_source_jump_restarts_and_reads_its_left_limit():
    """A PWM voltage source's edges are jumps too (period start and duty
    edge): every one restarts, and nothing else does."""
    b = p.CircuitBuilder()
    b.add_pwm_voltage_source("V", "in", "gnd", 10.0, 0.0, 100e3, 0.25)
    b.add_resistor("R", "in", "a", 1e3)
    b.add_capacitor("C", "a", "gnd", 1e-9)
    res = p.simulate(b, t_end=10 * TB + 1e-12, dt=TB / 100, engine="pwl", **ON)
    assert res.n_restart_steps == 60          # 3 x (2 edges x 10 periods, t = 0 among them)


# --- what does NOT restart ------------------------------------------------------
def test_a_consistent_dc_start_is_untouched():
    def build():
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "s", "gnd", 5.0)
        b.add_resistor("R", "s", "a", 1e3)
        b.add_capacitor("C", "a", "gnd", 1e-6)
        b.add_inductor("L", "a", "m", 1e-3)
        b.add_resistor("R2", "m", "gnd", 10.0)
        return b
    off = p.simulate(build(), t_end=1e-3, dt=1e-6, engine="pwl", start_from_dc_op=True)
    on = p.simulate(build(), t_end=1e-3, dt=1e-6, engine="pwl", start_from_dc_op=True, **ON)
    assert np.array_equal(np.asarray(off.states), np.asarray(on.states))
    assert on.n_restart_steps == 0


def test_smooth_segments_stay_trapezoidal():
    """An ideal LC over 1000 periods, energy measured from E[0] (the
    initial condition, not a sample the engine computed). The only
    restarts are the start window (i0 with no capacitor current history
    is an inconsistent start); past it the trapezoid conserves energy
    exactly as before."""
    w = 1 / math.sqrt(1e-3 * 1e-6)
    Tp = 2 * math.pi / w
    total, after = {}, {}
    for kw in ({}, ON):
        b = p.CircuitBuilder()
        b.add_inductor("L", "a", "gnd", 1e-3, i0=1.0)
        b.add_capacitor("C", "a", "gnd", 1e-6)
        r = p.simulate(b, t_end=1000 * Tp, dt=Tp / 50, engine="pwl", **kw)
        i, v = np.asarray(r.i("L")), np.asarray(r.v("a"))
        E = 0.5 * 1e-3 * i ** 2 + 0.5 * 1e-6 * v ** 2
        total[bool(kw)] = abs(E[-1] / E[0] - 1)
        after[bool(kw)] = abs(E[-1] / E[3] - 1)
        assert r.n_restart_steps == (3 if kw else 0)
    assert after[True] < 1e-10 and after[False] < 1e-10      # 2.7e-11, 2.8e-11
    assert total[True] < 1e-5, total                          # 5.49e-06
    assert total[False] == pytest.approx(3.932e-3, rel=1e-3)  # the half-step start


# --- resonant modes through many restarts (why the BE pair was replaced) -------
LT, CT = 100e-6, 100e-9
F0 = 1 / (2 * math.pi * math.sqrt(LT * CT))
T_RES = 1 / F0


def _tank_peak(N, **kw):
    """Series tank (Q = 50) driven at resonance by a +-50 V PWM source:
    two restarts per period, steady-state peak over periods 380-400."""
    b = p.CircuitBuilder()
    b.add_pwm_voltage_source("V", "a", "gnd", 50.0, -50.0, F0, 0.5)
    b.add_resistor("R", "a", "m", math.sqrt(LT / CT) / 50)
    b.add_inductor("L", "m", "n", LT)
    b.add_capacitor("C", "n", "gnd", CT)
    r = p.simulate(b, t_end=400 * T_RES, dt=T_RES / N, engine="pwl", **kw)
    t, i = np.asarray(r.times), np.asarray(r.i("L"))
    return float(np.abs(i[t > 380 * T_RES]).max()), r


def test_a_resonant_tank_keeps_its_peak_through_the_restarts():
    """Gate: flag-on peak error within 1.5x the flag-off error at T/50
    and T/100 (the BE pair: -11.9 % / -3.1 %, 14x and 56x)."""
    ref = _tank_peak(1600)[0]                       # 100.658 A
    for N, err_off in ((50, -0.855e-2), (100, -0.055e-2)):
        pk_off = _tank_peak(N)[0]
        pk_on, r = _tank_peak(N, **ON)
        assert pk_off / ref - 1 == pytest.approx(err_off, rel=0.02)
        assert abs(pk_on / ref - 1) <= 1.5 * abs(pk_off / ref - 1), (N, pk_on, pk_off, ref)
        assert r.n_restart_steps == 2400            # 3 x 2 edges x 400 periods
    # engine: -0.730 % at T/50, -0.040 % at T/100


def _isolated_lc(**kw):
    w = 1 / math.sqrt(1e-3 * 1e-6)
    Tp = 2 * math.pi / w
    b = p.CircuitBuilder()
    b.add_inductor("L", "a", "gnd", 1e-3, i0=1.0)
    b.add_capacitor("C", "a", "gnd", 1e-6)
    # An electrically isolated PWM source: its jumps restart the WHOLE system.
    b.add_pwm_voltage_source("Vp", "x", "gnd", 10.0, 0.0, 2.5 / Tp, 0.5)
    b.add_resistor("Rx", "x", "gnd", 10.0)
    r = p.simulate(b, t_end=100 * Tp, dt=Tp / 50, engine="pwl", **kw)
    i, v = np.asarray(r.i("L")), np.asarray(r.v("a"))
    E = 0.5e-3 * i ** 2 + 0.5e-6 * v ** 2
    return float(E[-1] / E[0]), r


def test_an_isolated_pwm_source_does_not_drain_an_lc():
    """Gate: the LC's energy after 100 periods within 0.5 % of flag-off
    (the BE pair kept 0.019 of it)."""
    e_off, _ = _isolated_lc()
    e_on, r = _isolated_lc(**ON)
    assert e_off == pytest.approx(0.9961, abs=2e-4)
    assert abs(e_on - e_off) <= 5e-3, (e_on, e_off)   # engine: 0.9973
    assert r.n_restart_steps == 1500


def test_a_resumed_run_is_exact_and_does_not_restart():
    def build():
        b = p.CircuitBuilder()
        b.add_voltage_source("Vin", "vin", "gnd", VIN)
        b.add_switch("S", "vin", "sw", GON, GOFF)
        b.add_diode("D1", "gnd", "sw", GON, GOFF)
        b.add_inductor("L", "sw", "vout", LB)
        b.add_capacitor("C", "vout", "gnd", CB)
        b.add_resistor("RL", "vout", "gnd", 50.0)
        return b
    h = TB / 100

    def fn_of(b):
        return _mask_fn(b, "S", lambda t: int(round(t / h)) % 100 < 30)
    b = build()
    full = p.simulate(b, t_end=20 * TB, dt=h, engine="pwl", switch_fn=fn_of(b), **ON)
    b = build()
    first = p.simulate(b, t_end=7.55 * TB, dt=h, engine="pwl", switch_fn=fn_of(b), **ON)
    b = build()
    second = p.simulate(b, t_end=20 * TB, dt=h, engine="pwl", switch_fn=fn_of(b),
                        resume_from=first.final_snapshot, **ON)
    tf = np.asarray(full.times)
    xf = np.asarray(full.states)[tf >= second.times[0] - 0.5 * h]
    xs = np.asarray(second.states)
    n = min(len(xf), len(xs))
    assert np.abs(xf[:n] - xs[:n]).max() <= 1e-12 * np.abs(xf[:n]).max()
    assert first.final_snapshot.restart_pending is False
    assert first.final_snapshot.restart_steps_left == 0


def test_a_resume_inside_a_restart_window_is_exact():
    """The window is a COUNT carried by the snapshot (and its pickle):
    stopping one step after an edge leaves steps owed, and the resumed
    run takes them exactly as the uninterrupted one."""
    import pickle

    def build():
        b = p.CircuitBuilder()
        b.add_voltage_source("Vin", "vin", "gnd", VIN)
        b.add_switch("S", "vin", "sw", GON, GOFF)
        b.add_diode("D1", "gnd", "sw", GON, GOFF)
        b.add_inductor("L", "sw", "vout", LB)
        b.add_capacitor("C", "vout", "gnd", CB)
        b.add_resistor("RL", "vout", "gnd", 1.0)
        return b
    h = TB / 100

    def fn_of(b):
        return _mask_fn(b, "S", lambda t: int(round(t / h)) % 100 < 50)
    b = build()
    full = p.simulate(b, t_end=10 * TB, dt=h, engine="pwl", switch_fn=fn_of(b), **ON)
    b = build()
    first = p.simulate(b, t_end=7.51 * TB, dt=h, engine="pwl", switch_fn=fn_of(b), **ON)
    snap = pickle.loads(pickle.dumps(first.final_snapshot))
    # the edge step took one of three; two owed
    assert snap.restart_steps_left == first.final_snapshot.restart_steps_left == 2
    b = build()
    second = p.simulate(b, t_end=10 * TB, dt=h, engine="pwl", switch_fn=fn_of(b),
                        resume_from=snap, **ON)
    tf = np.asarray(full.times)
    xf = np.asarray(full.states)[tf >= second.times[0] - 0.5 * h]
    xs = np.asarray(second.states)
    n = min(len(xf), len(xs))
    assert n > 200
    assert np.abs(xf[:n] - xs[:n]).max() <= 1e-12 * np.abs(xf[:n]).max()


def test_a_late_diode_flip_re_arms_the_window():
    """A series L-C-diode half-sine of 1.5 dt after a gate edge: the
    diode turns off inside the SECOND restart step, which re-arms the
    window for the next two. Start window 3 + edge window 4 = 7 (6
    without the re-arm)."""
    dt = 1e-6
    te = 10 * dt
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S", "in", "a", GON, GOFF)
    b.add_inductor("L", "a", "m", 1e-6)
    b.add_diode("D", "m", "k", GON, GOFF)
    b.add_capacitor("C", "k", "gnd", (1.5 * dt / math.pi) ** 2 / 1e-6)
    r = p.simulate(b, t_end=te + 10 * dt + 1e-12, dt=dt, engine="pwl",
                   switch_fn=_mask_fn(b, "S", lambda t: t >= te - 1e-15), **ON)
    assert r.n_restart_steps == 7
    assert np.abs(np.asarray(r.i("L"))[13:]).max() < 1e-6   # the diode holds the charge


def test_one_factor_per_mask_and_no_refactor():
    """Both SDIRK stages run on the trapezoidal factor of 2*gamma*dt
    (C/(gamma*dt)), kept by a second cache at that fixed step: one
    factorisation per mask visited by a restart, none in the event LRU
    (which refactored on every restart step of a VSI: 3172 builds in
    3157 restart steps; 81 factorisations now)."""
    caches = []
    runs = [_buck(1.0, 0.5, 100, nper=1000, on_cache=caches.append, **kw)
            for kw in ({}, ON)]
    assert runs[1].n_restart_steps >= 6000        # 3 steps x two edges per period
    assert runs[1].n_restart_factorizations == 4
    assert runs[0].n_restart_factorizations == 0
    off, on = caches
    m = on.metrics()
    assert m.event_builds == m.event_refactors == on.num_event_entries() == 0
    # The transitional masks of the event iteration are now solved on
    # the restart factor only: nominal segments 2 (4 without the rule),
    # i.e. 6 factorisations in all against 4.
    assert (on.num_built_segments(), off.num_built_segments()) == (2, 4)


def test_restart_and_substep_correction_are_refused_together():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_resistor("R", "a", "c", 1.0)
    b.add_capacitor("C", "c", "gnd", 1e-6)
    with pytest.raises(ValueError, match="discontinuity_restart.*substep|substep.*discontinuity_restart"):
        p.simulate(b, t_end=1e-5, dt=1e-6, engine="pwl",
                   enable_substep_state_correction=True, **ON)


# --- the step observer and the recorded mask --------------------------------------
def test_the_observer_keeps_its_label_under_the_rule():
    """Design D4: the gate of step (t_{k-1}, t_k] moves to switch_fn(t_{k-1}),
    but the observer keeps today's call (t_k, x_{k-1}) after the t_start
    prime; moving its label belongs to fix-control-tick-semantics."""
    def run(**kw):
        calls = []
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "s", "gnd", 1.0)
        b.add_resistor("R", "s", "a", 1.0)
        b.add_capacitor("C", "a", "gnd", 1e-6)
        res = p.simulate(b, t_end=5e-6, dt=1e-6, engine="pwl",
                         step_observer=lambda t, x: calls.append((t, float(x[1]))), **kw)
        return calls, np.asarray(res.v("a"))
    calls, v = run(**ON)
    off_calls, _ = run()
    assert [c[0] for c in calls] == [c[0] for c in off_calls]
    assert [c[0] for c in calls] == pytest.approx([0, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6], abs=1e-15)
    assert [c[1] for c in calls] == pytest.approx([v[0]] + list(v[:-1]), abs=1e-15)


# --- stateful devices ------------------------------------------------------------
def test_coss_charge_is_exact_from_the_first_step():
    """Q(v(t)) = I·t at every sample from the first: without the rule the
    trapezoidal start is short half a step of charge forever
    (test_nonlinear_capacitor.py pins that I·(t − h/2))."""
    C0, V0, m, i_src, dt = 2000e-12, 25.0, 0.5, 6.0, 2e-11
    b = p.CircuitBuilder()
    b.add_current_source("Isrc", "n", "gnd", i_src)
    b.add_nonlinear_capacitor("Coss", "n", "gnd", C0, V0, m)
    b.add_resistor("Rleak", "n", "gnd", 1e12)
    res = p.simulate(b, t_end=80e-9, dt=dt, engine="pwl", **ON)
    t, v = np.asarray(res.times), np.asarray(res.v("n"))
    q = C0 * V0 / (1 - m) * ((1 + v / V0) ** (1 - m) - 1)
    rel = np.abs(q[1:] - i_src * t[1:]) / (i_src * t[1:])
    assert rel.max() < 1e-6, rel.max()


L0S, ISAT, LRES = 1e-3, 2.0, 5e-5


def _sat_edge(dt, **kw):
    te = 1e-4
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S", "in", "a", GON, GOFF)
    b.add_resistor("R", "a", "m", 1.0)
    b.add_saturable_inductor("Ls", "m", "gnd", L0S, ISAT, LRES)
    res = p.simulate(b, t_end=te + 0.5e-3 + 1e-12, dt=dt, engine="pwl",
                     switch_fn=_mask_fn(b, "S", lambda t: t >= te - 1e-15), **kw)
    t, i = np.asarray(res.times), np.asarray(res.i("Ls"))
    j = int(np.argmin(abs(t - (te + 0.5e-3))))
    Lf = lambda x: LRES + (L0S - LRES) / (1 + (x / ISAT) ** 2)  # noqa: E731 — the engine's atan law
    ref = solve_ivp(lambda _t, y: [(10.0 - (1.0 + 1 / GON) * y[0]) / Lf(y[0])],
                    (0, 0.5e-3), [0.0], rtol=1e-12, atol=1e-14, method="DOP853").y[0, -1]
    return float(i[j] - ref)


def test_saturable_inductor_across_an_edge_is_second_order():
    on = [_sat_edge(dt, **ON) for dt in (4e-6, 2e-6, 1e-6, 5e-7)]
    assert all(f >= 3.7 for f in _factors(on)), on     # engine: 4.94e-5 ... 7.82e-7
    off = _sat_edge(2e-6)
    assert abs(off) >= 20 * abs(on[1]), (off, on[1])    # 1.07e-2 vs 1.24e-5


# --- the other stateful devices and coupled windings through a restart ---------
# Each stage-2 solve starts from the pseudo previous state x0* = x0 + c(x1 - x0);
# every device history extrapolates its own charge/flux the same way. A device
# left out would restart from a stale charge: first order (ratio 2), not 4.
def _richardson(fn, dts, **kw):
    y = [fn(dt, **kw)[0] for dt in dts]
    d = [a - b for a, b in zip(y, y[1:])]
    return [a / b for a, b in zip(d, d[1:])], fn(dts[0], **kw)[1]


def _at(res, sig, tq):
    t = np.asarray(res.times)
    return float(np.asarray(sig)[int(np.argmin(abs(t - tq)))])


def _lauritzen(dt, **kw):
    """+5 V -> -5 V source jump on R + Lauritzen diode: reverse recovery."""
    te = 1e-6
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "in", "gnd", 5.0, -5.0, te, 1.0, 0.0, 0.0, 0.0)
    b.add_resistor("R", "in", "a", 10.0)
    b.add_lauritzen_diode("D", "a", "gnd")
    r = p.simulate(b, t_end=te + 0.2e-6 + 1e-15, dt=dt, engine="pwl", **kw)
    return _at(r, r.i("R"), te + 0.1e-6), r.n_restart_steps


def _igbt_tail(dt, **kw):
    """Hard gate turn-off (a source jump) of an IGBT with a tail, resistive load."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc", "gnd", 600.0)
    b.add_pulse_voltage_source("Vg", "g", "gnd", 0.0, 15.0, 0.0, 2e-6, 0.0, 0.0, 0.0)
    b.add_resistor("Rload", "dc", "c", 6.0)
    b.add_igbt_level1("Q1", "c", "e", "g", 1.5, 0.05, 5.0, tau_tail=1e-6, k_tail=0.30)
    b.add_resistor("Rsense", "e", "gnd", 1e-4)
    n = b.graph.num_switches
    r = p.simulate(b, t_end=4e-6, dt=dt, engine="pwl",
                   switch_fn=lambda _t: p.SwitchStateMask(n), **kw)
    return _at(r, r.i("Rsense"), 3e-6), r.n_restart_steps


def _pmsm(dt, **kw):
    """A 6 V step on phase a of a salient MNA PMSM."""
    te = 1e-3
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("Vs_a", "ua", "gnd", 0.0, 6.0, te, 1.0, 0.0, 0.0, 0.0)
    b.add_voltage_source("Vs_b", "ub", "gnd", 0.0)
    b.add_voltage_source("Vs_c", "uc", "gnd", 0.0)
    b.add_pmsm_mna("M1", "ua", "ub", "uc", "nn", "w", "th", R_s=0.5, L_d=1e-3,
                   L_q=3e-3, psi_pm=0.05, pole_pairs=4, J=1e-4, B=0.0)
    r = p.simulate(b, t_end=te + 1e-3 + 1e-12, dt=dt, engine="pwl", **kw)
    return _at(r, r.i("M1_a"), te + 0.5e-3), r.n_restart_steps


def _transformer(dt, **kw):
    te = 1e-4
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S", "in", "a", GON, GOFF)
    b.add_resistor("Rp", "a", "p", 1.0)
    b.add_transformer("T", "p", "gnd", "s", "gnd", 1e-3, 4e-3, 0.95)
    b.add_resistor("Rs", "s", "gnd", 20.0)
    r = p.simulate(b, t_end=te + 0.2e-3 + 1e-12, dt=dt, engine="pwl",
                   switch_fn=_mask_fn(b, "S", lambda t: t >= te - 1e-15), **kw)
    return _at(r, r.i("Rs"), te + 0.1e-3), r.n_restart_steps


@pytest.mark.parametrize("fn, dts, off_ratio", [
    (_lauritzen, (4e-9, 2e-9, 1e-9, 5e-10), 2.1),      # on 4.47 3.78, off 2.10 2.16
    (_igbt_tail, (8e-9, 4e-9, 2e-9, 1e-9), 2.0),       # on 3.96 3.98, off 2.00 2.00
    (_transformer, (4e-6, 2e-6, 1e-6, 5e-7), 1.97),    # on 3.84 3.92, off 1.96 1.98
], ids=["lauritzen", "igbt_tail", "transformer"])
def test_stateful_devices_are_second_order_through_a_restart(fn, dts, off_ratio):
    on, n = _richardson(fn, dts, **ON)
    assert all(3.5 <= r <= 4.6 for r in on), on
    assert n == 6                                   # start window + the edge window
    off, _ = _richardson(fn, dts)
    assert all(abs(r - off_ratio) <= 0.2 for r in off), off


@pytest.mark.filterwarnings("ignore::UserWarning")     # its dt/2 retries, by design
def test_mna_pmsm_is_second_order_through_a_restart():
    """Every fixed step of an MNA PMSM goes through the dt/2 retry (its
    linear segment has no stamp for the phase currents), flag on or off;
    the retry's sub-steps are SDIRK steps inside the window. Flag off the
    step-halving differences do not even shrink (1.6e-3, 2.3e-3, -1.2e-3)."""
    on, n = _richardson(_pmsm, (4e-6, 2e-6, 1e-6, 5e-7), **ON)
    assert all(3.8 <= r <= 4.3 for r in on), on     # 4.05 4.03
    assert n == 6


# --- user b_extra_fn jumps --------------------------------------------------------
def _bx_rc(declare, **kw):
    te, calls = 2e-6, []
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 0.0)
    b.add_resistor("R", "in", "a", 1e3)
    b.add_capacitor("C", "a", "gnd", 1e-9)

    def bx(t):
        calls.append(t)
        v = np.zeros(3)
        v[2] = 10.0 if t >= te else 0.0     # the source's KVL row: a 10 V step
        return v
    if declare:
        bx.next_jump_after = lambda t: te if t < te else math.inf
    r = p.simulate(b, t_end=te + 4e-6 + 1e-12, dt=1e-7, engine="pwl", b_extra_fn=bx, **kw)
    t, v = np.asarray(r.times), np.asarray(r.v("a"))
    ex = np.where(t > te, np.sign(v[-1]) * 10 * (1 - np.exp(-(t - te) / 1e-6)), 0.0)
    return r, float(np.abs(v - ex).max()), np.asarray(calls)


def test_a_declared_b_extra_jump_restarts():
    """`fn.next_jump_after(t)` declares the callback's jumps: they restart
    and read the left limit like a built-in source. Undeclared jumps do
    not (documented): same error as without the rule."""
    r, err, _ = _bx_rc(True, **ON)
    assert r.n_restart_steps == 6
    assert err < 5e-3, err                       # engine: 2.63e-03 (the RC pulse's)
    r, err_u, _ = _bx_rc(False, **ON)
    assert r.n_restart_steps == 3                # the start window only
    assert err_u == pytest.approx(0.476, rel=0.01)


def test_b_extra_fn_is_called_once_per_stage_time():
    """One call per nominal step and one per SDIRK stage (two per restart
    step, the first OFF the grid at t_prev + gamma*dt); none discarded."""
    r, _, calls = _bx_rc(True, **ON)
    n_steps = len(r.times) - 1
    assert len(calls) == n_steps + r.n_restart_steps == 66
    assert np.all(np.diff(calls) > 0)
    gamma = 1 - 1 / math.sqrt(2)
    assert calls[0] == pytest.approx(gamma * 1e-7, rel=1e-12)


# --- the static path (no L, no C) follows the same sampling rule --------------------
def test_the_static_path_samples_the_gate_at_the_step_start():
    def run(**kw):
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "in", "gnd", 10.0)
        b.add_switch("S", "in", "a", GON, GOFF)
        b.add_resistor("R", "a", "gnd", 10.0)
        b.add_pwm_voltage_source("Vp", "x", "gnd", 5.0, 0.0, 1e5, 0.5)
        b.add_resistor("Rx", "x", "gnd", 1.0)
        cache = p.PwlStateSpaceCache(b.graph, b.pool)
        cache.build()                             # dt = 0: the static path
        opts = p.SimulationOptions(t_start=0.0, t_end=1e-5 + 1e-12, dt=1e-6)
        for k, v in kw.items():
            setattr(opts, k, v)
        res = p.run_transient(cache, b.graph, b.pool, opts,
                              switch_fn=_mask_fn(b, "S", lambda t: t >= 3e-6 - 1e-15))
        res._builder = b
        return np.asarray(res.v("a")) > 5.0, np.asarray(res.v("x"))
    closed_off, vx_off = run()
    closed_on, vx_on = run(**ON)
    assert int(np.argmax(closed_off)) == 3
    assert int(np.argmax(closed_on)) == 4        # the step (3, 4] us reads switch_fn(3 us)
    # the PWM falls at 5 us and rises at 10 us: the sample ON a jump reads the left limit
    assert list(vx_off[5:11]) == [0.0, 0.0, 0.0, 0.0, 0.0, 5.0]
    assert list(vx_on[5:11]) == [5.0, 0.0, 0.0, 0.0, 0.0, 0.0]
