"""The fixed engine across discontinuities: the restart rule.

`engine='pwl'` is a trapezoidal companion. After a mask change, a
source jump or an inconsistent start its derivative slots (a
capacitor's i_prev, an inductor's v_prev) describe a circuit that no
longer exists, and the trapezoid averages that stale derivative with
the new one: every edge lands half a step off (FIRST order) and every
algebraic quantity alternates at the Nyquist frequency, undamped.

`simulate(discontinuity_restart=True)` (add-discontinuity-restart-rule;
Marti & Lin's critical damping adjustment) takes the step after each
discontinuity as two backward-Euler half-steps on the nominal factor,
samples the gate mask at the step START and reads a source's left
limit on a jump at the step end. Each test below pins BOTH sides: the
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
    assert abs(on[1]) <= 1e-5, on                 # engine: -5.050e-06
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
    on = [_buck_err(1.0, 0.5, N, **ON) for N in (50, 100, 200, 400, 800)]
    assert abs(on[0]) <= 5e-5, on                 # engine: 2.20e-05
    assert all(f >= 3.8 for f in _factors(on)), on
    off = [_buck_err(1.0, 0.5, N) for N in (50, 100)]
    assert off[0] == pytest.approx(1.18e-2, rel=0.02)   # first order
    assert 1.9 < off[0] / off[1] < 2.1


def test_freewheel_diode_picks_up_the_current_in_the_first_half_step():
    """The pair is not atomic for the diode event iteration: decided
    only after the second half-step, the turn-off kick is damped away in
    the first and the buck's inductor current is chopped every period
    (prototype: v_out 0.29 V)."""
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
    assert abs(errs[0]) < 1.5e-3, errs                # engine: -9.56e-04
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
    """One pair leaves 4·C·tau·dV/h^2 of the unresolved 1 ps mode
    ringing (design D8): 0.4 A predicted at 1 ns, 0.18 A measured."""
    exact = 100.0 / (10.0 + 1 / GON)
    i_on, _ = _caploop(1e-9, **ON)
    assert np.abs(i_on - exact).max() <= 0.5, (i_on.min(), i_on.max())
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
    assert on[0] <= 0.03, on                          # engine: 2.19e-02
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
    assert res.n_restart_steps == 20          # 2 edges x 10 periods (t = 0 among them)


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
    """An ideal LC conserves energy over 1000 periods exactly as before:
    the only restart is the first step (i0 with no capacitor current
    history is an inconsistent start)."""
    w = 1 / math.sqrt(1e-3 * 1e-6)
    Tp = 2 * math.pi / w
    drift = {}
    for kw in ({}, ON):
        b = p.CircuitBuilder()
        b.add_inductor("L", "a", "gnd", 1e-3, i0=1.0)
        b.add_capacitor("C", "a", "gnd", 1e-6)
        r = p.simulate(b, t_end=1000 * Tp, dt=Tp / 50, engine="pwl", **kw)
        i, v = np.asarray(r.i("L")), np.asarray(r.v("a"))
        E = 0.5 * 1e-3 * i ** 2 + 0.5 * 1e-6 * v ** 2
        drift[bool(kw)] = abs(E[-1] / E[1] - 1)
        assert r.n_restart_steps == (1 if kw else 0)
    assert drift[True] < 1e-10 and drift[False] < 1e-10


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


def test_the_nominal_factor_is_reused():
    """The half-steps run on the factor of the nominal dt (C/h_e =
    2C/dt): no (mask, dt) entry, no event build, no extra segment."""
    caches = []
    runs = [_buck(1.0, 0.5, 100, nper=1000, on_cache=caches.append, **kw)
            for kw in ({}, ON)]
    assert runs[1].n_restart_steps >= 2000        # two edges per period
    off, on = caches
    m = on.metrics()
    assert m.event_builds == 0 and m.event_refactors == 0
    assert on.num_event_entries() == 0
    assert on.num_built_segments() == off.num_built_segments()   # one per visited mask


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
