"""The fixed engine with events inside a step: the event split.

`discontinuity_restart=True` makes a mask change at a grid point second
order, but an event BETWEEN grid points is still taken at the next one
(a timing error of up to dt: first order). `enable_substep_state_
correction=True` (add-fixed-step-event-splitting; requires the restart
rule) splits such a step at the event: a trapezoidal sub-step up to it
under the pre-event mask, then an SDIRK(2,2) restart step to the grid
point (and the two grid steps after it). Events: gate edges a schedule
names (`next_edge_after`), built-in source jumps, and diode crossings
located from the step's first solve.

Each test pins both sides: the restart rule alone (the measured
"before") and the split (the target, against an exact reference).
Removing the split fails every `split` assertion.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.optimize import brentq

import pulsim as p

RULE = dict(discontinuity_restart=True)
SPLIT = dict(discontinuity_restart=True, enable_substep_state_correction=True)
GON, GOFF = 1e3, 1e-9


def _factors(errs):
    return [abs(a / b) for a, b in zip(errs, errs[1:])]


def _closing_at(b, name, te, width=1.0):
    """A schedule (edge oracle) that closes `name` at te and keeps it closed."""
    return p.make_pwm_switch_fn(1 / (2 * width), 0.5, b.switch_index_of(name),
                                b.graph.num_switches, te)


# --- RL behind a switch that closes between grid points ---------------------
V, R, L = 10.0, 1.0, 1e-3
TAU = L / (R + 1 / GON)


def _rl(dt, frac, **kw):
    te = (round(1e-3 / dt) + frac) * dt
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", V)
    b.add_switch("S", "in", "a", GON, GOFF)
    b.add_resistor("R", "a", "m", R)
    b.add_inductor("L", "m", "gnd", L)
    # closed from te for 3 ms: one period of a 250 Hz, 75 % schedule
    fn = p.make_pwm_switch_fn(250.0, 0.75, b.switch_index_of("S"),
                              b.graph.num_switches, te)
    res = p.simulate(b, t_end=te + 0.6e-3 + 1e-12, dt=dt, engine="pwl",
                     switch_fn=fn, **kw)
    t, i = np.asarray(res.times), np.asarray(res.i("L"))
    j = int(np.argmin(abs(t - (te + 0.5e-3))))
    return res, i[j] - V / (R + 1 / GON) * (1 - math.exp(-(t[j] - te) / TAU))


@pytest.mark.parametrize("frac, at_2us, rule_2us", [
    (0.3, +1.012e-6, -8.497e-3),
    (0.7, +1.014e-6, -3.635e-3),
])
def test_gate_edge_off_the_grid_is_second_order(frac, at_2us, rule_2us):
    errs = [_rl(dt, frac, **SPLIT)[1] for dt in (8e-6, 4e-6, 2e-6, 1e-6)]
    assert abs(errs[2]) <= 1e-5, errs
    # The edge on the grid with the rule alone reads +1.01e-6 too.
    assert errs[2] == pytest.approx(at_2us, rel=2e-2)
    assert all(f >= 3.7 for f in _factors(errs)), errs
    # The rule alone takes the edge at the next grid point: first order.
    rule = [_rl(dt, frac, **RULE)[1] for dt in (4e-6, 2e-6)]
    assert rule[1] == pytest.approx(rule_2us, rel=1e-3)
    assert _factors(rule)[0] == pytest.approx(2.0, abs=0.02)


def test_output_grid_is_unchanged_and_splits_are_counted():
    res, _ = _rl(2e-6, 0.3, **SPLIT)
    t = np.asarray(res.times)
    assert np.array_equal(t, np.arange(len(t)) * 2e-6)   # times[k] = k·dt, exactly
    # The 75 % schedule's previous period ends 0.3 dt after t = 0 (the
    # inductor has no current then), then the edge at te: two splits.
    assert res.n_split_steps == 2
    assert _rl(2e-6, 0.3, **RULE)[0].n_split_steps == 0


def test_an_opaque_switch_fn_is_not_split():
    """No `next_edge_after`: the edge stays at the next grid point (design
    open question 4, the conservative option)."""
    te = 500.3 * 2e-6
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", V)
    b.add_switch("S", "in", "a", GON, GOFF)
    b.add_resistor("R", "a", "m", R)
    b.add_inductor("L", "m", "gnd", L)
    n, k = b.graph.num_switches, b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, t >= te)
        return m
    res = p.simulate(b, t_end=te + 0.6e-3, dt=2e-6, engine="pwl", switch_fn=fn, **SPLIT)
    assert res.n_split_steps == 0


# --- buck CCM with a duty that is not a whole number of steps ----------------
VIN, LB, CB, FSW = 24.0, 100e-6, 47e-6, 100e3
TB, T0 = 1 / FSW, 2e-6


def _prop(A, B, x, dt):
    M = np.zeros((3, 3))
    M[:2, :2] = A
    M[:2, 2] = B
    E = expm(M * dt)
    return E[:2, :2] @ x + E[:2, 2]


def _buck_exact(Rl, D, nper):
    """Exact piecewise-LTI solution with the diode events located."""
    def AB(gs, gd):
        k = 1 / (gs + gd)
        return (np.array([[-k / LB, -1 / LB], [1 / CB, -1 / (Rl * CB)]]),
                np.array([gs * VIN * k / LB, 0.0]))

    def seg(x, rem, gs):
        d_on = -((gs * VIN - x[0]) / (gs + GOFF)) >= 0
        gd = GON if d_on else GOFF
        A, B = AB(gs, gd)
        f = lambda h: -((gs * VIN - _prop(A, B, x, h)[0]) / (gs + gd))  # noqa: E731
        if f(1e-15) * f(rem) < 0:
            h = brentq(f, 1e-15, rem, xtol=1e-18, rtol=1e-15)
            A2, B2 = AB(gs, GOFF if d_on else GON)
            return _prop(A2, B2, _prop(A, B, x, h), rem - h)
        return _prop(A, B, x, rem)
    x = np.zeros(2)
    for _ in range(nper):
        x = seg(x, D * TB, GON)
        x = seg(x, (1 - D) * TB, GOFF)
    return x


def _buck_builder(Rl):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", VIN)
    b.add_switch("S", "vin", "sw", GON, GOFF)
    b.add_diode("D1", "gnd", "sw", GON, GOFF)
    b.add_inductor("L", "sw", "vout", LB)
    b.add_capacitor("C", "vout", "gnd", CB)
    b.add_resistor("RL", "vout", "gnd", Rl)
    return b


def _buck(Rl, N, nper, fn=None, D=0.4567, phase=T0, **kw):
    b = _buck_builder(Rl)
    if fn is None:
        fn = p.make_pwm_switch_fn(FSW, D, b.switch_index_of("S"),
                                  b.graph.num_switches, phase)
    else:
        fn = fn(b)
    return p.simulate(b, t_end=phase + nper * TB + 1e-12, dt=TB / N,
                      engine="pwl", switch_fn=fn, **kw)


def _buck_err(N, **kw):
    res = _buck(1.0, N, 40, **kw)
    t = np.asarray(res.times)
    j = int(np.argmin(abs(t - (T0 + 40 * TB))))
    return float(np.asarray(res.v("vout"))[j] - _buck_exact(1.0, 0.4567, 40)[1])


def test_duty_off_the_grid_is_second_order():
    errs = [_buck_err(N, **SPLIT) for N in (50, 100, 200, 400)]
    assert abs(errs[0]) <= 2e-6, errs          # engine: +1.07e-6 (prototype 7.3e-6)
    assert all(f >= 3.5 for f in _factors(errs)), errs   # 4.2 4.2 4.3
    # The rule alone snaps the duty edge to the grid: 8.08e-2 V at T/50,
    # and it does not converge until the snapped duty changes.
    assert _buck_err(50, **RULE) == pytest.approx(8.08e-2, rel=1e-2)
    assert _buck_err(100, **RULE) == pytest.approx(8.08e-2, rel=1e-2)


# --- DCM buck: the diode turn-off is located -----------------------------------
def _grid_fn(N):
    def make(b):
        n, k, Don = b.graph.num_switches, b.switch_index_of("S"), round(0.3 * N)
        dt = TB / N

        def fn(t):
            m = p.SwitchStateMask(n)
            m.set(k, bool(int(round(t / dt)) % N < Don))
            return m
        return fn
    return make


def _off_events(res, N, after):
    """(t_est/dt, zero crossing − t_est in dt) for each OFF event; the
    crossing extrapolates the falling ramp of the two samples before it."""
    dt = TB / N
    t, iL = np.asarray(res.times), np.asarray(res.i("L"))
    out = []
    for e in res.commutation_events:
        if e.new_state or e.t_estimated < after:
            continue
        j = int(np.floor(e.t_estimated / dt + 1e-9))
        i1, i2 = iL[j - 1], iL[j]
        tz = t[j] + i2 / ((i1 - i2) / dt)
        out.append((e.t_estimated / dt, (tz - e.t_estimated) / dt))
    return out


def test_diode_turn_off_is_located_within_005_dt():
    # p16: 1000 periods (steady state), the last five turn-offs.
    kw = dict(fn=_grid_fn(100), phase=0.0)

    def offs(**k2):
        return _off_events(_buck(50.0, 100, 1000, **kw, **k2), 100, 995 * TB)
    off = offs(**SPLIT)
    assert len(off) == 5
    assert all(abs(d) <= 0.05 for _, d in off), off          # engine: |d| < 1e-4
    assert all(abs(k - round(k)) > 0.01 for k, _ in off), off
    # The rule alone: every estimate is a grid index, 0.98 dt early.
    off = offs(**RULE)
    assert all(abs(k - round(k)) < 1e-3 for k, _ in off), off
    assert all(0.95 <= d <= 1.0 for _, d in off), off
    # Default engine: a grid index too, 0.32-0.33 dt early.
    off = offs()
    assert all(abs(k - round(k)) < 1e-3 for k, _ in off), off
    assert all(0.31 <= d <= 0.34 for _, d in off), off


def test_dcm_result_does_not_depend_on_where_the_grid_falls():
    # p15/p12: the default engine is 0.50 % off at T/101 (0.023 % at T/100).
    # Relative error against the exact steady state of each run's own
    # duty (round(0.3 N)/N), 2000 periods (8.5 RC).
    def rel(N, **kw):
        res = _buck(50.0, N, 2000, fn=_grid_fn(N), phase=0.0, store_every=N, **kw)
        v = float(np.asarray(res.v("vout"))[-1])
        return v / _buck_exact(50.0, round(0.3 * N) / N, 2000)[1] - 1
    split = {N: rel(N, **SPLIT) for N in (100, 101, 400)}
    assert all(abs(r) < 1e-6 for r in split.values()), split      # engine: < 8e-8
    # Task 1.3: the restart rule ALONE already removes the outlier
    # (T/101 +6.9e-6 relative); the split takes it to 7e-8.
    assert abs(rel(101, **RULE)) < 2e-5
    assert rel(101) == pytest.approx(-4.97e-3, rel=0.05)


def test_split_off_is_the_restart_rule_bit_for_bit():
    """With every edge on the grid (D = 0.5 at T/100) the split finds no
    event: split ON is the restart rule alone, bit for bit; and an
    explicit False is the rule too."""
    on = _buck(1.0, 100, 10, D=0.5, **SPLIT)
    rule = _buck(1.0, 100, 10, D=0.5, **RULE)
    off = _buck(1.0, 100, 10, D=0.5, enable_substep_state_correction=False,
                **RULE)
    assert on.n_split_steps == 0
    for r in (on, off):
        assert np.array_equal(np.asarray(r.states), np.asarray(rule.states))
        assert np.array_equal(np.asarray(r.switch_states),
                              np.asarray(rule.switch_states))


def test_split_without_the_restart_rule_is_refused():
    with pytest.raises(p.UnsupportedError, match="discontinuity_restart") as ei:
        _buck(1.0, 100, 2, enable_substep_state_correction=True)
    assert "enable_substep_state_correction" in str(ei.value)
    assert ei.value.feature == "enable_substep_state_correction"


def test_split_steps_agree_with_trbdf2_on_dcm():
    # Time-weighted v_out over the last 20 periods of a DCM buck at T/200.
    def run(**kw):
        b = _buck_builder(50.0)
        fn = p.make_pwm_switch_fn(FSW, 0.3, b.switch_index_of("S"),
                                  b.graph.num_switches, 0.0)
        return p.simulate(b, t_end=60 * TB, switch_fn=fn, **kw)

    def mean(r):
        t, v = np.asarray(r.times), np.asarray(r.v("vout"))
        m = t >= 40 * TB - 1e-12
        return np.trapezoid(v[m], t[m]) / (t[m][-1] - t[m][0])
    ref = mean(run(engine="trbdf2"))
    assert mean(run(dt=TB / 200, engine="pwl", **SPLIT)) == pytest.approx(ref, rel=1e-4)


# --- several diodes in one step ----------------------------------------------
def _bridge(**kw):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "b", v_dc=0.0, v_amplitude=20.0,
                              frequency=50.0, phase=0.0)
    for name, an, ca in (("D1", "a", "p"), ("D2", "b", "p"),
                         ("D3", "gnd", "a"), ("D4", "gnd", "b")):
        b.add_diode(name, an, ca, GON, GOFF)
    b.add_capacitor("C", "p", "gnd", 100e-6)
    b.add_resistor("RL", "p", "gnd", 50.0)
    return p.simulate(b, t_end=0.1, **kw)


def test_bridge_rectifier_at_coarse_dt_has_no_breach():
    res = _bridge(dt=2e-4, engine="pwl", **SPLIT)
    assert len(res.event_iteration_breaches) == 0
    assert res.n_split_steps > 0
    v = np.asarray(res.v("p"))
    ref = _bridge(engine="trbdf2", rtol=1e-8, atol=1e-10)
    tr, vr = np.asarray(ref.times), np.asarray(ref.v("p"))
    # engine: peak 19.99920 V (source 20 V; trbdf2 19.99869); max |err|
    # vs trbdf2 2.45e-3 V (flag off: peak 20.019 V, 7.88e-2 V)
    assert np.all(np.isfinite(v)) and v.max() < 20.0 * (1 + 1e-4)
    assert float(np.abs(v - np.interp(np.asarray(res.times), tr, vr)).max()) < 5e-3


# --- resonant modes through split steps (the restart rule's gates) ----------
LT, CT = 100e-6, 100e-9
F0 = 1 / (2 * math.pi * math.sqrt(LT * CT))


def _tank_peak(N, phase_dt=0.0, **kw):
    """Q = 50 tank at resonance from a +-50 V PWM source, its edges moved
    `phase_dt` steps off the grid; peak current over periods 380-400."""
    T = 1 / F0
    b = p.CircuitBuilder()
    b.add_pwm_voltage_source("V", "a", "gnd", 50.0, -50.0, F0, 0.5, phase_dt * T / N)
    b.add_resistor("R", "a", "m", math.sqrt(LT / CT) / 50)
    b.add_inductor("L", "m", "n", LT)
    b.add_capacitor("C", "n", "gnd", CT)
    r = p.simulate(b, t_end=400 * T, dt=T / N, engine="pwl", **kw)
    t, i = np.asarray(r.times), np.asarray(r.i("L"))
    return float(np.abs(i[t > 380 * T]).max()), r


def test_a_resonant_tank_keeps_its_peak_through_split_steps():
    ref = _tank_peak(1600)[0]
    for N, off_err in ((50, -0.855e-2), (100, -0.055e-2)):
        pk, r = _tank_peak(N, 0.37, **SPLIT)
        assert r.n_split_steps == 800
        err = pk / ref - 1
        # engine: -0.592 % / -0.011 % (rule alone, same edges: -0.663 / -0.024)
        assert abs(err) <= 1.0 * abs(off_err), (N, err)


def test_an_isolated_pwm_source_does_not_drain_an_lc_through_splits():
    Tp = 2 * math.pi * math.sqrt(1e-3 * 1e-6)
    b = p.CircuitBuilder()
    b.add_inductor("L", "a", "gnd", 1e-3, i0=1.0)
    b.add_capacitor("C", "a", "gnd", 1e-6)
    b.add_pwm_voltage_source("Vp", "x", "gnd", 10.0, 0.0, 2.5 / Tp, 0.5, 0.37 * Tp / 50)
    b.add_resistor("Rx", "x", "gnd", 10.0)
    r = p.simulate(b, t_end=100 * Tp, dt=Tp / 50, engine="pwl", **SPLIT)
    i, v = np.asarray(r.i("L")), np.asarray(r.v("a"))
    E = 0.5e-3 * i ** 2 + 0.5e-6 * v ** 2
    assert r.n_split_steps == 500
    assert E[-1] / E[0] == pytest.approx(0.9980, abs=5e-4)   # rule 0.9973, off 0.9961


# --- stateful devices across an off-grid event --------------------------------
def _richardson(fn, dts, **kw):
    y = [fn(dt, **kw)[0] for dt in dts]
    d = [a - b for a, b in zip(y, y[1:])]
    return [a / b for a, b in zip(d, d[1:])], fn(dts[0], **kw)[1]


def _at(res, sig, tq):
    t = np.asarray(res.times)
    return float(np.asarray(sig)[int(np.argmin(abs(t - tq)))])


def _sat_edge(dt, **kw):
    te = 1e-4 + 0.37e-6                      # off every grid used
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S", "in", "a", GON, GOFF)
    b.add_resistor("R", "a", "m", 1.0)
    b.add_saturable_inductor("Ls", "m", "gnd", 1e-3, 2.0, 5e-5)
    r = p.simulate(b, t_end=te + 0.4e-3, dt=dt, engine="pwl",
                   switch_fn=_closing_at(b, "S", te), **kw)
    return _at(r, r.i("Ls"), 4e-4), r.n_split_steps     # a grid point of every dt


def _transformer(dt, **kw):
    te = 1e-4 + 0.37e-6
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S", "in", "a", GON, GOFF)
    b.add_resistor("Rp", "a", "p", 1.0)
    b.add_transformer("T", "p", "gnd", "s", "gnd", 1e-3, 4e-3, 0.95)
    b.add_resistor("Rs", "s", "gnd", 20.0)
    r = p.simulate(b, t_end=te + 0.2e-3, dt=dt, engine="pwl",
                   switch_fn=_closing_at(b, "S", te), **kw)
    return _at(r, r.i("Rs"), 2e-4), r.n_split_steps


def _lauritzen(dt, **kw):
    """+5 V -> -5 V pulse-source jump (off the grid) on R + Lauritzen diode."""
    te = 1e-6 + 1.3e-9
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "in", "gnd", 5.0, -5.0, te, 1.0, 0.0, 0.0, 0.0)
    b.add_resistor("R", "in", "a", 10.0)
    b.add_lauritzen_diode("D", "a", "gnd")
    r = p.simulate(b, t_end=te + 0.2e-6, dt=dt, engine="pwl", **kw)
    return _at(r, r.i("R"), 1.1e-6), r.n_split_steps


def _igbt_tail(dt, **kw):
    """Hard gate turn-off (a pulse-source jump off the grid), resistive load."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc", "gnd", 600.0)
    b.add_pulse_voltage_source("Vg", "g", "gnd", 0.0, 15.0, 0.0, 2e-6 + 1.3e-9,
                               0.0, 0.0, 0.0)
    b.add_resistor("Rload", "dc", "c", 6.0)
    b.add_igbt_level1("Q1", "c", "e", "g", 1.5, 0.05, 5.0, tau_tail=1e-6, k_tail=0.30)
    b.add_resistor("Rsense", "e", "gnd", 1e-4)
    n = b.graph.num_switches
    r = p.simulate(b, t_end=4e-6, dt=dt, engine="pwl",
                   switch_fn=lambda _t: p.SwitchStateMask(n), **kw)
    return _at(r, r.i("Rsense"), 3e-6), r.n_split_steps


@pytest.mark.parametrize("fn, dts", [
    (_sat_edge, (4e-6, 2e-6, 1e-6, 5e-7)),
    (_transformer, (4e-6, 2e-6, 1e-6, 5e-7)),
    (_lauritzen, (4e-9, 2e-9, 1e-9, 5e-10)),
    (_igbt_tail, (8e-9, 4e-9, 2e-9, 1e-9)),
], ids=["saturable", "transformer", "lauritzen", "igbt_tail"])
def test_stateful_devices_are_second_order_across_an_off_grid_event(fn, dts):
    """Every sub-step tells the device its own step and rule (trapezoidal
    dt1, then the SDIRK stages of gamma*dt2): a device stamped at the
    nominal dt would be first order (Richardson ratio 2), not 4."""
    on, n = _richardson(fn, dts, **SPLIT)
    assert n >= 1
    assert all(3.4 <= r <= 4.7 for r in on), on
    # The rule alone snaps the event to the grid: not second order.
    rule, _ = _richardson(fn, dts, **RULE)
    assert not all(3.4 <= r <= 4.7 for r in rule), rule


# --- interactions --------------------------------------------------------------
def _ccm_split(t_end, **kw):
    b = _buck_builder(1.0)
    fn = p.make_pwm_switch_fn(FSW, 0.3456, b.switch_index_of("S"),
                              b.graph.num_switches, 0.0)
    return p.simulate(b, t_end=t_end, dt=TB / 100, engine="pwl", switch_fn=fn,
                      **kw, **SPLIT)


@pytest.mark.parametrize("cut", [7.55, 7.355], ids=["mid-period", "in-window"])
def test_a_resumed_split_run_is_exact(cut):
    """Resume at a grid point inside the restart window that a split
    armed (cut 7.355 T: one step after the 0.3456 T edge) or away from it."""
    full = _ccm_split(20 * TB)
    first = _ccm_split(cut * TB)
    second = _ccm_split(20 * TB, resume_from=first.final_snapshot)
    tf = np.asarray(full.times)
    xf = np.asarray(full.states)[tf >= second.times[0] - 0.5 * TB / 100]
    xs = np.asarray(second.states)
    n = min(len(xf), len(xs))
    assert n > 1000
    assert np.abs(xf[:n] - xs[:n]).max() <= 1e-12 * np.abs(xf[:n]).max()
    assert full.n_split_steps >= 20       # the 0.3456 T edge, every period


def test_dc_start_with_the_split():
    res = _ccm_split(5 * TB, start_from_dc_op=True)
    assert res.n_split_steps == 5


def test_b_extra_fn_is_called_once_per_solve_time_through_a_split():
    """A split step calls b_extra_fn at dt1's end and at each SDIRK stage
    of the remainder; nothing discarded, times strictly increasing."""
    calls = []
    te = (10 + 0.4) * 1e-7
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S", "in", "a", GON, GOFF)
    b.add_resistor("R", "a", "c", 1e3)
    b.add_capacitor("C", "c", "gnd", 1e-9)

    def bx(t):
        calls.append(t)
        return np.zeros(b.pool.state_size(b.graph))
    r = p.simulate(b, t_end=20e-7 + 1e-15, dt=1e-7, engine="pwl", b_extra_fn=bx,
                   switch_fn=_closing_at(b, "S", te), **SPLIT)
    assert r.n_split_steps == 1
    n_steps = len(r.times) - 1
    # one per trapezoidal step, two per SDIRK step; the split step
    # three (dt1's end, then two stages) instead of one
    assert len(calls) == n_steps + r.n_restart_steps + 2
    assert np.all(np.diff(calls) > 0)
    assert te in calls


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_dt_retry_of_a_step_that_holds_an_edge():
    """The mains rectifier of test_hostile_circuits (a Newton diode) at
    dt = 1 ms needs the dt-retry ladder on the steps ending at 21, 22 and
    23 ms, with the rule as without it. A load switched in at 22.4 ms
    puts a gate edge inside a retried step: the ladder rolls back to the
    step snapshot and re-takes it WITHOUT the split (shortcut: interior
    events are ignored by a retry), the next step sees the new mask and
    restarts, and the output grid stays exact."""
    dt = 1e-3
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "ac", "gnd", 0.0, 400.0, 60.0)
    q = p.IdealDiodeParams()
    q.kappa = 100.0
    b.add_nonlinear_diode("D", "ac", "vout", q, allow_non_passive=True)
    b.add_resistor("R", "vout", "gnd", 50.0)
    b.add_capacitor("C", "vout", "gnd", 100e-6)
    b.add_switch("S", "vout", "x", GON, GOFF)
    b.add_resistor("R2", "x", "gnd", 5000.0)
    fn = _closing_at(b, "S", 22.4 * dt)
    r = p.simulate(b, t_end=3.4e-2, dt=dt, switch_fn=fn, engine="pwl", **SPLIT)
    assert [round(d.t / dt, 6) for d in r.dt_retries] == [21.0, 22.0, 23.0]
    assert r.n_split_steps == 0              # the edge's step was retried
    assert r.n_restart_steps == 3 + 3        # start window; the edge at 23 ms
    t = np.asarray(r.times)
    np.testing.assert_allclose(t, np.arange(len(t)) * dt, rtol=0, atol=1e-12)
    assert np.isfinite(np.asarray(r.states)).all()
    # Moved off the retried steps (20.4 ms) the same edge is split.
    r = p.simulate(b, t_end=3.4e-2, dt=dt, switch_fn=_closing_at(b, "S", 20.4 * dt),
                   engine="pwl", **SPLIT)
    assert r.n_split_steps == 1


def _rc_jump(dt, **kw):
    """Pulse 0 -> 10 V at 2.33 µs (off every grid used) into tau = 1 µs."""
    te = 2.33e-6
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "in", "gnd", 0.0, 10.0, te, 50e-6,
                               0.0, 0.0, 0.0)
    b.add_resistor("R", "in", "a", 1e3)
    b.add_capacitor("C", "a", "gnd", 1e-9)
    r = p.simulate(b, t_end=8e-6 + 1e-12, dt=dt, engine="pwl", **kw)
    t, v = np.asarray(r.times), np.asarray(r.v("a"))
    ex = np.where(t > te, 10 * (1 - np.exp(-(t - te) / 1e-6)), 0.0)
    return r, float(np.abs(v - ex).max())


def test_source_jump_inside_a_step_is_split():
    runs = [_rc_jump(dt, **SPLIT) for dt in (2e-7, 1e-7, 5e-8)]
    errs = [e for _, e in runs]
    assert all(r.n_split_steps == 1 for r, _ in runs)
    assert all(f >= 3.5 for f in _factors(errs)), errs
    rule = [_rc_jump(dt, **RULE)[1] for dt in (2e-7, 1e-7)]
    assert rule[0] > 10 * errs[0], (rule, errs)


# --- review findings M1, M3, M4 --------------------------------------------------
_TE, _DEAD = 10.37e-6, 1e-9


class _DeadTimePair:
    """S1 off at _TE, S2 on 1 ns later: two edges < snap (0.01 dt) apart."""

    def __init__(self, n, k1, k2):
        self.n, self.k1, self.k2 = n, k1, k2

    def __call__(self, t):
        m = p.SwitchStateMask(self.n)
        m.set(self.k1, t < _TE)
        m.set(self.k2, t >= _TE + _DEAD)
        return m

    def next_edge_after(self, t):
        for e in (_TE, _TE + _DEAD):
            if e > t:
                return e
        return math.inf


def _pair(dt=None, opaque=False, **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "p1", "gnd", 10.0)
    b.add_voltage_source("V2", "p2", "gnd", -5.0)
    b.add_switch("S1", "p1", "a", 1e3, 1e-9)
    b.add_switch("S2", "p2", "a", 1e3, 1e-9)
    b.add_resistor("Rp", "a", "gnd", 10.0)
    b.add_resistor("R", "a", "m", 1.0)
    b.add_inductor("L", "m", "gnd", 10e-6)
    fn = _DeadTimePair(b.graph.num_switches, b.switch_index_of("S1"),
                       b.switch_index_of("S2"))
    if dt is None:
        return p.simulate(b, t_end=32e-6, engine="trbdf2", switch_fn=fn,
                          rtol=1e-10, atol=1e-12)
    return p.simulate(b, t_end=32e-6, dt=dt, engine="pwl",
                      switch_fn=fn.__call__ if opaque else fn, **kw)


def test_a_dead_time_pair_inside_one_step_lands_both_edges():
    """Merged within snap, the interval after the pair samples the gate
    at its LAST edge (it sampled the first one: S2 stayed off until the
    next grid point)."""
    ref = float(np.asarray(_pair().i("L"))[-1])
    e = [float(np.asarray(_pair(dt, **SPLIT).i("L"))[-1]) - ref
         for dt in (4e-6, 2e-6, 1e-6)]
    # engine: -2.562e-2, -6.520e-3, -1.236e-3 (factors 3.93, 5.28)
    assert abs(e[0]) < 3e-2, e
    assert all(f >= 3.5 for f in _factors(e)), e


def test_an_opaque_gate_with_the_split_is_counted_and_warned():
    with pytest.warns(UserWarning, match="could not be located"):
        r = _pair(1e-6, opaque=True, **SPLIT)
    assert r.counters["unlocated_gate_steps"] == 1     # one change, both edges
    r = _pair(1e-6, **SPLIT)
    assert r.counters["unlocated_gate_steps"] == 0
    assert r.counters["split_event_overflows"] == 0


def test_a_declared_b_extra_jump_off_the_grid_is_split():
    te, tau, t_end = 2.037e-6, 1e-6, 6.4e-6

    def run(dt, **kw):
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "in", "gnd", 0.0)
        b.add_resistor("R", "in", "a", 1e3)
        b.add_capacitor("C", "a", "gnd", 1e-9)

        def bx(t):
            v = np.zeros(3)
            v[2] = 10.0 if t >= te else 0.0      # the source's KVL row
            return v
        bx.next_jump_after = lambda t: te if t < te else math.inf
        r = p.simulate(b, t_end=t_end, dt=dt, engine="pwl", b_extra_fn=bx,
                       **kw)
        ex = -10 * (1 - math.exp(-(t_end - te) / tau))
        return float(np.asarray(r.v("a"))[-1]) - ex, r.n_split_steps

    dts = (4e-7, 2e-7, 1e-7, 5e-8)
    split = [run(dt, **SPLIT) for dt in dts]
    rule = [run(dt, **RULE)[0] for dt in dts]
    assert [n for _, n in split] == [1, 1, 1, 1]
    e = [v for v, _ in split]
    # engine: -6.356e-3, -1.719e-3, -4.469e-4, -1.141e-4 (order 1.89,
    # 1.94, 1.97); the rule alone: -1.08e-2, -6.30e-3, +3.96e-3, -3.10e-4
    assert abs(e[0]) < 7e-3, e
    assert all(f >= 3.6 for f in _factors(e)), e
    assert abs(e[1]) < abs(rule[1]), (e, rule)


# --- the split x the devices of the integration branch ---------------------------
def _split_devices():
    import importlib.util
    from pathlib import Path
    path = (Path(__file__).resolve().parents[2] / "openspec" / "changes"
            / "add-fixed-step-event-splitting" / "probes" / "split_devices.py")
    if not path.exists():
        pytest.skip("probe not shipped")
    spec = importlib.util.spec_from_file_location("split_devices", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# engine, last two orders by successive differences (T/100..T/800):
# mlcc 1.98 1.99, dc_machine 1.94 1.98, controlled 1.97 1.99,
# pulse I 1.94 1.95, sine I 1.88 1.95, table I 1.93 1.97, V_f diode 1.98 1.99
@pytest.mark.parametrize("case", ["mlcc", "dc_machine", "controlled_sources",
                                  "pulse_current", "sine_current",
                                  "table_current", "diode_with_drop"])
def test_split_is_second_order_on_each_integrated_device(case):
    mod = _split_devices()
    _, orders, splits = mod.run(getattr(mod, case))
    assert splits >= 8
    assert orders[-1] >= 1.9 and orders[-2] >= 1.85, orders
