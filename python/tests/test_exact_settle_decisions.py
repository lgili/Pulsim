"""The exact engine's diode decisions at a source corner and at a located root (F5).

(a) A source corner on an exact zero: a pulse at 0 V starts its rise at a corner
    where the diode's v_D is exactly 0; the corner changes no mask, so no decision
    ran and the scan after it left the zero "to the mode decision": D1 stayed OFF
    under 10 V (DiodeModeInconsistentError at 6 us).
(b) A located turn-off kept ON: on the falling ramp i_D = 0 is located, but the
    OFF mode read the root's residual amplified by 1 + R*g_on = 1e5 (+1.35e-10 V)
    as a forward bias and kept D1 ON.

Truth: the circuits' own state equations, x' = A x + B u (u piecewise linear, a
source corner a break of its slope), solved per mode by the matrix exponential of
the augmented system [x; u; u'] in mpmath (40 digits); the diode's law (V_th = V_f
= 0): ON -> OFF when its current reaches 0, OFF -> ON when v_D reaches 0; at a
corner or start where the signal is zero, the sign of its first non-vanishing
derivative after the instant decides. The pulse cases were checked against
benchmarks/tools/exact_truth.py (feat/core-soft-switching-validation fa4b5c71):
turn-off at 59.2793718175008149 us (v0 = 0) and 58.9377110436850184 us
(v0 = -1), the same to 1e-24 s.
"""
from __future__ import annotations

import functools
import math

import numpy as np
import pytest

import pulsim as p

mp = pytest.importorskip("mpmath")

SELECTORS = ("legacy", "lcp", "auto")
FORMS = ("conductance", "branch")
G_ON, G_OFF, R, C, RL = 1e3, 1e-9, 100.0, 1e-7, 1e3
_I_OFF = 11.0 * G_OFF + 1e-12   # an OFF diode's g_off current at the circuits' largest |v_D|


def _pulse_pieces(v0, v1=10.0, td=5e-6, tr=1e-6, pw=50e-6, tf=30e-6):
    """(t_k, u(t_k), slope) of the pulse, corners as the engine rounds them."""
    c = [0.0, td, td + tr, td + (tr + pw), td + (tr + pw + tf)]
    return [(c[0], v0, 0.0), (c[1], v0, (v1 - v0) / tr), (c[2], v1, 0.0),
            (c[3], v1, (v0 - v1) / tf), (c[4], v0, 0.0)]


def _rc_model(r_src=0.0):
    """in -[r_src]- D1 - R - o, C || RL at o; x = [v_C]; g = drive - v_C."""
    def model(on):
        G = 1 / (1 / mp.mpf(G_ON if on else G_OFF) + mp.mpf(R) + mp.mpf(r_src))
        A = mp.matrix([[-(G + 1 / mp.mpf(RL)) / mp.mpf(C)]])
        B = mp.matrix([[G / mp.mpf(C)]])
        return A, B, mp.matrix([[-1]]), mp.mpf(1)
    return model


def _prefilter_model(R1=100.0, C1=1e-8):
    """in - R1 - m, C1 at m; m - D1 - R - o, C || RL at o; x = [v_m, v_C]."""
    def model(on):
        G = 1 / (1 / mp.mpf(G_ON if on else G_OFF) + mp.mpf(R))
        g1, c1, c = 1 / mp.mpf(R1), mp.mpf(C1), mp.mpf(C)
        A = mp.matrix([[-(g1 + G) / c1, G / c1], [G / c, -(G + 1 / mp.mpf(RL)) / c]])
        B = mp.matrix([[g1 / c1], [0]])
        return A, B, mp.matrix([[1, -1]]), mp.mpf(0)
    return model


@functools.lru_cache(maxsize=None)
def _truth_cached(key):
    return _truth(*key)


def _truth(model_key, pieces, t_end):
    """Events [(t, on)] and a sampler x(t) of the piecewise-LTI circuit."""
    mp.mp.dps = 40
    model = {"rc": _rc_model(), "rc_src1": _rc_model(1.0), "pre": _prefilter_model()}[model_key]
    pcs = [(mp.mpf(t), mp.mpf(u), mp.mpf(s)) for t, u, s in pieces]
    n = model(False)[0].rows

    def H(on):
        A, B, _, _ = model(on)
        h = mp.zeros(n + 2, n + 2)
        for i in range(n):
            for j in range(n):
                h[i, j] = A[i, j]
            h[i, n] = B[i, 0]
        h[n, n + 1] = 1
        return h

    Hs = {False: H(False), True: H(True)}

    def g_of(on, y):
        _, _, Cg, Dg = model(on)
        return sum(Cg[0, j] * y[j] for j in range(n)) + Dg * y[n]

    def flow(on, y, s):
        return mp.expm(Hs[on] * s) * y

    def lead_sign(on, y):
        # the first non-vanishing derivative of g along mode `on` (g = c y, y' = H y)
        Hk = mp.eye(n + 2)
        for _ in range(n + 3):
            v = g_of(on, Hk * y)
            if abs(v) > mp.mpf(10) ** -30:
                return 1 if v > 0 else -1
            Hk = Hs[on] * Hk
        return 0

    segs, events = [], []
    y = mp.matrix([0] * n + [pcs[0][1], pcs[0][2]])
    t, on, k, crossed = mp.mpf(0), False, 0, False
    T = mp.mpf(t_end)
    while True:
        t1 = pcs[k + 1][0] if k + 1 < len(pcs) else T
        t1 = min(t1, T)
        g0 = g_of(on, y)
        # decide: by value away from zero, by the leading derivative at a zero / a root
        want = (g0 > 0) if (abs(g0) > mp.mpf(10) ** -25 and not crossed) else (lead_sign(False, y) > 0)
        crossed = False
        if want != on:
            on = want
            events.append((t, on))
            continue
        segs.append((t, on, y))
        # a root of g in (t, t1): scan + bisection
        N = 200
        E = mp.expm(Hs[on] * (t1 - t) / N)
        ya, root = y, None
        for i in range(N):
            yb = E * ya
            ga, gb = g_of(on, ya), g_of(on, yb)
            if (ga > 0) != (gb > 0) and abs(ga) > mp.mpf(10) ** -25 and gb != 0:
                ta = t + (t1 - t) * i / N
                root = mp.findroot(lambda s: g_of(on, flow(on, ya, s - ta)),
                                   (ta, ta + (t1 - t) / N), solver="bisect")
                break
            ya = yb
        if root is not None:
            y = flow(on, y, root - t)
            t, on, crossed = root, not on, True
            events.append((t, on))
            continue
        y = flow(on, y, t1 - t)
        t = t1
        if t >= T:
            break
        k += 1
        y[n], y[n + 1] = pcs[k][1], pcs[k][2]   # the source's new piece (value, slope)

    def x_at(tq):
        tq = mp.mpf(tq)
        t0, m, y0 = max((s for s in segs if s[0] <= tq), key=lambda s: s[0])
        return flow(m, y0, tq - t0)

    return events, x_at


def _circuit(source, form, model="rc"):
    b = p.CircuitBuilder()
    src = source(b)
    kw = {} if form == "conductance" else {"form": "branch"}
    if model == "pre":
        b.add_resistor("R1", src, "m", 100.0)
        b.add_capacitor("C1", "m", "gnd", 1e-8)
        src = "m"
    b.add_diode("D1", src, "a", G_ON, G_OFF, **kw)
    b.add_resistor("R", "a", "o", R)
    b.add_capacitor("C", "o", "gnd", C)
    b.add_resistor("RL", "o", "gnd", RL)
    return b


def _pulse(v0):
    def add(b):
        b.add_pulse_voltage_source("Vp", "in", "gnd", v0, 10.0, 5e-6, 50e-6, 200e-6, 1e-6, 30e-6)
        return "in"
    return add


def _check(res, events, x_at, t_end, state_col=0, node="o", t_tol=1e-12, x_tol=1e-9):
    got = [t for t in res.event_times]
    want = [float(t) for t, _ in events]
    assert len(got) == len(want), (got, want)
    for a, b in zip(got, want):
        assert abs(a - b) <= t_tol, (got, want)
    t = np.asarray(res.times)
    v = np.asarray(res.v(node))
    idx = np.unique(np.linspace(0, len(t) - 1, 25).astype(int))
    for i in idx:
        assert abs(v[i] - float(x_at(t[i])[state_col])) <= x_tol, (t[i], v[i])
    # the diode law holds on every sample: no reverse current beyond g_off * 11 V
    assert np.all(np.asarray(res.i("D1")) >= -_I_OFF)


@pytest.mark.parametrize("selector", SELECTORS)
@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("v0", [0.0, -1.0])
def test_pulse_into_a_diode_and_rc(v0, form, selector):
    """(a) v0 = 0: the rise starts on v_D = 0 exactly -> ON at the corner.
    (b) v0 = -1: the turn-off located on the falling ramp is kept."""
    t_end = 100e-6
    events, x_at = _truth_cached(("rc", tuple(_pulse_pieces(v0)), t_end))
    if v0 == 0.0:
        assert events[0] == (mp.mpf(5e-6), True)
    res = p.simulate(_circuit(_pulse(v0), form), t_end, engine="exact", mode_selection=selector)
    _check(res, events, x_at, t_end)


@pytest.mark.parametrize("v0", [-math.ulp(10.0), -math.ulp(0.0), math.ulp(0.0), math.ulp(10.0)])
def test_a_corner_one_ulp_off_the_zero(v0):
    """v_D one ulp below / above zero before the corner (of zero, of the 10 V scale).
    Below: OFF until the corner, ON at it (the truth's turn-on is 1e-22 s after it).
    Above: forward-biased by an ulp, the start decides ON (the truth too) and the
    corner is no event. Either way D1 conducts from the corner on; the turn-off
    and every state against the truth."""
    t_end = 100e-6
    events, x_at = _truth_cached(("rc", tuple(_pulse_pieces(v0)), t_end))
    res = p.simulate(_circuit(_pulse(v0), "conductance"), t_end, engine="exact")
    early = [t for t in res.event_times if t <= 6e-6]
    late = [(t, o) for t, o in events if t > 6e-6]
    if v0 < 0:
        assert len(early) == 1 and abs(early[0] - 5e-6) <= 1e-12, early
    else:
        assert early == [], early
    t = np.asarray(res.times)
    assert np.all(np.asarray(res.i("D1"))[(t > 5e-6) & (t < 56e-6)] > 0)
    res_late = [t for t in res.event_times if t > 6e-6]
    assert len(res_late) == len(late) and all(abs(a - float(b)) <= 1e-12 for a, (b, _) in zip(res_late, late))
    _check(res, [(mp.mpf(e), True) for e in early] + late, x_at, t_end)


def test_a_corner_where_the_slope_is_zero_too():
    """g = 0 and dg/dt = 0 after the corner (an RC prefilter: v_m starts with zero
    slope), d2g/dt2 > 0: decided by the next derivative -> ON at the corner."""
    t_end = 100e-6
    events, x_at = _truth_cached(("pre", tuple(_pulse_pieces(0.0)), t_end))
    assert events[0] == (mp.mpf(5e-6), True)
    for selector in SELECTORS:
        res = p.simulate(_circuit(_pulse(0.0), "conductance", "pre"), t_end, engine="exact",
                         mode_selection=selector)
        _check(res, events, x_at, t_end, state_col=1)


def test_a_table_source_s_corners():
    """The same drive from a PWL table current source into 1 ohm (Thevenin 1 ohm in
    series): the table's corner at 5 us is on the zero."""
    t_end = 100e-6
    tk, ik = [0.0, 5e-6, 6e-6, 56e-6, 86e-6], [0.0, 0.0, 10.0, 10.0, 0.0]
    pieces = [(tk[k], ik[k], (ik[k + 1] - ik[k]) / (tk[k + 1] - tk[k])) for k in range(len(tk) - 1)]
    pieces.append((tk[-1], ik[-1], 0.0))
    events, x_at = _truth_cached(("rc_src1", tuple(pieces), t_end))
    assert events[0] == (mp.mpf(5e-6), True)

    def add(b):
        b.add_pwl_current_source("I", "in", "gnd", tk, ik)
        b.add_resistor("Rs", "in", "gnd", 1.0)
        return "in"

    for selector in SELECTORS:
        res = p.simulate(_circuit(add, "conductance"), t_end, engine="exact", mode_selection=selector)
        _check(res, events, x_at, t_end)


def test_a_ramp_ending_on_a_zero():
    """The fall ends at 0 V on a purely resistive diode path (no capacitor after
    it): the current reaches 0 exactly at the corner and stays 0 -> OFF there."""
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("Vp", "in", "gnd", 0.0, 10.0, 5e-6, 50e-6, 200e-6, 1e-6, 30e-6)
    b.add_diode("D1", "in", "a", G_ON, G_OFF)
    b.add_resistor("R", "a", "gnd", R)
    b.add_resistor("R2", "in", "m", 1e3)
    b.add_capacitor("C2", "m", "gnd", 1e-8)
    for selector in SELECTORS:
        res = p.simulate(b, 100e-6, engine="exact", mode_selection=selector)
        ev = list(res.event_times)
        assert abs(ev[0] - 5e-6) <= 1e-12 and abs(ev[-1] - 86e-6) <= 1e-12, ev
        t = np.asarray(res.times)
        i_d = np.asarray(res.i("D1"))
        assert np.all(i_d >= -_I_OFF)
        assert np.all(np.abs(i_d[t > 86e-6]) <= 1e-12)


def test_a_sine_zero_on_the_diode_s_zero():
    """A sine into D1 + R: the source's zero crossings ARE the diode's zeros, a turn-on
    at k/f and a turn-off at (k + 1/2)/f, every one."""
    f = 1e4
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "in", "gnd", 0.0, 10.0, f)
    b.add_diode("D1", "in", "a", G_ON, G_OFF)
    b.add_resistor("R", "a", "gnd", R)
    for selector in SELECTORS:
        res = p.simulate(b, 3.2e-4, engine="exact", mode_selection=selector)
        ev = np.asarray(res.event_times)
        want = np.array([k / (2 * f) for k in range(1, 7)])
        assert len(ev) == len(want), ev
        assert np.max(np.abs(ev - want)) <= 1e-12, ev
        assert np.all(np.asarray(res.i("D1")) >= -_I_OFF)


P3 = 2.0943951023931953


def _bridge_3ph():
    """Three-phase diode bridge with 0.5 mH line inductance (test_exact_integrals_sources)."""
    b = p.CircuitBuilder()
    for ph, n, ph0 in (("a", "Va", 0.0), ("b", "Vb", -P3), ("c", "Vc", P3)):
        b.add_sine_voltage_source(n, "s" + ph, "gnd", 0.0, 325.0, 50.0, ph0)
        b.add_inductor("L" + ph, "s" + ph, ph, 0.5e-3)
    for nm, a, c in (("D1", "a", "dp"), ("D3", "b", "dp"), ("D5", "c", "dp"),
                     ("D4", "dn", "a"), ("D6", "dn", "b"), ("D2", "dn", "c")):
        b.add_diode(nm, a, c, G_ON, G_OFF)
    b.add_capacitor("C", "dp", "dn", 220e-6)
    b.add_resistor("RL", "dp", "dn", 40.0)
    b.add_resistor("Rtie", "dn", "gnd", 1e6)
    return b


# benchmarks/tools/exact_truth.py (fa4b5c71), 40 digits, 0..6 ms: D1 OFF, D5 OFF, D2 ON with
# D6 still ON, D6 OFF 1.35 us later; V(dp, dn)(6 ms)
_BRIDGE_TRUTH = (3.7193567186268455e-4, 1.4902582186689645e-3,
                 5.0000002509129349e-3, 5.0013509430908255e-3)
_BRIDGE_V_END = 624.64746549969316


def _bridge_events(selector):
    res = p.simulate(_bridge_3ph(), 6e-3, engine="exact", mode_selection=selector)
    return np.asarray(res.event_times), res


def test_the_guard_keeps_the_overlap_of_d2_and_d6():
    """(c) The mode guard judged the settled mode (D6 still ON at D2's turn-on) at
    1e4 h_e and read D6's own turn-off 1.35 us later as a violation: it turned D6 OFF
    at once (3 events, not 4). A law broken only after the diode first moved into its
    admissible side past the parasitic transient is the mode's next event."""
    ev, res = _bridge_events("legacy")
    assert len(ev) == len(_BRIDGE_TRUTH), ev
    for got, want in zip(ev[1:], _BRIDGE_TRUTH[1:]):
        assert abs(got - want) <= 1e-12, (got, want)
    # the first turn-off is 2e-10 s early on base and tip alike (open, not this rule)
    assert abs(ev[0] - _BRIDGE_TRUTH[0]) <= 1e-9
    v = np.asarray(res.v("dp")) - np.asarray(res.v("dn"))
    assert abs(v[-1] - _BRIDGE_V_END) <= 1e-6


@pytest.mark.parametrize("selector", ["lcp", "auto"])
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "release finding F5 (c), lcp selection: the complementarity solve itself (run_selector_, "
    "no ties) returns D6 OFF at D2's turn-on (5.000000250913 ms); with every upper diode OFF D6 "
    "carries leakage-level current, a split the LCP does not resolve; the truth keeps D6 ON to "
    "5.001350943091 ms. The guard rule fixes the legacy selector only"))
def test_the_lcp_keeps_the_overlap_of_d2_and_d6(selector):
    ev, _ = _bridge_events(selector)
    assert len(ev) == len(_BRIDGE_TRUTH), ev
