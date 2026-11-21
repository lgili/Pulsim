"""The exact engine's recorder statistics on EVERY route it runs
(fixes/exact-buck-integrals.md, "The diode-free route").

A circuit without diodes, loss port, output rule or ticks runs on
``integrator='auto'`` (PEDSimulatorAuto). That scheduler had no step hooks,
so its recorders took the trapezoid of the stored samples, O(h^2) and
silent, while a rectifier (routed to the event-driven integrator) was
exact. Its exact steps now drive the same in-loop feed and closed forms;
its hooks only observe (stored samples and states bit-identical with and
without a recorder).

Truths, none from pulsim:
* a SYNCHRONOUS buck (two ideal switches, no diode) from rest, derived by
  hand: x = [i_L, v_C], one affine law x' = M x + c per mode (M is the same
  in both modes: g_on + g_off is), x(t) = V e^{Λt} V^{-1} (x0 + xp) - xp,
  the window integrals of y and y^2 in closed form per piece, mpmath at 40
  digits;
* RL and series LC on sines: the hand derivations of
  ``test_exact_integrals_sources.py`` (the same truths, the default route).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import pulsim as p

mp = pytest.importorskip("mpmath")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_exact_integrals_sources as src  # noqa: E402

# --- synchronous buck, by hand -------------------------------------------------

VIN, GON, GOFF, L, C, R = 24.0, 1e3, 1e-9, 100e-6, 10e-6, 5.0
F, D = 20e3, 0.4
A, B = 0.21e-3 + 1.234e-9, 0.87e-3 + 4.321e-9   # inside steps on every grid
T_END = 1e-3


def _sync_buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", VIN)
    b.add_switch("S1", "in", "sw", GON, GOFF)
    b.add_switch("S2", "sw", "gnd", GON, GOFF)
    b.add_inductor("L", "sw", "out", L)
    b.add_capacitor("C", "out", "gnd", C)
    b.add_resistor("R", "out", "gnd", R)
    s1, s2 = b.switch_index_of("S1"), b.switch_index_of("S2")
    edges = []
    for k in range(int(round(T_END * F)) + 1):
        t0 = k / F
        edges += [(t0, s1, True), (t0, s2, False),
                  (t0 + D / F, s1, False), (t0 + D / F, s2, True)]
    return b, p.make_edge_list_fn(edges, b.graph.num_switches)


def _sync_buck_truth():
    """{probe: (∫y, ∫y²)} over [A, B] and the state at B, 40 digits."""
    mp.mp.dps = 40
    gs = mp.mpf(GON) + mp.mpf(GOFF)
    Lm, Cm, Rm = mp.mpf(L), mp.mpf(C), mp.mpf(R)
    M = mp.matrix([[-1 / (Lm * gs), -1 / Lm], [1 / Cm, -1 / (Rm * Cm)]])
    lam, V = mp.eig(M)
    Vi = V ** -1

    def piece(x0, g1, tau, rows):
        """End state after tau, and per row l: (∫ l·x, ∫ (l·x)²) over [0, tau]."""
        c = mp.matrix([mp.mpf(g1) * VIN / (gs * Lm), 0])
        xp = mp.lu_solve(M, c)
        w = Vi * (x0 + xp)
        x1 = V * mp.matrix([mp.exp(lam[k] * tau) * w[k] for k in range(2)]) - xp

        def E(s):   # ∫_0^tau e^{s t} dt
            return tau if s == 0 else (mp.exp(s * tau) - 1) / s

        out = []
        for row in rows:
            a = [sum(row[i] * V[i, k] for i in range(2)) * w[k] for k in range(2)]
            d = -sum(row[i] * xp[i] for i in range(2))
            i1 = sum(a[k] * E(lam[k]) for k in range(2)) + d * tau
            i2 = (sum(a[j] * a[k] * E(lam[j] + lam[k]) for j in range(2) for k in range(2))
                  + 2 * d * sum(a[k] * E(lam[k]) for k in range(2)) + d * d * tau)
            out.append((mp.re(i1), mp.re(i2)))
        return mp.matrix([mp.re(x1[0]), mp.re(x1[1])]), out

    rows = [(1, 0), (0, 1)]   # i(L), v(out)
    acc = [[mp.mpf(0), mp.mpf(0)] for _ in rows]
    x = mp.matrix([0, 0])
    T = 1 / mp.mpf(F)
    cuts = []
    for k in range(int(round(T_END * F)) + 1):
        cuts += [k * T, k * T + mp.mpf(D) * T]
    cuts = sorted(set([c for c in cuts if c < mp.mpf(B)] + [mp.mpf(A), mp.mpf(B)]))
    for t0, t1 in zip(cuts[:-1], cuts[1:]):
        on = (t0 - mp.floor(t0 / T) * T) < mp.mpf(D) * T * (1 - mp.mpf(10) ** -30)
        x, ints = piece(x, GON if on else GOFF, t1 - t0, rows)
        if t0 >= mp.mpf(A):
            for j in range(len(rows)):
                acc[j][0] += ints[j][0]
                acc[j][1] += ints[j][1]
    il, vo = acc
    return {"i(L)": il, "v(out)": vo, "p(R)": (vo[1] / Rm, None)}, x


def test_diode_free_dc_buck_on_the_default_route_against_the_hand_truth():
    b, fn = _sync_buck()
    truths, xB = _sync_buck_truth()
    rec = p.Recorder("w", list(truths), keep="none", windows=[(A, B)])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # no line, no warning
        res = p.simulate(b, B, switch_fn=fn, engine="exact", recorders=[rec])
    # control: the route and the states (a state error is not an integral error)
    assert res.recorder_feed["mode"] == "in-loop", res.recorder_feed
    cf = res.recorder_feed["closed_form"]
    assert cf["n_line"] == 0 and cf["n_closed"] > 0, cf
    e = max(abs(res.i("L")[-1] / float(xB[0]) - 1), abs(res.v("out")[-1] / float(xB[1]) - 1))
    if e > 1e-12:
        raise RuntimeError(f"exact-engine state at B off by {e:.2e}")
    w = res.recorded["w"].windows
    T = B - A
    for q, (i1, i2) in truths.items():
        assert abs(w[q]["integral"][0] / float(i1) - 1) < 1e-11, (q, w[q]["integral"][0], i1)
        if i2 is not None:
            rms = mp.sqrt(i2 / T)
            assert abs(w[q]["rms"][0] / float(rms) - 1) < 1e-11, (q, w[q]["rms"][0], rms)


@pytest.mark.parametrize("detune", [0.0, 1e-3, 1.7])
def test_series_lc_on_a_sine_on_the_default_route(detune):
    """The predecessor's LC truths (src._lc), integrator left at 'auto'."""
    b, truths = src._lc(detune)
    A_, B_ = 0.1234e-3, 1.8765e-3
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        # release: the default output rule warns about steps of a defective mode it cannot
        # bound; that is not the recorder's line fallback this test forbids
        warnings.filterwarnings("ignore", message=".*output grid.*", category=RuntimeWarning)
        res = p.simulate(b, 2e-3, engine="exact", allow_implicit_segments=False,
                         recorders=[p.Recorder("r", list(truths), keep="none",
                                               windows=[(A_, B_)])])
    assert res.recorder_feed["closed_form"]["n_line"] == 0, res.recorder_feed
    src._check(res.recorded["r"], truths, A_, B_)


def test_rl_on_sines_on_the_default_route():
    srcs = [(3.0, 10.0, 50.0, 0.3), (0.0, 2.5, 150.0, -1.1)]
    R_, L_ = 2.0, 10e-3
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V1", "in", "m", *srcs[0])
    b.add_sine_voltage_source("V3", "m", "gnd", *srcs[1])
    b.add_resistor("R", "in", "a", R_)
    b.add_inductor("L", "a", "gnd", L_)
    A_, B_ = 1.234e-3, 17.891e-3
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = p.simulate(b, 20e-3, engine="exact", recorders=[
            p.Recorder("r", ["i(L)", "v(in)*i(L)"], keep="none", windows=[(A_, B_)])])
    assert res.recorder_feed["closed_form"]["n_line"] == 0
    i, v = src._rl_terms(R_, L_, srcs)
    src._check(res.recorded["r"], {"i(L)": i, "v(in)*i(L)": lambda t: v(t) * i(t)}, A_, B_)


# --- every route x {with, without recorder} ------------------------------------

def _halfwave():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "s", "gnd", 0.0, 20.0, 1000.0, 0.4)
    b.add_diode("D1", "s", "a", 1e3, 1e-9)
    b.add_resistor("R", "a", "o", 10.0)
    b.add_inductor("L", "o", "gnd", 1e-3)
    return b, None


ROUTES = {   # name: (circuit, options, expected recorder path, the binding that runs)
    # release: the default output rule runs on the event scheduler (the "output_rtol" row);
    # the diode-free scheduler is reached with the rule off
    "auto (diode-free)": (_sync_buck, {"output_rtol": 0}, "closed", "auto"),
    "rk45": (_sync_buck, {"integrator": "rk45"}, "closed", "ped"),
    "auto -> rk45 (diode)": (_halfwave, {}, "closed", "ped"),
    "output_rtol": (_sync_buck, {"output_rtol": 1e-6}, "closed", "ped"),
    "bdf2": (_sync_buck, {"engine": "dsed", "integrator": "bdf2"}, "stored", "bdf2"),
    "t_start": (_sync_buck, {"t_start": 1e-5, "output_rtol": 0}, "stored", "auto"),
}


@pytest.mark.parametrize("route", list(ROUTES))
def test_every_route_is_closed_form_or_a_counted_warned_fallback(route, monkeypatch):
    make, kw, expect, sched = ROUTES[route]
    ran = []
    for name in ("auto", "ped", "bdf2"):
        fn = getattr(p._pulsim, f"run_{name}_native_builder")
        monkeypatch.setattr(p._pulsim, f"run_{name}_native_builder",
                            lambda *a, _f=fn, _n=name, **k: (ran.append(_n), _f(*a, **k))[1])
    kw = {"engine": "exact", **kw}
    t_end = 0.4e-3

    def run(recs):
        b, fn = make()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = p.simulate(b, t_end, switch_fn=fn, recorders=recs, **kw)
        return res, [w for w in caught if "recorders" in str(w.message)]

    base, w0 = run(None)
    assert not w0 and ran == [sched], ran   # the route is the scheduler named
    rec = p.Recorder("w", ["i(L)"], keep="none", windows=[(0.1e-3, 0.3e-3)])
    res, w1 = run([rec])
    # the feed observes: what is stepped and stored does not move
    assert np.array_equal(np.asarray(base.times), np.asarray(res.times))
    assert np.array_equal(np.asarray(base.i("L")), np.asarray(res.i("L")))
    feed = res.recorder_feed
    if expect == "closed":
        assert feed["mode"] == "in-loop" and feed["closed_form"]["n_line"] == 0, feed
        assert feed["closed_form"]["n_closed"] > 0 and not w1, (feed, w1)
    else:   # the trapezoid of the stored samples: counted and warned, never silent
        assert feed["mode"] == "stored-arrays" and feed["why"], feed
        assert len(w1) == 1 and "trapezoid" in str(w1[0].message), w1
