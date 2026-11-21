"""The exact engine's recorder integrals with TIME-VARYING sources
(fixes/exact-buck-integrals.md, "Time-varying sources").

The engine propagates a sine source exactly as an oscillator pair (sin ωt,
cos ωt) augmented to the mode, and a pulse / PWM / table source as level +
ramp per step. The recorders' closed forms (dsed/segment_integrals.hpp,
make_exo_mode) append the same inputs to the state as an exosystem, so a
window or period statistic is the trajectory's, not the chord between the
10 us samples (it used to be 1e-4 off, with a warning).

Truths, none from pulsim:
* RL and series LC on sines, derived by hand below (transient + steady
  state; superposition for several sources, a DC offset, three phases; the
  LC driven at, 1e-8 from, 1e-3 from and far from its own frequency), the
  integrals by mpmath quadrature of the closed forms at 40 digits;
* diode circuits (a half-wave RL rectifier, a single-phase and a
  three-phase bridge with line inductance, an RC on a pulse with finite
  edges): the repo's mpmath engine `benchmarks/tools/exact_truth.py`
  (version fa4b5c71), solved once offline; its numbers are pasted below
  with the script that produced them named.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import pulsim as p

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

TOL = 1e-11   # relative, default grid


def _run(b, t_end, recs, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # no line, no warning
        # release: the default output rule warns about steps of a defective mode it cannot
        # bound; that is not the recorder's line fallback this test forbids
        warnings.filterwarnings("ignore", message=".*output grid.*", category=RuntimeWarning)
        return p.simulate(b, t_end, engine="exact", recorders=recs, **{"integrator": "rk45", **kw})


def _rel(a, b, scale=None):
    return abs(float(a) - float(b)) / max(abs(float(b if scale is None else scale)), 1e-300)


def _check(r, truths, A, B, tol=TOL):
    """`truths`: probe -> y(t) (mpmath). ∫y relative to ∫|y| (the sum's own
    scale: near resonance ∫v(c) is 2600x smaller than ∫|v(c)|) and the rms
    relative to itself, over [A, B]."""
    T = mp.mpf(B) - mp.mpf(A)
    for k, fn in truths.items():
        pts = mp.linspace(mp.mpf(A), mp.mpf(B), 33)
        i1 = mp.quad(fn, pts)
        a1 = mp.quad(lambda t: abs(fn(t)), pts)
        i2 = mp.quad(lambda t: fn(t) ** 2, pts)
        got1, got2 = r.windows[k]["integral"][0], r.windows[k]["rms"][0]
        assert _rel(got1, i1, a1) <= tol, (k, "integral", got1, float(i1), float(a1))
        assert _rel(got2, mp.sqrt(i2 / T)) <= tol, (k, "rms", got2, float(mp.sqrt(i2 / T)))


# --- RL on a sum of sines + DC, by hand --------------------------------------

def _rl_terms(R, L, srcs):
    """i(t) of a series RL from i(0) = 0, driven by Σ (dc, amp, f, phase):
    per source  dc/R (1 − e^{−t/τ})  +  Ip sin(ωt + φ − θ) − Ip sin(φ − θ) e^{−t/τ},
    Ip = amp/|R + jωL|, θ = atan2(ωL, R)."""
    R, L = mp.mpf(R), mp.mpf(L)
    tau = L / R

    def i(t):
        s = mp.mpf(0)
        for dc, amp, f, ph in srcs:
            s += mp.mpf(dc) / R * (1 - mp.exp(-t / tau))
            if amp:
                w = 2 * mp.pi * mp.mpf(f)
                Ip = mp.mpf(amp) / mp.sqrt(R ** 2 + (w * L) ** 2)
                th = mp.atan2(w * L, R)
                s += Ip * mp.sin(w * t + ph - th) - Ip * mp.sin(ph - th) * mp.exp(-t / tau)
        return s

    def v(t):
        return sum(mp.mpf(dc) + mp.mpf(amp) * mp.sin(2 * mp.pi * mp.mpf(f) * t + ph)
                   for dc, amp, f, ph in srcs)

    return i, v


def test_rl_on_sines_dc_offset_and_two_frequencies():
    """Two series sine sources (50 Hz with a DC offset, 150 Hz, phases) into
    an RL; the window starts and ends inside steps."""
    srcs = [(3.0, 10.0, 50.0, 0.3), (0.0, 2.5, 150.0, -1.1)]
    R, L = 2.0, 10e-3
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V1", "in", "m", *srcs[0])
    b.add_sine_voltage_source("V3", "m", "gnd", *srcs[1])
    b.add_resistor("R", "in", "a", R)
    b.add_inductor("L", "a", "gnd", L)
    A, B = 1.234e-3, 17.891e-3
    rec = p.Recorder("r", ["i(L)", "v(a)", "v(in)*i(L)", "v(in)"], keep="none",
                     windows=[(A, B)])
    res = _run(b, 20e-3, [rec])
    cf = res.recorder_feed["closed_form"]
    assert cf["n_line"] == 0 and cf["n_closed"] > 0, cf
    i, v = _rl_terms(R, L, srcs)
    _check(res.recorded["r"], {"i(L)": i, "v(a)": lambda t: v(t) - R * i(t),
                               "v(in)*i(L)": lambda t: v(t) * i(t), "v(in)": v}, A, B)


def test_three_phase_set_folds_into_one_oscillator():
    """Three sources at 120 degrees (one oscillator pair in the mode), each
    into its own RL to a grounded star point."""
    R, L, amp, f = 1.5, 4e-3, 325.0, 50.0
    b = p.CircuitBuilder()
    for k, ph in enumerate((0.0, -2 * math.pi / 3, 2 * math.pi / 3)):
        b.add_sine_voltage_source(f"V{k}", f"s{k}", "gnd", 0.0, amp, f, ph + 0.2)
        b.add_resistor(f"R{k}", f"s{k}", f"a{k}", R)
        b.add_inductor(f"L{k}", f"a{k}", "gnd", L)
    A, B = 0.77e-3, 13.3e-3
    probes = [f"i(L{k})" for k in range(3)] + ["v(s0)*i(L0)+v(s1)*i(L1)+v(s2)*i(L2)"]
    res = _run(b, 15e-3, [p.Recorder("r", probes, keep="none", windows=[(A, B)])])
    assert res.recorder_feed["closed_form"]["n_line"] == 0
    terms = [_rl_terms(R, L, [(0.0, amp, f, ph + 0.2)])
             for ph in (0.0, -2 * math.pi / 3, 2 * math.pi / 3)]
    truths = {f"i(L{k})": terms[k][0] for k in range(3)}
    truths[probes[3]] = lambda t: sum(v(t) * i(t) for i, v in terms)
    _check(res.recorded["r"], truths, A, B)


# --- series LC at, near and far from its own frequency, by hand ---------------

def _lc(detune):
    """V sin(ωt) — L — C, no resistance: v'' + ω0² v = ω0² V sin ωt, v(0) =
    v'(0) = 0. ω ≠ ω0: v = V ω0²/(ω0² − ω²) (sin ωt − (ω/ω0) sin ω0 t);
    ω = ω0: v = V/2 (sin ω0 t − ω0 t cos ω0 t) (secular growth). i = C v'."""
    L, C, V = 1e-3, 1e-6, 10.0
    w0 = 1.0 / math.sqrt(L * C)
    f = w0 * (1.0 + detune) / (2 * math.pi)
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, V, f)
    b.add_inductor("L", "in", "c", L)
    b.add_capacitor("C", "c", "gnd", C)
    W0 = 1 / mp.sqrt(mp.mpf(L) * mp.mpf(C))
    W = 2 * mp.pi * mp.mpf(f)
    Cm = mp.mpf(C)
    if detune == 0.0:
        def vc(t):
            return V / 2 * (mp.sin(W0 * t) - W0 * t * mp.cos(W0 * t))

        def i(t):
            return Cm * V / 2 * W0 ** 2 * t * mp.sin(W0 * t)
    else:
        k = V * W0 ** 2 / (W0 ** 2 - W ** 2)

        def vc(t):
            return k * (mp.sin(W * t) - W / W0 * mp.sin(W0 * t))

        def i(t):
            return Cm * k * W * (mp.cos(W * t) - mp.cos(W0 * t))
    return b, {"i(L)": i, "v(c)": vc, "v(c)*i(L)": lambda t: vc(t) * i(t)}


@pytest.mark.parametrize("detune", [0.0, 1e-8, 1e-3, 1.7])
def test_series_lc_driven_at_near_and_far_from_its_resonance(detune):
    """The source's ±jω on the circuit's own ±jω0: a defective augmented
    mode (exactly at it), a near-defective one (1e-8, 1e-3: the u·κ^p rule
    sends the products to the Gramian form) and a far one (modal)."""
    b, truths = _lc(detune)
    A, B = 0.1234e-3, 1.8765e-3
    res = _run(b, 2e-3, [p.Recorder("r", list(truths), keep="none", windows=[(A, B)])],
               allow_implicit_segments=False)
    assert res.recorder_feed["closed_form"]["n_line"] == 0
    _check(res.recorded["r"], truths, A, B)


# --- diode circuits against the mpmath engine ---------------------------------
# openspec/changes/update-number-changing-release/fixes/exact_sources_truth/
# diode_truths.py (benchmarks/tools/exact_truth.py at fa4b5c71, 40 digits):
# window mean / rms of the columns, the exact energy of every element over the
# window (sources: delivered, so p(V) = -it) and over the run.

G = (1e3, 1e-9)
P3 = 2.0943951023931953


def _halfwave():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "s", "gnd", 0.0, 20.0, 1000.0, 0.4)
    b.add_diode("D1", "s", "a", *G)
    b.add_inductor("L", "a", "o", 2e-3)
    b.add_capacitor("C", "o", "gnd", 4.7e-6)
    b.add_resistor("RL", "o", "gnd", 50.0)
    return b


def _bridge_1ph():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "s", "gnd", 0.0, 325.0, 50.0)
    b.add_resistor("Rs", "s", "a", 0.5)
    b.add_inductor("Ls", "a", "ac1", 1e-3)
    for nm, a, c in (("D1", "ac1", "dp"), ("D2", "gnd", "dp"), ("D3", "dn", "ac1"),
                     ("D4", "dn", "gnd")):
        b.add_diode(nm, a, c, *G)
    b.add_capacitor("C", "dp", "dn", 470e-6)
    b.add_resistor("RL", "dp", "dn", 100.0)
    b.add_resistor("Rtie", "dn", "gnd", 1e6)
    return b


def _bridge_3ph():
    b = p.CircuitBuilder()
    for ph, n, ph0 in (("a", "Va", 0.0), ("b", "Vb", -P3), ("c", "Vc", P3)):
        b.add_sine_voltage_source(n, "s" + ph, "gnd", 0.0, 325.0, 50.0, ph0)
        b.add_inductor("L" + ph, "s" + ph, ph, 0.5e-3)
    for nm, a, c in (("D1", "a", "dp"), ("D3", "b", "dp"), ("D5", "c", "dp"),
                     ("D4", "dn", "a"), ("D6", "dn", "b"), ("D2", "dn", "c")):
        b.add_diode(nm, a, c, *G)
    b.add_capacitor("C", "dp", "dn", 220e-6)
    b.add_resistor("RL", "dp", "dn", 40.0)
    b.add_resistor("Rtie", "dn", "gnd", 1e6)
    return b


def _pulse_ramps():
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("Vp", "in", "gnd", 0.0, 10.0, 5e-6, 50e-6, 200e-6, 20e-6, 30e-6)
    b.add_resistor("R", "in", "a", 100.0)
    b.add_inductor("L", "a", "o", 1e-3)
    b.add_capacitor("C", "o", "gnd", 1e-7)
    b.add_resistor("RL", "o", "gnd", 1e3)
    return b


# name: (make, t_end, A, B, {column: (mean, rms)}, {element: window energy})
TRUTH = {
    "halfwave_rlc": (_halfwave, 6e-3, 0.7123e-3, 5.3321e-3,
                     {"i(L)": (0.2389839958753485, 0.4116187124808296),
                      "v(o)": (10.913873536995693, 13.645152242513841)},
                     {"RL": 0.017203227845536677, "D1": 7.838068051049384e-07,
                      "V": -0.01899768755088158}),
    "bridge_1ph": (_bridge_1ph, 30e-3, 7.31e-3, 27.77e-3,
                   {"i(Ls)": (-0.6157949606244998, 7.651876929138404),
                    "v(dp,dn)": (309.36681322482605, 309.9620438990747)},
                   {"Rs": 0.5989789861107023, "RL": 19.657245487445604,
                    "Rtie": 0.000517035693071294, "D1": 0.0004076041365811977,
                    "D2": 0.0007926999078229924, "D3": 0.0007920610713981463,
                    "D4": 0.0004069580071094272, "Vac": -21.85783982627384}),
    "bridge_3ph": (_bridge_3ph, 12e-3, 3.137e-3, 11.21e-3,
                   {"i(La)": (0.8736390357356065, 2.3395725414034874),
                    "i(Lb)": (6.986614666618448, 15.448025427386558),
                    "v(dp,dn)": (619.6005028747597, 628.2470739420087)},
                   {"RL": 79.6591944376371, "Rtie": 0.0005853591617751494,
                    "D1": 4.456623129511066e-05, "D3": 0.001928164882993402,
                    "D5": 2.7684770905803586e-06, "D4": 1.81046097342028e-06,
                    "D6": 1.1644008233754166e-06, "D2": 0.0019708703921011038,
                    "Va": -1.4234583321251446, "Vb": -14.948711641949794,
                    "Vc": -18.645421264582836}),
    "pulse_ramps": (_pulse_ramps, 600e-6, 37.3e-6, 512.9e-6,
                    {"i(L)": (0.0026469359893109747, 0.017717367504433715),
                     "v(o)": (4.129991076848562, 5.932582616939619)},
                    {"R": 1.4929327092817511e-05, "RL": 1.6738997162640807e-05,
                     "Vp": -2.7963114482156922e-05}),
}

# The engine's own trajectory limits the three-phase bridge: its end states
# are up to 1.8e-9 off the truth (i(La) at 8 ms; bus 7.7e-11 already at
# 4.9 ms) and it takes the truth's D2 turn-on at 5.0000003 ms and D5 turn-off
# 1.35 us later as one event; the integrals cannot be closer than that.
TRAJ = {"bridge_3ph": 1e-8}


@pytest.mark.parametrize("name", list(TRUTH))
def test_diode_and_pulse_circuits_against_the_mpmath_engine(name):
    """Window means and RMS (relative to the rms: a mean of an AC current is
    small) and every element's window energy (relative to the window's
    source energy, the ledger's scale: a diode's 4e-4 J is 5e4x below it)
    on the DEFAULT grid; windows start and end inside steps and hold diode
    commutations (12, 7 and 10 events)."""
    make, t_end, A, B, cols, energy = TRUTH[name]
    probes = list(cols) + [f"p({e})" for e in energy]
    res = _run(make(), t_end, [p.Recorder("r", probes, keep="none", windows=[(A, B)])])
    assert res.recorder_feed["closed_form"]["n_line"] == 0
    r = res.recorded["r"]
    tol = TRAJ.get(name, TOL)
    for c, (mean, rms) in cols.items():
        assert _rel(r.windows[c]["avg"][0], mean, rms) <= tol, (c, r.windows[c]["avg"][0], mean)
        assert _rel(r.windows[c]["rms"][0], rms) <= tol, (c, r.windows[c]["rms"][0], rms)
    src = sum(abs(v) for k, v in energy.items() if k.startswith("V"))
    for e, tru in energy.items():
        got = r.windows[f"p({e})"]["integral"][0]
        assert _rel(got, tru, src) <= tol, (e, got, tru)


def test_rectifier_energy_ledger_closes_on_the_default_grid():
    """Single-phase bridge with line inductance, [0, 30 ms] on the default
    grid: the closed-form integrals of every element's absorbed power (L
    included) plus the capacitor's stored energy at the end close to
    rounding (measured 1.0e-12 of the source's 54.38 J), and the source's
    energy is the mpmath truth's (54.384383077361804 J)."""
    els = ["Rs", "RL", "Rtie", "D1", "D2", "D3", "D4", "Ls", "Vac"]
    res = _run(_bridge_1ph(), 30e-3, [p.Recorder("r", [f"p({e})" for e in els], keep="none",
                                                 windows=[(0.0, 30e-3)])])
    E = {e: res.recorded["r"].windows[f"p({e})"]["integral"][0] for e in els}
    vc = res.v("dp")[-1] - res.v("dn")[-1]
    ledger = sum(E.values()) + 0.5 * 470e-6 * vc ** 2
    assert abs(ledger) <= 1e-11 * abs(E["Vac"]), ledger
    assert _rel(-E["Vac"], 54.384383077361804) <= TOL


def test_statistics_do_not_depend_on_the_output_grid():
    """Default grid, dt = 1e-7 and output_rtol = 1e-8 agree on the recorder
    statistics to 1e-10 (of the rms; the closed forms' rounding adds up over
    dt = 1e-7's 2.3e6 pieces: measured 2.4e-12)."""
    make, t_end, A, B, cols, energy = TRUTH["halfwave_rlc"]
    probes = list(cols) + [f"p({e})" for e in energy]
    runs = [_run(make(), t_end, [p.Recorder("r", probes, keep="none", windows=[(A, B)],
                                            period=1e-3)], **kw).recorded["r"]
            for kw in ({}, {"dt": 1e-7}, {"output_rtol": 1e-8})]
    for k in probes:
        sc = max(abs(runs[0].windows[k]["rms"][0]), 1e-300)
        for o in runs[1:]:
            for f in ("integral", "avg", "rms"):
                assert abs(o.windows[k][f][0] - runs[0].windows[k][f][0]) <= 1e-10 * sc * (
                    B - A if f == "integral" else 1), (k, f)
            np.testing.assert_allclose(o.periods[k]["avg"], runs[0].periods[k]["avg"],
                                       rtol=0, atol=1e-10 * sc)
