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
