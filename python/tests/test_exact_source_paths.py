"""The exact engine's recorder closed forms on the source paths the sources
slice left untested (reviewer R4, fixes/exact-buck-integrals.md), and the
engine findings met on the way (fixed by the F5 settle decisions; kept as regressions).

Truths, by hand: every circuit is first order, y' = (K u(t) - y)/tau with a
piecewise-linear input u = a + b s on each piece, so
  y(s) = K(a + b s) - K b tau + (y0 - K a + K b tau) e^{-s/tau},
and the window integrals are mpmath quadratures of that closed form (40
digits) — no pulsim number enters them. Sign conventions (a current
source's direction) are read from one stored sample: a convention, not a value.
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

TOL = 1e-11


def _first_order(u_pieces, K, tau, y0=0.0):
    """y(t) for y' = (K u - y)/tau, u given as [(t0, a, b), ...] (u = a + b(t - t0)
    from t0 to the next t0). Returns y(t) and u(t) (mpmath)."""
    K, tau = mp.mpf(K), mp.mpf(tau)
    starts = [mp.mpf(t0) for t0, _, _ in u_pieces]
    ys = [mp.mpf(y0)]
    for k, (t0, a, b) in enumerate(u_pieces[:-1]):
        s = starts[k + 1] - starts[k]
        a, b = mp.mpf(a), mp.mpf(b)
        ys.append(K * (a + b * s) - K * b * tau + (ys[-1] - K * a + K * b * tau) * mp.exp(-s / tau))

    def piece(t):
        k = max(j for j in range(len(starts)) if starts[j] <= t)
        return k, t - starts[k], mp.mpf(u_pieces[k][1]), mp.mpf(u_pieces[k][2])

    def y(t):
        k, s, a, b = piece(t)
        return K * (a + b * s) - K * b * tau + (ys[k] - K * a + K * b * tau) * mp.exp(-s / tau)

    def u(t):
        _, s, a, b = piece(t)
        return a + b * s

    return y, u, starts


def _quad(fn, A, B, cuts):
    pts = sorted({mp.mpf(A), mp.mpf(B)} | {c for c in cuts if mp.mpf(A) < c < mp.mpf(B)})
    return mp.quad(fn, pts)


def _check(r, truths, A, B, cuts):
    T = mp.mpf(B) - mp.mpf(A)
    for k, fn in truths.items():
        i1 = _quad(fn, A, B, cuts)
        a1 = _quad(lambda t: abs(fn(t)), A, B, cuts)
        i2 = _quad(lambda t: fn(t) ** 2, A, B, cuts)
        got1, got2 = r.windows[k]["integral"][0], r.windows[k]["rms"][0]
        assert src._rel(got1, i1, a1) <= TOL, (k, "integral", got1, float(i1))
        assert src._rel(got2, mp.sqrt(i2 / T)) <= TOL, (k, "rms", got2, float(mp.sqrt(i2 / T)))


def _run(b, t_end, probes, A, B):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # no line, no warning
        res = p.simulate(b, t_end, engine="exact", recorders=[
            p.Recorder("r", probes, keep="none", windows=[(A, B)])])
    cf = res.recorder_feed["closed_form"]
    assert cf["n_line"] == 0 and cf["n_closed"] > 0, cf
    return res


def test_a_table_current_source_s_own_current_and_power():
    """A PWL table current source into R || C: the source's own i(I) is a row
    of the exosystem (exo_current_row); v(o) its first-order response."""
    R, C = 50.0, 2e-6
    tk, ik = [0.0, 40e-6, 90e-6, 150e-6, 220e-6], [0.0, 0.3, 0.3, -0.1, 0.2]
    b = p.CircuitBuilder()
    b.add_pwl_current_source("I", "gnd", "o", tk, ik)
    b.add_resistor("R", "o", "gnd", R)
    b.add_capacitor("C", "o", "gnd", C)
    pieces = [(tk[k], ik[k], (ik[k + 1] - ik[k]) / (tk[k + 1] - tk[k])) for k in range(len(tk) - 1)]
    pieces.append((tk[-1], ik[-1], 0.0))
    v, i, cuts = _first_order(pieces, R, R * C)
    A, B = 13.7e-6, 251.3e-6
    # p(I) of a table source is refused by name (no sign convention for it):
    # its power is written as v·i
    with pytest.raises(ValueError, match="p\\(I\\)"):
        _run(b, 300e-6, ["p(I)"], A, B)
    res = _run(b, 300e-6, ["i(I)", "v(o)*i(I)", "v(o)"], A, B)
    # the source's own current sign from one sample (60 us: |i| = 0.3 A)
    k = int(np.argmin(abs(np.asarray(res.times) - 60e-6)))
    si, sv = float(np.sign(res.i("I")[k])), float(np.sign(res.v("o")[k]))
    assert abs(abs(res.i("I")[k]) - 0.3) < 1e-12
    _check(res.recorded["r"], {"i(I)": lambda t: si * i(t),
                               "v(o)*i(I)": lambda t: sv * si * v(t) * i(t),
                               "v(o)": lambda t: sv * v(t)}, A, B, cuts)


def _rl_pieces_pwm(vh, vl, f, d, t_end):
    out, T = [], 1.0 / f
    for k in range(int(t_end * f) + 1):
        out += [(k * T, vh, 0.0), (k * T + d * T, vl, 0.0)]
    return out


def test_a_pwm_source_s_level_into_an_rl():
    """A PWM voltage source (level per step, a corner a time event) into an RL:
    v(in) is the level itself, i(L) its first-order response."""
    R, L, f, d = 3.0, 0.2e-3, 20e3, 0.3
    b = p.CircuitBuilder()
    b.add_pwm_voltage_source("V", "in", "gnd", 12.0, -2.0, f, d)
    b.add_resistor("R", "in", "a", R)
    b.add_inductor("L", "a", "gnd", L)
    i, v, cuts = _first_order(_rl_pieces_pwm(12.0, -2.0, f, d, 1e-3), 1 / R, L / R)
    A, B = 21.3e-6, 487.9e-6
    res = _run(b, 0.5e-3, ["v(in)", "i(L)", "v(in)*i(L)"], A, B)
    _check(res.recorded["r"], {"v(in)": v, "i(L)": i, "v(in)*i(L)": lambda t: v(t) * i(t)},
           A, B, cuts)


def test_mixed_pwm_and_ramping_pulse_sources_in_one_circuit():
    """A PWM source in series with a pulse whose edges ramp (20 / 30 us): both
    exosystem kinds in one mode (levels, one slope state), into an RL."""
    R, L, f, d = 2.0, 0.1e-3, 25e3, 0.6
    b = p.CircuitBuilder()
    b.add_pwm_voltage_source("Vw", "in", "m", 5.0, 0.0, f, d)
    b.add_pulse_voltage_source("Vp", "m", "gnd", 0.0, 8.0, 30e-6, 100e-6, 0.0, 20e-6, 30e-6)
    b.add_resistor("R", "in", "a", R)
    b.add_inductor("L", "a", "gnd", L)
    # the pulse: 0 to 30 us, ramp to 8 V by 50 us, flat to 150 us, ramp to 0 by 180 us
    knots = [(0.0, 0.0), (30e-6, 0.0), (50e-6, 8.0), (150e-6, 8.0), (180e-6, 0.0)]
    pw = _rl_pieces_pwm(5.0, 0.0, f, d, 1e-3)
    cuts = sorted({t for t, _ in knots} | {t for t, _, _ in pw if t < 0.4e-3})

    def vp(t):   # (value, slope) of the pulse on the piece from t
        for (t0, v0), (t1, v1) in zip(knots[:-1], knots[1:]):
            if t0 <= t < t1:
                sl = (mp.mpf(v1) - v0) / (mp.mpf(t1) - mp.mpf(t0))
                return v0 + sl * (t - mp.mpf(t0)), sl
        return mp.mpf(0), mp.mpf(0)

    def vw(t):
        return next(a for t0, a, _ in reversed(pw) if t0 <= t)

    pieces = [(c, vw(c) + vp(c)[0], vp(c)[1]) for c in cuts]
    i, v, mcuts = _first_order(pieces, 1 / R, L / R)
    A, B = 7.7e-6, 377.7e-6
    res = _run(b, 0.4e-3, ["v(in)", "i(L)", "v(in)*i(L)"], A, B)
    _check(res.recorded["r"], {"v(in)": v, "i(L)": i, "v(in)*i(L)": lambda t: v(t) * i(t)},
           A, B, mcuts)


@pytest.mark.parametrize("f", [0.0, -50.0])
def test_a_sine_with_non_positive_frequency_is_its_dc_level(f):
    """Decision: f <= 0 is the model's documented DC case (models/
    sine_voltage_source.hpp value_at: v_dc, the amplitude and phase unused) on
    every engine; the exact engine's closed form agrees with that law."""
    R, L, vdc = 2.0, 1e-3, 3.0
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", vdc, 5.0, f, 0.7)
    b.add_resistor("R", "in", "a", R)
    b.add_inductor("L", "a", "gnd", L)
    i, v, cuts = _first_order([(0.0, vdc, 0.0)], 1 / R, L / R)
    A, B = 0.1234e-3, 1.7654e-3
    res = _run(b, 2e-3, ["v(in)", "i(L)", "v(in)*i(L)"], A, B)
    _check(res.recorded["r"], {"v(in)": v, "i(L)": i, "v(in)*i(L)": lambda t: v(t) * i(t)},
           A, B, cuts)


# --- engine findings, fixed by fix/core-release-f5-settle-decisions (regressions) ---

def _pulse_diode(v0):
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("Vp", "in", "gnd", v0, 10.0, 5e-6, 50e-6, 200e-6, 1e-6, 30e-6)
    b.add_diode("D1", "in", "a", 1e3, 1e-9)
    b.add_resistor("R", "a", "o", 100.0)
    b.add_capacitor("C", "o", "gnd", 1e-7)
    b.add_resistor("RL", "o", "gnd", 1e3)
    return b


def test_a_ramp_starting_on_a_diode_s_zero_turns_it_on():
    # was a strict xfail; fixed by the settle decision at a source corner (F5 (a))
    res = p.simulate(_pulse_diode(0.0), 100e-6, engine="exact")
    t_on = [t for t in res.event_times if t >= 5e-6]
    assert t_on and t_on[0] - 5e-6 < 1e-12


def test_a_turn_off_on_a_falling_ramp_is_kept():
    # was a strict xfail; fixed by F5 (b) (a located root's error is a time error).
    # OFF is g_off = 1e-9 S: the reverse leakage g_off * v_D (to -7.8 nA) is the OFF law.
    res = p.simulate(_pulse_diode(-1.0), 100e-6, engine="exact")
    assert [t for t in res.event_times if t > 50e-6] == pytest.approx([58.937711e-6], abs=1e-12)
    t = np.asarray(res.times)
    after = t > 59e-6
    v_d = np.asarray(res.v("in"))[after] - np.asarray(res.v("a"))[after]
    assert np.all(v_d < 0)
    np.testing.assert_allclose(np.asarray(res.i("D1"))[after], 1e-9 * v_d, rtol=1e-9, atol=1e-18)


# The three-phase bridge's merged D2 / D6 flips (the same circuit and truth instants) are
# test_exact_settle_decisions.py::test_the_overlap_of_d2_and_d6_is_kept (strict xfail, F5 (c)).
