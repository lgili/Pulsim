"""The floating-bus bridge against its exact reference.

Truth: benchmarks/tools/bridge_ground_truth.py (mpmath, 50 digits;
benchmarks/BRIDGE_GROUND_TRUTH.md): the 30-40 ms time-weighted average of
dp - dn is 311.343881428885841697 V and the mode sequence is unique.

trbdf2 follows the exact mode sequence (the diode-event count varies 6-13
under rounding: leakage-scale decisions; not asserted here); its error is the integration
error at the default step ceiling (span/1000 = 4e-5 s), O(h^2): -2.47e-3 V
at rtol 2e-5 (7.9e-6 of the bus, inside rtol*|v| = 6.2e-3 V per step),
-1.8e-4 V with a 1e-5 s ceiling, -1.91e-4 V at rtol 1e-7. The 1e-15
rounding fuzz moves the average by <= 9e-6 V although the diode-event count
varies (leakage-scale decisions at the zero-current turn-offs). An average
over the samples t >= 30 ms divided by 10 ms, without clipping the window
at 30 ms, read 311.154 V (a 6 us slice of the window lost): that was the
"-0.19 V" of DECISIONS #19, a metric error.
"""
import numpy as np
import pytest

import pulsim as p

from .test_trbdf2_rounding_fuzz import FUZZ

TRUTH = 311.343881428885841697


def _bridge(d, form):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "s", "gnd", 0.0, 325.0 * (1 + d[0]), 50.0)
    b.add_resistor("Rs", "s", "a", 0.5 * (1 + d[1]))
    b.add_inductor("Ls", "a", "ac1", 1e-3 * (1 + d[2]))
    for nm, a, c in (("D1", "ac1", "dp"), ("D2", "gnd", "dp"),
                     ("D3", "dn", "ac1"), ("D4", "dn", "gnd")):
        if form == "branch":
            b.add_diode(nm, a, c, form="branch", R_on=1e-3, g_off=1e-9)
        else:
            b.add_diode(nm, a, c, 1e3, 1e-9)
    b.add_capacitor("C", "dp", "dn", 470e-6 * (1 + d[3]))
    b.add_resistor("RL", "dp", "dn", 100.0 * (1 + d[4]))
    b.add_resistor("Rtie", "dn", "gnd", 1e6 * (1 + d[5]))
    return b


def _window_avg(r, a=30e-3, b=40e-3):
    """Time-weighted mean over exactly [a, b] (end points interpolated)."""
    t = np.asarray(r.times)
    v = np.asarray(r.v("dp")) - np.asarray(r.v("dn"))
    m = (t > a) & (t < b)
    tt = np.concatenate(([a], t[m], [b]))
    vv = np.concatenate(([np.interp(a, t, v)], v[m], [np.interp(b, t, v)]))
    return float(np.trapezoid(vv, tt)) / (b - a)


@pytest.mark.parametrize("form", ["branch", "conductance"])
@pytest.mark.parametrize("rtol,bound", [(1e-7, 1e-3), (2e-5, 3e-3)])
def test_bus_average_is_the_exact_one_under_rounding_fuzz(form, rtol, bound):
    rng = np.random.default_rng(20261117)
    errs = []
    for k in range(20):
        d = np.zeros(6) if k == 0 else rng.uniform(-1, 1, 6) * FUZZ
        r = p.simulate(_bridge(d, form), t_end=40e-3, engine="trbdf2",
                       rtol=rtol, atol=rtol * 1e-3)
        errs.append(_window_avg(r) - TRUTH)
    errs = np.asarray(errs)
    assert np.abs(errs).max() <= bound, errs
    # rounding moves the answer far less than the engine's own error
    assert np.ptp(errs) <= 1e-4, errs


def test_the_exact_engine_matches_the_ground_truth():
    # integration batch 11: -4.5e-10 V measured (0.0272 V low before the
    # floating-island fix); 1e-6 V leaves room for platform rounding
    r = p.simulate(_bridge(np.zeros(6), "conductance"), t_end=40e-3,
                   engine="exact")
    assert abs(_window_avg(r) - TRUTH) <= 1e-6


# BRIDGE_GROUND_TRUTH.md, "Events (exact, 50 digits)": the 11 commutations
EVENTS = [(0.004453708135472069740740, "D1-D4-"), (0.010005784513217728980424, "D3+"),
          (0.013234043039166346336001, "D2+"), (0.016037029154868641651128, "D2-"),
          (0.019993816884401272433491, "D3-"), (0.023578074673711604264767, "D1+D4+"),
          (0.026164880107693963425638, "D1-D4-"), (0.030006040329416432127840, "D3+"),
          (0.033448621701173689951408, "D2+"), (0.036121739554460249850922, "D2-"),
          (0.039993904164045823079728, "D3-")]


def _commutations(b, r, dt):
    """[(t, 'D1-D4-'), ...]: the diode changes of the stored switch states, after the
    t = 0 start-up; changes 1.5 dt apart or less are one commutation (the D1/D4 tie,
    which float64 resolves only 1e-8 s after it, may land on two adjacent steps)."""
    t = np.asarray(r.times)
    s = np.asarray(r.switch_states)[:, [b.switch_index_of(f"D{j}") for j in range(1, 5)]]
    out = []
    for k in np.nonzero((s[1:] != s[:-1]).any(axis=1))[0] + 1:
        if t[k] <= 10 * dt:
            continue
        ch = {f"D{j + 1}{'+' if s[k, j] else '-'}" for j in range(4) if s[k, j] != s[k - 1, j]}
        if out and t[k] - out[-1][2] <= 1.5 * dt:
            out[-1][1].update(ch)
            out[-1][2] = t[k]
        else:
            out.append([t[k], ch, t[k]])
    return [(t0, "".join(sorted(ch, key=lambda c: c[1]))) for t0, ch, _ in out]


@pytest.mark.parametrize("form", ["branch", "conductance"])
def test_pwl_split_bus_is_the_same_under_rounding_fuzz(form):
    # The D1+D4+ turn-on (23.578 ms) is split at a D4 crossing read from round-off;
    # the restart's stage 1 then cycles (D4 on/off). That cycle used to make stage 2
    # skip its solve, so stage 1's state (at t + gamma*h) was kept as the step's:
    # -1.5045e-4 V instead of -3.733e-6 V on the branch form (the conductance form
    # -9.594e-6 V), chosen by 1e-15 of rounding: by the platform's libm on CI.
    dt = 1e-6
    rng = np.random.default_rng(7)
    errs = []
    for k in range(5):   # draws 1-4 hit the cycle on macOS arm64 before the fix
        d = np.zeros(6) if k == 0 else rng.uniform(-1, 1, 6) * FUZZ
        r = p.simulate(_bridge(d, form), t_end=40e-3, dt=dt, engine="pwl",
                       discontinuity_restart=True, enable_substep_state_correction=True)
        errs.append(_window_avg(r) - TRUTH)
    errs = np.asarray(errs)
    assert np.abs(errs).max() <= 1e-5, errs
    assert np.ptp(errs) <= 1e-6, errs


@pytest.mark.parametrize("form", ["branch", "conductance"])
@pytest.mark.parametrize("mode", [None, "lcp"])
def test_pwl_finds_the_exact_bus_and_the_eleven_commutations(form, mode):
    # pwl, restart + split, dt = 1 us: -3.733e-6 V measured for both forms and
    # both selections; every commutation lands within a step of the exact one
    dt = 1e-6
    b = _bridge(np.zeros(6), form)
    extra = {"mode_selection": mode} if mode else {}
    r = p.simulate(b, t_end=40e-3, dt=dt, engine="pwl", discontinuity_restart=True,
                   enable_substep_state_correction=True, **extra)
    assert abs(_window_avg(r) - TRUTH) <= 1e-5
    got = _commutations(b, r, dt)
    assert [c for _, c in got] == [c for _, c in EVENTS], got
    for (tg, c), (te, _) in zip(got, EVENTS):
        assert abs(tg - te) <= 1.5 * dt, (c, tg, te)
