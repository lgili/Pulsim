"""Every pair of stateful device kinds composes on the kernel-built refresh.

add-unified-device-interface step 2: the Newton refresh is one kernel
function that stamps the static devices and then every stateful history.
The chain it replaced lost every earlier stamp when one wrapper captured
the raw callback (a disconnected saturable inductor erased an IGBT tail:
16.87 mJ -> 0.045 mJ). Here each kind runs alone and beside each other
kind on its own, electrically separate nodes: its own waveform must not
move beyond Newton tolerance.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import pulsim as p


def _sat(b, k):
    b.add_sine_voltage_source(f"V{k}", f"{k}a", "gnd", v_dc=0.0,
                              v_amplitude=50.0, frequency=50e3, phase=0.0)
    b.add_resistor(f"R{k}", f"{k}a", f"{k}m", 0.5)
    b.add_saturable_inductor(f"L{k}", f"{k}m", "gnd", 1e-5, 5.0, 5e-7)
    return f"{k}m"


def _coss(b, k):
    b.add_sine_voltage_source(f"V{k}", f"{k}a", "gnd", v_dc=60.0,
                              v_amplitude=50.0, frequency=100e3, phase=0.0)
    b.add_resistor(f"R{k}", f"{k}a", f"{k}c", 100.0)
    b.add_nonlinear_capacitor(f"C{k}", f"{k}c", "gnd", 2e-9, 25.0, 0.5)
    return f"{k}c"


def _laur(b, k):
    b.add_sine_voltage_source(f"V{k}", f"{k}a", "gnd", v_dc=0.0,
                              v_amplitude=5.0, frequency=1e6, phase=0.0)
    b.add_resistor(f"R{k}", f"{k}a", f"{k}k", 1.0)
    b.add_lauritzen_diode(f"D{k}", f"{k}k", "gnd", tau=1e-7, T_M=1e-8)
    return f"{k}k"


def _tail(b, k):
    b.add_voltage_source(f"Vdc{k}", f"{k}dc", "gnd", 600.0)
    b.add_pulse_voltage_source(f"Vg{k}", f"{k}g", "gnd", 0.0, 15.0, 0.0,
                               5e-6, 0.0, 1e-8, 1e-8)
    b.add_resistor(f"Rl{k}", f"{k}dc", f"{k}c", 6.0)
    b.add_igbt_level1(f"Q{k}", f"{k}c", "gnd", f"{k}g", 1.5, 0.05, 5.0,
                      tau_tail=1e-6, k_tail=0.30)
    return f"{k}c"


KINDS = {"saturable": _sat, "coss": _coss, "lauritzen": _laur,
         "igbt_tail": _tail}


def _probe(res, kind, node):
    """The kind's own state: the inductor current, else its node voltage."""
    if kind == "saturable":
        return np.asarray(res.i("La"))
    return np.asarray(res.v(node))


def _run(kinds, dt, engine="pwl"):
    b = p.CircuitBuilder()
    nodes = {kind: KINDS[kind](b, "ab"[i]) for i, kind in enumerate(kinds)}
    kw = {"dt": dt} if engine == "pwl" else {"rtol": 1e-6}
    res = p.simulate(b, t_end=2000 * dt, engine=engine, **kw)
    return res, nodes


# The MNA PMSM is not in this table: alone, with the 2.0 builder parameters, it
# takes a dt/2 retry on EVERY step at dt = 1 us, 100 ns and 10 ns (Newton
# stalls on its branch rows against the absolute tolerance; same on the
# build before this change), so a partner would be integrated on another
# grid and move by its own discretisation error, not by a lost stamp.
PAIRS = list(itertools.permutations(KINDS, 2))


@pytest.mark.parametrize("a,b", PAIRS)
def test_a_kind_beside_another_keeps_its_single_kind_result(a, b):
    alone, n1 = _run([a], 1e-8)
    pair, n2 = _run([a, b], 1e-8)
    ref = _probe(alone, a, n1[a])
    got = _probe(pair, a, n2[a])
    assert np.all(np.isfinite(got))
    # Same time grid (a partner that forced retries would move the result
    # by discretisation error, not by a lost stamp) ...
    assert np.array_equal(np.asarray(pair.times), np.asarray(alone.times))
    # ... so only Newton's stopping point differs, and the waveform agrees
    # to far better than 1e-6 (a lost stamp moves it by O(1)).
    scale = max(np.abs(ref).max(), 1e-12)
    assert np.abs(got - ref).max() <= 1e-6 * scale, (a, b)


def _time_weighted(res, y):
    """Mean and RMS over the run, time-weighted (adaptive grid)."""
    t = np.asarray(res.times)
    span = t[-1] - t[0]
    return (np.trapezoid(y, t) / span,
            np.sqrt(np.trapezoid(y * y, t) / span))


@pytest.mark.parametrize("a,b", PAIRS)
def test_on_trbdf2_a_kind_beside_another_keeps_its_result(a, b):
    """The TR-BDF2 refresh is the same kernel function. The partner's
    events shift the adaptive grid, and a sample-by-sample compare then
    measures the grid offset across a steep edge (0.35 V of 5 V on the
    Lauritzen snap-off), so the time-weighted mean and RMS are compared:
    a lost stamp moves them by O(1)."""
    alone, n1 = _run([a], 1e-8, "trbdf2")
    pair, n2 = _run([a, b], 1e-8, "trbdf2")
    ref = _probe(alone, a, n1[a])
    got = _probe(pair, a, n2[a])
    assert np.all(np.isfinite(got))
    scale = max(np.abs(ref).max(), 1e-12)
    for r, g in zip(_time_weighted(alone, ref), _time_weighted(pair, got)):
        assert abs(g - r) <= 1e-3 * scale, (a, b, r, g)


def test_enable_nonlinear_refresh_is_deprecated_and_changes_nothing():
    """On a STATIC Newton device. None and True were the same before this
    change too; what is new is the warning."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", v_dc=0.0, v_amplitude=5.0,
                              frequency=1e5, phase=0.0)
    b.add_shockley_diode("D", "a", "o")
    b.add_resistor("R", "o", "gnd", 10.0)
    plain = p.simulate(b, t_end=2e-5, dt=1e-8, engine="pwl")
    with pytest.warns(DeprecationWarning, match="enable_nonlinear_refresh"):
        forced = p.simulate(b, t_end=2e-5, dt=1e-8, engine="pwl",
                            enable_nonlinear_refresh=True)
    assert np.array_equal(np.asarray(plain.states), np.asarray(forced.states))
    # and the diode really conducts: the half-wave reaches the load
    assert np.asarray(plain.v("o")).max() > 3.0
