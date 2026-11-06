"""Time-weighted statistics (`pulsim.stats`).

Every analytic case below uses a deliberately NON-uniform grid --
samples clustered the way a variable-step engine clusters them
around an event -- on which `y.mean()` is far from the truth. The
helpers integrate the piecewise-linear waveform through the samples
exactly, so on piecewise-linear signals (ramps, triangles, steps
with a repeated event timestamp) they must be exact to rounding.
"""

import math

import numpy as np
import pytest

import pulsim as p
from pulsim.stats import time_window


def _clustered(n_dense=2000, n_sparse=10, split=0.01, t_end=1.0):
    """`n_dense` samples in [0, split), `n_sparse` across the rest."""
    return np.concatenate((np.linspace(0.0, split, n_dense, endpoint=False),
                           np.linspace(split, t_end, n_sparse)))


# --------------------------------------------------------------------------
# Analytic answers on clustered grids
# --------------------------------------------------------------------------

def test_ramp_on_a_clustered_grid_is_exact_where_mean_is_not():
    t = _clustered()
    y = t.copy()                     # y = t on [0, 1]
    assert abs(y.mean() - 0.5) > 0.4                 # the trap: ~0.01
    assert p.time_average(t, y) == pytest.approx(0.5, abs=1e-14)
    assert p.time_rms(t, y) == pytest.approx(1.0 / math.sqrt(3.0), abs=1e-14)


def test_triangle_rms_is_exact_for_the_piecewise_linear_waveform():
    # Triangle 0 -> 1 -> 0 -> -1 -> 0 over one period, its corners on
    # samples, with the rising edge sampled 500x more densely.
    seg = [np.linspace(0.0, 0.25, 1000, endpoint=False),
           np.linspace(0.25, 0.75, 3, endpoint=False),
           np.linspace(0.75, 1.0, 2)]
    t = np.concatenate(seg)
    y = np.interp(t, [0.0, 0.25, 0.75, 1.0], [0.0, 1.0, -1.0, 0.0])
    assert abs(y.mean()) > 0.3
    assert p.time_average(t, y) == pytest.approx(0.0, abs=1e-14)
    assert p.time_rms(t, y) == pytest.approx(1.0 / math.sqrt(3.0), rel=1e-13)


def test_step_with_a_repeated_event_timestamp():
    # The engines record an event instant twice, once per side of the
    # jump. y = 1 on [0, 0.3), 0 on [0.3, 1]; the "on" interval is
    # sampled 1000x, the "off" interval 5x.
    t_on = np.linspace(0.0, 0.3, 1001)
    t_off = np.linspace(0.3, 1.0, 5)
    t = np.concatenate((t_on, t_off))            # 0.3 appears twice
    y = np.concatenate((np.ones_like(t_on), np.zeros_like(t_off)))
    assert np.count_nonzero(np.diff(t) == 0.0) == 1
    assert y.mean() > 0.99
    assert p.time_average(t, y) == pytest.approx(0.3, abs=1e-14)
    assert p.time_rms(t, y) == pytest.approx(math.sqrt(0.3), abs=1e-14)
    # A window that starts or ends ON the repeated instant takes the
    # side of the jump that lies inside the window.
    assert p.time_average(t, y, t_from=0.3) == pytest.approx(0.0, abs=1e-15)
    assert p.time_average(t, y, t_to=0.3) == pytest.approx(1.0, abs=1e-15)


def test_window_edges_between_samples_are_interpolated():
    t = np.array([0.0, 0.37, 0.81, 1.0])
    y = 2.0 * t + 1.0
    # Mean of 2t+1 over [0.1, 0.9] is 2*0.5+1 = 2; neither edge is a
    # sample.
    assert p.time_average(t, y, 0.1, 0.9) == pytest.approx(2.0, abs=1e-14)
    # A window inside a single interval.
    assert p.time_average(t, y, 0.4, 0.5) == pytest.approx(1.9, abs=1e-14)
    tw, yw = time_window(t, y, 0.1, 0.9)
    assert tw[0] == 0.1 and tw[-1] == 0.9
    assert yw[0] == pytest.approx(1.2) and yw[-1] == pytest.approx(2.8)
    np.testing.assert_array_equal(tw[1:-1], [0.37, 0.81])


def test_sine_on_a_random_clustered_grid():
    rng = np.random.default_rng(1234)
    f = 50.0
    # Two periods; 90 % of the samples crammed into the first 2 ms
    # (positive half-wave) -- the sample mean is badly biased.
    t = np.unique(np.concatenate((rng.uniform(0.0, 2e-3, 9000),
                                  rng.uniform(2e-3, 0.04, 1000),
                                  [0.0, 0.04])))
    y = 3.0 + 2.0 * np.sin(2 * np.pi * f * t)
    assert abs(y.mean() - 3.0) > 0.5
    assert p.time_average(t, y) == pytest.approx(3.0, abs=1e-4)
    # Not exact: a sine is not piecewise linear. The sparse stretch has
    # ~38 us between samples; the chord error on y**2 is ~1e-5.
    assert p.time_rms(t, y) == pytest.approx(math.sqrt(9.0 + 2.0), rel=1e-4)


def test_uniform_grid_matches_trapezoid_and_is_close_to_mean():
    t = np.linspace(0.0, 1e-3, 10001)
    y = np.sin(2 * np.pi * 3e3 * t) ** 2
    ta = p.time_average(t, y)
    assert ta == pytest.approx(np.trapezoid(y, t) / 1e-3, rel=1e-12)
    assert ta == pytest.approx(y.mean(), rel=1e-3)


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kw, msg", [
    (dict(t_from=0.5, t_to=0.5), "empty time window"),
    (dict(t_from=0.7, t_to=0.2), "empty time window"),
    (dict(t_from=-0.1), "outside the simulated span"),
    (dict(t_to=1.5), "outside the simulated span"),
    (dict(t_from=float("nan")), "finite"),
])
def test_bad_windows_raise(kw, msg):
    t = np.linspace(0.0, 1.0, 11)
    with pytest.raises(ValueError, match=msg):
        p.time_average(t, t, **kw)
    with pytest.raises(ValueError, match=msg):
        p.time_rms(t, t, **kw)


@pytest.mark.parametrize("t, y, msg", [
    ([0.0, 0.2, 0.1], [1.0, 2.0, 3.0], "non-decreasing"),
    ([0.0, 1.0], [1.0, 2.0, 3.0], "same shape"),
    ([0.0], [1.0], "at least two samples"),
    ([[0.0, 1.0]], [[1.0, 2.0]], "1-D"),
    ([0.0, float("inf")], [1.0, 2.0], "NaN or inf"),
    ([0.0, 0.5, 1.0], [1.0, float("nan"), 2.0], "NaN or inf"),
    ([0.0, 0.0], [1.0, 2.0], "empty time window"),
])
def test_malformed_traces_raise(t, y, msg):
    with pytest.raises(ValueError, match=msg):
        p.time_average(t, y)


def test_nan_outside_the_window_is_ignored():
    t = np.linspace(0.0, 1.0, 11)
    y = t.copy()
    y[0] = np.nan                 # e.g. an undefined first sample
    assert p.time_average(t, y, t_from=0.2) == pytest.approx(0.6)


# --------------------------------------------------------------------------
# steady_state_stats on real results
# --------------------------------------------------------------------------

def _rectifier():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", v_dc=0.0, v_amplitude=20.0,
                              frequency=1e3, phase=0.0)
    b.add_inductor("Lc", "a", "m", 111e-6)
    b.add_diode("D", "m", "o", 1e3, 1e-9, 0.7)
    b.add_resistor("R", "o", "gnd", 2.0)
    b.add_capacitor("C", "o", "gnd", 10e-6)
    return b


@pytest.fixture(scope="module")
def rectifier_runs():
    fixed = p.simulate(_rectifier(), t_end=3e-3, dt=2e-7, engine="pwl")
    variable = p.simulate(_rectifier(), t_end=3e-3, dt=4e-6, rtol=1e-6,
                          atol=1e-9, engine="trbdf2")
    return fixed, variable


def test_steady_state_stats_agree_across_engines(rectifier_runs):
    """The case that started this: on this rectifier the sample mean
    of the TR-BDF2 trace read 3.04 V against 6.33 V. Time-weighted,
    the two engines agree."""
    fixed, variable = rectifier_runs
    sf = p.steady_state_stats(fixed, "o", t_from=2e-3)
    sv = p.steady_state_stats(variable, "o", t_from=2e-3)
    assert sf.mean > 5.0
    assert sv.mean == pytest.approx(sf.mean, rel=5e-3)
    assert sv.rms == pytest.approx(sf.rms, rel=5e-3)
    assert (sf.t_from, sf.t_to) == (2e-3, pytest.approx(3e-3))
    assert sf.ripple_pp == pytest.approx(sf.max - sf.min)


def test_steady_state_stats_window_and_lookup(rectifier_runs):
    fixed, _ = rectifier_runs
    by_last = p.steady_state_stats(fixed, "o", last=1e-3)
    by_from = p.steady_state_stats(fixed, "o", t_from=2e-3)
    assert by_last.mean == pytest.approx(by_from.mean, rel=1e-12)
    # A branch name resolves to its current; an array is taken as is.
    i_l = p.steady_state_stats(fixed, "Lc", t_from=2e-3)
    i_l2 = p.steady_state_stats(fixed, fixed.i("Lc"), t_from=2e-3)
    assert i_l.mean == i_l2.mean
    # In steady state the capacitor carries no DC, so the inductor's
    # average current is the load's: V_avg / R.
    assert i_l.mean == pytest.approx(by_from.mean / 2.0, rel=2e-2)
    v = p.steady_state_stats(fixed, "o", t_from=2e-3, kind="v")
    assert v.mean == by_from.mean


def test_steady_state_stats_refusals(rectifier_runs):
    fixed, _ = rectifier_runs
    with pytest.raises(ValueError, match="exactly one of t_from= or last="):
        p.steady_state_stats(fixed, "o")
    with pytest.raises(ValueError, match="exactly one of t_from= or last="):
        p.steady_state_stats(fixed, "o", t_from=1e-3, last=1e-3)
    with pytest.raises(ValueError, match="last must be > 0"):
        p.steady_state_stats(fixed, "o", last=0.0)
    with pytest.raises(ValueError, match="own time base"):
        p.steady_state_stats(fixed, np.ones(7), t_from=1e-3)
    with pytest.raises(ValueError, match="kind must be"):
        p.steady_state_stats(fixed, "o", t_from=1e-3, kind="x")
    with pytest.raises(KeyError, match="neither a node nor a branch"):
        p.steady_state_stats(fixed, "nope", t_from=1e-3)


def test_a_name_that_is_both_node_and_branch_is_refused():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "x", "gnd", 1.0)
    b.add_resistor("x", "x", "gnd", 1.0)          # branch named like the node
    res = p.simulate(b, t_end=1e-5, dt=1e-6)
    with pytest.raises(ValueError, match="both a node and a branch"):
        p.steady_state_stats(res, "x", t_from=5e-6)
    # (from 5 us: the t=0 sample is the zero initial state)
    assert p.steady_state_stats(res, "x", t_from=5e-6, kind="i").mean == \
        pytest.approx(1.0)
    assert p.steady_state_stats(res, "x", t_from=5e-6, kind="v").mean == \
        pytest.approx(1.0)
