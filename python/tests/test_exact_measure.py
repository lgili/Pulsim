"""The ``exact`` measure's three parts (benchmarks/tools/metrics.exact_parts) on
synthetic traces whose parts are known by construction: smooth error, event
instants, one-sided limits. Pure numpy, no engine and no truth solver."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "tools"))
import metrics  # noqa: E402

H = 0.1                                   # the grid step (and the unit of |δ|/h)
GRID = np.round(np.arange(0.0, 1.0 + H / 2, H), 12)


def step_truth(tau, lo=0.0, hi=1.0):
    """A step from lo to hi at tau: the truth at stored rows (both sides on tau)."""
    def cand(t):
        t = np.asarray(t, dtype=float)
        c = np.full((t.size, 2), np.nan)
        c[:, 0] = np.where(t <= tau, lo, hi)
        c[t == tau, 1] = hi
        return c
    return cand, [(tau, "S+", lo, hi)]


def test_smooth_trace_reads_its_relative_error_and_nothing_else():
    f = np.sin(3 * GRID)
    r = metrics.exact_parts(GRID, f + 0.02, f[:, None], span=2.0, h=H)
    assert r["smooth"] == pytest.approx(0.01, rel=1e-12)
    assert r["instants"] == 0.0 and r["limits"] is None
    assert (r["summary"], r["part"]) == (r["smooth"], "smooth")
    assert r["smooth_rows"] == GRID.size - 1          # every row but the initial instant


@pytest.mark.parametrize("right_row", [True, False])
def test_a_displaced_switch_reads_its_displacement_only(right_row):
    """Exact values on both sides of the engine's own instant, which is
    0.005 h before the truth's: smooth 0, limits 0, instants sqrt(jump² δ / T)."""
    tau, delta = 0.5 + 0.005 * H, -0.005 * H
    cand, tev = step_truth(tau)
    y = np.where(GRID <= 0.5, 0.0, 1.0)              # the engine switched at 0.5
    eng = [(0.5, "S+", 0.0, 1.0 if right_row else None, 0.0, 0.0)]
    r = metrics.exact_parts(GRID, y, cand(GRID), span=1.0, h=H, truth_events=tev, engine_events=eng)
    assert r["smooth"] == 0.0
    assert r["limits"] == 0.0
    assert r["instants"] == pytest.approx(np.sqrt(abs(delta) / 1.0), rel=1e-9)
    assert r["max_dt_h"] == pytest.approx(-0.005, rel=1e-9)
    assert r["part"] == "instants" and not r["missing"] and not r["spurious"]


def test_the_old_measure_mispriced_that_switch():
    """Why the definition changed (vcswitch_resistor pwl+split): the old
    measure saw nothing without the right-limit row and charged jump² × h/2
    (half a step, 100× the true cost in the square) with it."""
    tau = 0.5 + 0.005 * H
    t = np.r_[GRID[:6], np.nextafter(0.5, 1.0), GRID[6:]]          # right limit one ulp after 0.5
    y = np.where(t > 0.5, 1.0, 0.0)
    truth = np.where(t <= tau, 0.0, 1.0)[:, None]                   # beyond the 1e-3 h snap: one branch
    gap = np.zeros((t.size, 1))
    old = metrics.err_exact(t, y, truth, span=1.0, gap=gap)
    assert old == pytest.approx(np.sqrt(H / 2 / 0.9), rel=1e-9)   # the right row's weight h/2 of 0.9
    keep = t != t[6]
    assert metrics.err_exact(t[keep], y[keep], truth[keep], span=1.0, gap=gap[keep]) == 0.0
    cand, tev = step_truth(tau)
    new = metrics.exact_parts(GRID, np.where(GRID > 0.5, 1.0, 0.0), cand(GRID), span=1.0, h=H,
                              truth_events=tev, engine_events=[(0.5, "S+", 0.0, 1.0, 0.0, 0.0)])
    assert new["summary"] == pytest.approx(np.sqrt(0.005 * H / 1.0), rel=1e-9)   # jump² |δ| over T
    assert (old / new["summary"]) ** 2 > 100.0


@pytest.mark.parametrize("right_rows", [True, False])
def test_wrong_right_values_with_exact_timing_read_in_limits_only(right_rows):
    """dsed style (right limits recorded) and trbdf2 style (none: the left
    value is the only one-sided value it records)."""
    tau = 0.5
    cand, tev = step_truth(tau)
    y = np.where(GRID <= tau, 0.0, 1.0)
    left_err = 0.0 if right_rows else 0.04
    eng = [(tau, "S+", 0.0 + left_err, 1.03 if right_rows else None, 0.0, 1.0)]
    r = metrics.exact_parts(GRID, y, cand(GRID), span=1.0, h=H, truth_events=tev, engine_events=eng)
    assert r["smooth"] == 0.0 and r["instants"] == 0.0
    assert r["n_limits"] == (2 if right_rows else 1)
    assert r["limits"] == pytest.approx(0.03 if right_rows else 0.04, rel=1e-12)
    assert r["part"] == "limits"


def test_a_right_limit_row_without_an_event_is_graded_against_the_truth_value():
    # F-SB-1's restart-window rows: a 'jump' recorded where nothing happens
    f = GRID.copy()
    r = metrics.exact_parts(GRID, f, f[:, None], span=1.0, h=H,
                            right_rows=[(0.3, 0.3 + 0.05, 0.3), (0.0, 17.0, 0.0)])
    assert r["limits"] == pytest.approx(0.05, rel=1e-12)    # the 0+ row (t = 0) is not graded
    assert r["n_limits"] == 1 and r["smooth"] == 0.0


def test_a_missing_event_is_named_priced_and_its_effect_is_in_smooth():
    tau = 0.45
    cand, tev = step_truth(tau)
    y = np.zeros(GRID.size)                                    # the engine never switches
    r = metrics.exact_parts(GRID, y, cand(GRID), span=1.0, h=H, truth_events=tev)
    assert r["missing"] == [(tau, "S+")] and r["n_matched"] == 0
    # no row is left out: 0.5 .. 0.9 (0.1 each) and 1.0 (0.05) read 1, of 0.9 s graded
    assert r["smooth"] == pytest.approx(np.sqrt(0.55 / 0.9), rel=1e-12)
    # and the missing event is charged as displaced by the window (no later event of S)
    assert r["instants"] == pytest.approx(np.sqrt(2 * H / 1.0), rel=1e-12)
    assert r["part"] == "smooth"


def test_a_short_pulse_the_engine_misses_between_two_rows_is_charged():
    """Reviewer case: the truth conducts over (50.2, 50.9) h and the engine
    never switches: no row sees it; the two missing events are charged."""
    t = np.arange(0.0, 101.0)
    a, b = 50.2, 50.9
    cand = np.zeros((t.size, 1))
    tev = [(a, "S+", 0.0, 1.0), (b, "S-", 1.0, 0.0)]
    r = metrics.exact_parts(t, np.zeros(t.size), cand, span=1.0, h=1.0, truth_events=tev)
    assert r["smooth"] == 0.0 and len(r["missing"]) == 2
    # S+ until the next S event (0.7), S- for the window (2 h)
    assert r["instants"] == pytest.approx(np.sqrt((0.7 + 2.0) / 100.0), rel=1e-12)


def test_the_row_closing_an_off_grid_event_step_is_graded():
    """Reviewer case: an exact inner event at 50.3 (pwl+split) and a wrong
    stored row at 51: it is not inside any displacement, so it is graded."""
    t = np.arange(0.0, 101.0)
    cand = np.where(t > 50.3, 1.0, 0.0)[:, None]
    y = cand[:, 0].copy()
    y[51] = 100.0
    r = metrics.exact_parts(t, y, cand, span=1.0, h=1.0, truth_events=[(50.3, "S+", 0.0, 1.0)],
                            engine_events=[(50.3, "S+", 0.0, 1.0, 0.0, 1.0)])
    assert r["smooth"] == pytest.approx(99.0 * np.sqrt(1.0 / 99.0), rel=1e-12)
    assert r["part"] == "smooth"


def test_a_truth_that_cannot_be_evaluated_scores_inf():
    """Reviewer case: a NaN truth limit was dropped by max() and read as 0."""
    f = np.zeros(GRID.size)
    r = metrics.exact_parts(GRID, f, f[:, None], span=1.0, h=H,
                            engine_events=[(0.5, "S+", 5.0, None, np.nan, np.nan)])
    assert r["summary"] == np.inf and r["part"] == "non-finite"


def test_spurious_events_are_named_and_initial_selection_is_not_spurious():
    f = np.zeros(GRID.size)
    eng = [(0.0, "S+", 0.0, 0.0, 0.0, 0.0),                    # the engine's initial mode selection
           (0.31, "S+", 0.0, 0.0, 0.0, 0.0), (0.33, "S-", 0.0, 0.0, 0.0, 0.0)]
    r = metrics.exact_parts(GRID, f, f[:, None], span=1.0, h=H, engine_events=eng)
    assert r["spurious"] == [(0.31, "S+"), (0.33, "S-")] and r["n_initial"] == 1
    assert r["smooth_rows"] == GRID.size - 1                   # no row is left out around a spurious event


def test_an_event_at_t0_and_the_0plus_row_are_not_graded():
    """three_phase_diode_rectifier pwl+restart: the right limit at t = 0
    (17.22 V, a quadrature line start) was 99 % of the old score."""
    f = np.zeros(GRID.size)
    eng = [(0.0, "D1+", 0.0, 17.22, 0.0, 0.0)]
    r = metrics.exact_parts(GRID, f, f[:, None], span=20.0, h=H, engine_events=eng,
                            right_rows=[])
    assert r["summary"] == 0.0 and r["n_initial"] == 1 and r["n_limits"] == 0
    t = np.r_[0.0, np.nextafter(0.0, 1.0), GRID[1:]]
    y = np.r_[0.0, 17.22, f[1:]]
    assert metrics.err_exact(t, y, np.zeros((t.size, 1)), span=20.0) > 0.0   # the old measure grades it


def test_a_chattering_engine_matches_once_in_order():
    truth = [(1.0, "D+")]
    eng = [(0.96, "D+"), (0.97, "D-"), (0.98, "D+"), (0.99, "D-"), (1.0, "D+")]
    pairs, missing, spurious = metrics.match_events(truth, eng, 2 * H)
    assert pairs == [(0, 4)] and missing == [] and spurious == [0, 1, 2, 3]
    # by device: an engine flip of another device is not a match
    pairs, missing, spurious = metrics.match_events(truth, [(1.0, "S+")], 2 * H)
    assert pairs == [] and missing == [0] and spurious == [0]
    # beyond the window: missing + spurious
    assert metrics.match_events(truth, [(1.0 + 2.5 * H, "D+")], 2 * H)[1:] == ([0], [0])


def test_no_event_record_makes_instants_not_available():
    tau = 0.5
    cand, tev = step_truth(tau)
    y = np.where(GRID <= tau, 0.0, 1.0)
    r = metrics.exact_parts(GRID, y, cand(GRID), span=1.0, h=H, truth_events=tev,
                            events_available=False)
    assert r["instants"] is None and r["missing"] == [(tau, "S+")]
    assert r["part"] == "smooth" and r["summary"] == 0.0


def test_a_non_finite_recorded_value_scores_inf():
    f = np.zeros(GRID.size)
    r = metrics.exact_parts(GRID, f, f[:, None], span=1.0, h=H, right_rows=[(0.5, np.nan, 0.0)])
    assert r["summary"] == np.inf and r["part"] == "non-finite"


def test_float_instants_and_the_record_end_are_not_graded():
    """An engine instant one ulp from the truth's is not a timing error, and a
    truth event at the run's end (three_phase_diode_rectifier: Dal- at
    t_end) cannot be recorded by any engine."""
    tau = 0.5
    cand, tev = step_truth(tau)
    y = np.where(GRID <= tau, 0.0, 1.0)
    eng = [(np.nextafter(tau, 1.0), "S+", 0.0, 1.0, 0.0, 1.0)]
    r = metrics.exact_parts(GRID, y, cand(GRID), span=1.0, h=H, truth_events=tev + [(1.0, "D+", 0.0, 0.0)],
                            engine_events=eng)
    assert r["instants"] == 0.0 and r["max_dt_h"] == 0.0
    assert r["missing"] == [] and r["summary"] == 0.0


def test_the_row_one_step_after_an_event_is_graded():
    """Only rows with an event at them or strictly inside an adjacent step are
    left out: a wrong first solved value of the new mode is not hidden."""
    tau = 0.5
    cand, tev = step_truth(tau)
    y = np.where(GRID <= tau, 0.0, 1.0)
    y[6] = 1.5                                       # the first stored row after the switch
    r = metrics.exact_parts(GRID, y, cand(GRID), span=1.0, h=H, truth_events=tev,
                            engine_events=[(tau, "S+", 0.0, None, 0.0, 1.0)])
    assert r["smooth_rows"] == GRID.size - 2         # only the row at the event
    assert r["smooth"] == pytest.approx(np.sqrt(0.25 * H / 0.8), rel=1e-12)   # weight 0.1 of the 0.8 s kept


# rule (1), the projected transient (diode_half_wave_rl_load's shape): the truth is
# continuous at the event (1.0 on both sides) and decays to `settle` with time constant
# TAU_F; the engine records `right` (the post-transient value) as its right limit.
TAU_F = 1e-4
WINDOW = 1e-2                                    # the engine's declared parasitic window


def collapse_case(*, reported, settle=0.0, right=0.0, window=WINDOW, drift=0.0):
    tau = 0.5

    def after(s):
        s = np.asarray(s, dtype=float)
        return settle + (1.0 - settle) * np.exp(-s / TAU_F) + drift * s
    t = GRID
    truth = np.where(t <= tau, 1.0, after(np.maximum(t - tau, 0.0)))
    y = np.where(t <= tau, 1.0, settle)
    eng = [(tau, "D-", 1.0, right, 1.0, 1.0, reported)]
    return metrics.exact_parts(
        t, y, truth[:, None], span=1.0, h=H, truth_events=[(tau, "D-", 1.0, 1.0)], engine_events=eng,
        truth_after=lambda te, s: after(s), parasitic=window, limits_tol=1e-3)


def test_a_reported_projection_the_truth_reaches_is_priced_as_a_collapsed_transient():
    r = collapse_case(reported=True)
    assert r["n_projected"] == 1 and r["projected"] == [0.5]
    assert r["limits"] <= 2 * metrics.PROJECTION_SETTLE        # what is left once the truth settled
    # tau_eff: the settling time (20.7 TAU_F to 1e-9), on a 2x grid
    lo, hi = TAU_F * np.log(1e9), 2 * TAU_F * np.log(1e9)
    assert np.sqrt(lo / 1.0) <= r["instants"] <= np.sqrt(hi / 1.0)
    assert r["part"] == "instants" and r["smooth"] < 1e-12


def test_the_part_of_a_collapsed_right_value_that_is_wrong_stays_in_limits():
    r = collapse_case(reported=True, right=2e-4)               # the settled truth is 0.0
    assert r["n_projected"] == 1
    assert r["limits"] == pytest.approx(2e-4, rel=1e-3)


def test_without_a_projection_record_the_right_value_is_a_wrong_limit():
    r = collapse_case(reported=False)
    assert r["n_projected"] == 0
    assert r["limits"] == pytest.approx(1.0, rel=1e-12) and r["part"] == "limits"


@pytest.mark.parametrize("case", [dict(settle=0.5),                # settles, but at 0.5, not 0.0
                                  dict(window=10 * TAU_F),         # reaches 0.0 only after the window
                                  dict(drift=1.0)])                # never settles inside the window
def test_a_truth_that_does_not_settle_at_the_recorded_value_leaves_the_limit_wrong(case):
    r = collapse_case(reported=True, **case)
    assert r["n_projected"] == 0 and r["limits"] == pytest.approx(1.0, rel=1e-12)


def test_an_impulse_the_truth_makes_too_is_not_a_collapse():
    """The fixed engine's impulse records: the truth's own right limit is the
    recorded value (it jumps too); nothing is collapsed, nothing moves."""
    cand, tev = step_truth(0.5)
    y = np.where(GRID <= 0.5, 0.0, 1.0)
    r = metrics.exact_parts(GRID, y, cand(GRID), span=1.0, h=H, truth_events=tev,
                            engine_events=[(0.5, "S+", 0.0, 1.0, 0.0, 1.0, True)],
                            truth_after=lambda te, s: np.ones(np.size(s)), parasitic=WINDOW)
    assert r["n_projected"] == 0 and r["summary"] == 0.0
