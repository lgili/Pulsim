"""Recorder state across final_snapshot / resume_from (task 0b.7).

A run split at a mid-window instant and resumed gives the uninterrupted
run's record: kept rows, event rows, declared windows and periods (to
1e-9 on pwl — bit for bit on macOS, but CI compares no bits — and to rtol
on trbdf2, whose resumed controller takes its own steps). A
snapshot without recorder state starts the recorders at the resume instant
and says so; a window that began before it is never reported complete.
"""

from __future__ import annotations

import pickle
import sys
import warnings

import numpy as np
import pytest

import pulsim as p

from .test_recorder_device_readings import _buck

_RUNS = {"pwl": dict(engine="pwl", dt=1e-6),
         "trbdf2": dict(engine="trbdf2", rtol=1e-7, atol=1e-10)}
T_SPLIT, T_END = 0.43e-3, 1e-3


def _recs():
    return [p.Recorder("rows", ["v(out)", "i(C)", "i(D1)"], keep="decimate", every=3),
            p.Recorder("stats", ["v(out)", "p(S1)", "i(L) * 2"], keep="none",
                       windows=[(0.2e-3 + 1.1e-9, 0.8e-3 + 2.3e-9)], period=0.13e-3)]


def _sim(t_end, **kw):
    b, fn, *_ = _buck()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end, switch_fn=fn, **kw)


def _compare(a, b, paired):
    # pwl: the resumed steps are the uninterrupted ones, rows paired, to
    # 1e-9 (bit for bit on macOS; no bit-equality in CI tests). trbdf2: the
    # resumed controller takes its own steps (two rows fewer here), so the
    # statistics agree to the run's accuracy and the rows are not paired
    close = dict(rtol=1e-9, atol=0.0, equal_nan=True)
    worst = 0.0
    for name in ("rows", "stats"):
        ra, rb = a.recorded[name], b.recorded[name]
        if paired:
            np.testing.assert_allclose(ra.times, rb.times, rtol=1e-12, atol=0.0)
            np.testing.assert_allclose(ra.event_t, rb.event_t, rtol=1e-12, atol=0.0)
        for q in ra.windows:
            if paired:
                sc = max(float(np.max(np.abs(ra[q]), initial=0.0)), 1e-30)
                np.testing.assert_allclose(ra[q], rb[q], rtol=1e-9, atol=1e-9 * sc,
                                           err_msg=f"{name} {q}")
            for group in ("windows", "periods"):
                ga, gb = getattr(ra, group)[q], getattr(rb, group)[q]
                for f in ga:
                    x, y = np.asarray(ga[f]), np.asarray(gb[f])
                    assert x.shape == y.shape, (name, q, group, f)
                    if paired:
                        np.testing.assert_allclose(x, y, **close, err_msg=f"{name} {q} {group} {f}")
                    elif x.size:
                        sc = max(np.max(np.abs(x)), 1e-30)
                        worst = max(worst, float(np.max(np.abs(x - y))) / sc)
    return worst


@pytest.mark.parametrize("eng", sorted(_RUNS))
def test_resumed_record_equals_the_uninterrupted_one(eng):
    whole = _sim(T_END, recorders=_recs(), **_RUNS[eng])
    first = _sim(T_SPLIT, recorders=_recs(), **_RUNS[eng])
    snap = first.final_snapshot
    assert len(snap.recorder_state) > 0
    snap = pickle.loads(pickle.dumps(snap))          # through the pickle (field 18)
    assert len(snap.__getstate__()) == 19
    second = _sim(T_END, recorders=_recs(), resume_from=snap, **_RUNS[eng])
    assert second.recorder_feed["mode"] == "in-loop"
    assert second.recorder_feed["resume"].startswith("continued"), second.recorder_feed
    assert np.isfinite(second.recorded["stats"].windows["v(out)"]["avg"][0])
    # in the first part the window is open: reported NaN, not a partial value
    assert np.isnan(first.recorded["stats"].windows["v(out)"]["avg"][0])
    worst = _compare(whole, second, eng == "pwl")
    print(f"{eng}: worst relative difference of the statistics {worst:.2e}",
          file=sys.stderr)
    assert worst <= 1e-4, worst


@pytest.mark.parametrize("eng", sorted(_RUNS))
def test_resume_without_recorder_state_says_so(eng):
    snap = _sim(T_SPLIT, **_RUNS[eng]).final_snapshot   # no recorders: no state
    assert len(snap.recorder_state) == 0
    rec = p.Recorder("s", ["v(out)"], keep="none", period=0.13e-3)
    res = _sim(T_END, recorders=[rec], resume_from=snap, **_RUNS[eng])
    assert "started at the resume instant" in res.recorder_feed["resume"]
    t0 = res.recorded["s"].periods["v(out)"]["t0"]
    k = int(T_SPLIT // 0.13e-3)                      # the period holding the seam
    # only complete periods are reported: none before the seam's
    assert t0.size > 0 and np.min(t0) > T_SPLIT and np.isclose(np.min(t0), (k + 1) * 0.13e-3)
    # a declared window that began before the resume is refused, not cut
    with pytest.raises(ValueError, match="not covered"):
        _sim(T_END, resume_from=snap, **_RUNS[eng], recorders=[
            p.Recorder("w", ["v(out)"], windows=[(0.2e-3, 0.8e-3)])])


def test_resume_with_other_recorders_is_refused_by_name():
    first = _sim(T_SPLIT, recorders=_recs(), **_RUNS["pwl"])
    with pytest.raises(ValueError, match="written by other recorders"):
        _sim(T_END, resume_from=first.final_snapshot, **_RUNS["pwl"],
             recorders=[p.Recorder("rows", ["v(out)"])])


def test_an_18_field_snapshot_still_loads():
    snap = _sim(T_SPLIT, recorders=_recs(), **_RUNS["pwl"]).final_snapshot
    st = snap.__getstate__()
    old = type(snap).__new__(type(snap))
    old.__setstate__(st[:18])
    assert len(old.recorder_state) == 0
    assert list(old.dev_state) == list(snap.dev_state)


def test_a_statistics_only_recorder_with_more_samples_than_state_resumes():
    # its saved state is a few dozen values while it has counted ~430 samples
    rec = [p.Recorder("n", ["v(out)"], keep="none", period=0.13e-3)]
    first = _sim(T_SPLIT, recorders=rec, **_RUNS["pwl"])
    assert len(first.final_snapshot.recorder_state) < 430
    res = _sim(T_END, recorders=[p.Recorder("n", ["v(out)"], keep="none", period=0.13e-3)],
               resume_from=first.final_snapshot, **_RUNS["pwl"])
    assert res.recorder_feed["resume"].startswith("continued")
