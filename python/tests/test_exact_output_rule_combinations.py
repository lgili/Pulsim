"""Integration batch 14o: the exact engine's output rule (`output_rtol`, batch
14n) against the features that arrived with the exact engine's snapshot /
junction loop / in-loop recorders (add-exact-propagator-strategy sessions
24-26). Each pair was built on a branch that had not seen the other.

- a run split at a gate edge and resumed WITH the output rule equals the
  uninterrupted run at their common instants (the rule's state is per run:
  the resumed run rebuilds it from its own start);
- with a recorder attached the rule is ignored exactly as with store_every /
  dt (documented on simulate(): the samples are today's);
- the junction temperatures do not depend on the output grid.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p

from .test_exact_junction_loop import _buck
from .test_exact_resume import _ccm, _dcm

OUT = 1e-5


def _sim(b, t_end, fn, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end, switch_fn=fn, engine="dsed", **kw)


def _common(a, b):
    """Row indices of the instants both runs stored; an instant stored twice
    (an event's two limits) compares its last row (the right limit)."""
    ta, tb = np.asarray(a.times), np.asarray(b.times)
    t = np.intersect1d(ta, tb)
    ia = np.searchsorted(ta, t, side="right") - 1
    ib = np.searchsorted(tb, t, side="right") - 1
    return t, ia, ib


def test_a_resumed_run_with_the_output_rule_equals_the_uninterrupted_one():
    for make, t_split, t_end in ((_ccm, 100e-6, 200e-6), (_dcm, 105e-6, 200e-6)):
        b, fn = make()
        u = _sim(b, t_end, fn, output_rtol=OUT)
        b1, fn1 = make()
        a = _sim(b1, t_split, fn1, output_rtol=OUT)
        b2, fn2 = make()
        r = _sim(b2, t_end, fn2, output_rtol=OUT, resume_from=a.final_snapshot)
        assert "output_grid" in r.diagnostics      # the rule ran on the resumed run
        t, iu, ir = _common(u, r)
        assert len(t) >= 10 and t[-1] == t_end
        xu, xr = np.asarray(u.states)[iu], np.asarray(r.states)[ir]
        scale = np.max(np.abs(np.asarray(u.states)), axis=0) + 1e-300
        assert np.max(np.abs(xu - xr) / scale) <= 1e-12
        np.testing.assert_allclose(r.final_snapshot.x, u.final_snapshot.x, rtol=1e-12, atol=1e-12)
        # every event instant of the second half is a sample of both runs
        tu = np.asarray(u.times)
        ev_u = np.unique(tu[1:][np.diff(tu) == 0])
        ev_r = np.unique(np.asarray(r.times)[1:][np.diff(np.asarray(r.times)) == 0])
        np.testing.assert_allclose(ev_u[ev_u > t_split], ev_r[ev_r > t_split], rtol=0, atol=1e-15)


def test_the_output_rule_runs_with_a_recorder_and_is_ignored_with_dt():
    """simulate()'s docstring: ignored with dt / store_every (the exact engine
    refuses store_every by name): the samples are then exactly today's, with
    no output-grid report. A recorder keeps the rule (F-B2-1): the run is
    the one without it (samples and states byte for byte), and its window
    statistics, each segment's closed form, are those of the run without
    the rule to 1e-12."""
    rec = [p.Recorder("s", ["v(vout)"], keep="none", windows=[(10e-6, 90e-6)])]
    b, fn = _ccm()
    bare = _sim(b, 100e-6, fn, output_rtol=OUT)
    b, fn = _ccm()
    ruled = _sim(b, 100e-6, fn, output_rtol=OUT, recorders=rec)
    assert "output_grid" in ruled.diagnostics
    assert np.asarray(ruled.times).tobytes() == np.asarray(bare.times).tobytes()
    assert np.asarray(ruled.states).tobytes() == np.asarray(bare.states).tobytes()
    b, fn = _ccm()
    base = _sim(b, 100e-6, fn, recorders=rec)
    w0 = base.recorded["s"].windows["v(vout)"]["integral"][0]
    w1 = ruled.recorded["s"].windows["v(vout)"]["integral"][0]
    assert w1 == pytest.approx(w0, rel=1e-12, abs=0)
    b, fn = _ccm()
    plain = _sim(b, 100e-6, fn, dt=1e-6)
    b, fn = _ccm()
    ruled = _sim(b, 100e-6, fn, dt=1e-6, output_rtol=OUT)
    assert "output_grid" not in ruled.diagnostics
    assert np.asarray(ruled.times).tobytes() == np.asarray(plain.times).tobytes()
    assert np.asarray(ruled.states).tobytes() == np.asarray(plain.states).tobytes()

def test_junction_temperatures_do_not_depend_on_the_output_grid():
    """The junction loop's heat is the step's exact response to its conduction
    heat plus impulses at events: cutting the steps finer changes T_j only by
    rounding at the instants both grids stored."""
    b, fn, names = _buck()
    a = _sim(b, 202.5e-6, fn)
    b, fn, _ = _buck()
    r = _sim(b, 202.5e-6, fn, output_rtol=OUT)
    assert len(r.times) != len(a.times)            # the grid did change
    t, ia, ir = _common(a, r)
    assert len(t) >= 20
    for k in range(len(names)):
        ta, tr = np.asarray(a.v(f"tj{k}"))[ia], np.asarray(r.v(f"tj{k}"))[ir]
        rise = np.max(np.asarray(a.v(f"tj{k}"))) - 25.0
        assert rise > 0.01
        assert np.max(np.abs(ta - tr)) <= 1e-12 * np.max(np.abs(ta)), k
    assert r.heat_injected == a.heat_injected or abs(r.heat_injected - a.heat_injected) <= 1e-12 * a.heat_injected
