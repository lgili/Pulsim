"""The independent periodicity check of the evaluator's exact route
(review item c1): the kernel's verification period is the shooting map
itself, so one plain ``simulate(engine='dsed')`` period from ``x*`` must
return to ``x*``; once per (topology, event pattern), a mismatch fails
the design by name."""

import numpy as np

import pulsim as p
import pulsim.design as design
from pulsim.design import Evaluator, SteadySpec

from .test_design_evaluation import CCM, T, buck

DESIGNS = [dict(CCM, L=L) for L in (60e-6, 100e-6, 220e-6)]


def _ev():
    return Evaluator(buck, period=T, loads=("RL",), output="vout",
                     steady=SteadySpec(method="pss"))


def _spy(monkeypatch, perturb=0.0):
    calls = []
    real = p.simulate

    def sim(*a, **kw):
        r = real(*a, **kw)
        calls.append(kw.get("t_end"))
        if perturb:
            r.states_reduced = np.asarray(r.states_reduced) * (1 + perturb)
        return r

    monkeypatch.setattr(p, "simulate", sim)
    return calls


def test_one_check_per_event_pattern_and_records_unchanged(monkeypatch):
    calls = _spy(monkeypatch)
    ev = _ev()
    recs = [ev.evaluate(d) for d in DESIGNS]
    assert [r.status for r in recs] == ["ok"] * 3
    assert len(calls) == 1 and calls[0] == T        # one plain period
    ((key, dev),) = ev.periodicity_checks.items()
    assert dev <= 1e-12                             # rounding, not 1e-8
    # the check is not in the records: the same records without it
    monkeypatch.setattr(design.Evaluator, "_check_periodic",
                        lambda *a: None)
    for d, r in zip(DESIGNS, recs):
        r2 = _ev().evaluate(d)
        assert (r2.status, r2.kpis, r2.losses, r2.counters) == \
            (r.status, r.kpis, r.losses, r.counters)


def test_a_map_the_engine_does_not_reproduce_fails_by_name(monkeypatch):
    calls = _spy(monkeypatch, perturb=1e-6)
    ev = _ev()
    recs = [ev.evaluate(d) for d in DESIGNS]
    assert [r.status for r in recs] == ["failed"] * 3
    assert all(r.reason.startswith("periodicity:") for r in recs)
    assert len(calls) == 1                  # the failure is cached
    (dev,) = ev.periodicity_checks.values()
    assert 1e-8 < dev < 1e-5


def test_the_check_restarts_in_a_pickled_evaluator():
    import pickle
    ev = _ev()
    ev.evaluate(DESIGNS[0])
    assert len(ev.periodicity_checks) == 1
    assert pickle.loads(pickle.dumps(ev)).periodicity_checks == {}
