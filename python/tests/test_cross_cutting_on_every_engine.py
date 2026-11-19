"""add-kernel-simulation-object 1.4: the cross-cutting rules of simulate(),
asserted per engine against today's behaviour (recorded 2026-11-18), so a
plan / route refactor that moves them cannot change one engine silently.

Builder ICs, resume_from validation, preflight ties, the undriven-switch
behaviour (still per engine: DECISIONS 8 / task 11.3) and voltage sanity.
"""

import warnings

import numpy as np
import pytest

import pulsim as p

ENGINES = {"pwl": dict(engine="pwl", dt=1e-6), "trbdf2": dict(engine="trbdf2"),
           "exact": dict(engine="exact")}
USED = {"pwl": "pwl", "trbdf2": "trbdf2", "exact": "dsed"}


def _rc(c0=None):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 1.0)
    b.add_resistor("R", "in", "out", 1e3)
    b.add_capacitor("C", "out", "gnd", 1e-6, c0)
    return b


def _run(b, engine, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = p.simulate(b, 1e-5, **ENGINES[engine], **kw)
    return r, [str(x.message) for x in w]


@pytest.mark.parametrize("engine", ENGINES)
def test_builder_initial_conditions_seed_the_run(engine):
    r, _ = _run(_rc(0.5), engine)
    assert r.engine_used == USED[engine]
    assert float(np.asarray(r.v("out"))[0]) == pytest.approx(0.5, abs=1e-9)


@pytest.mark.parametrize("engine", ENGINES)
def test_an_invalid_snapshot_is_refused_before_any_engine(engine):
    with pytest.raises(ValueError, match="snapshot is not valid"):
        p.simulate(_rc(), 1e-5, resume_from=object(), **ENGINES[engine])


@pytest.mark.parametrize("engine", ENGINES)
def test_preflight_ties_a_floating_island_on_the_runs_copy(engine):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "b", 1.0)
    b.add_resistor("R", "a", "b", 1e3)
    n_before = b.graph.num_branches
    r, w = _run(b, engine)
    assert len(r.plan.ties) == 1
    assert any("preflight" in m for m in w)
    assert b.graph.num_branches == n_before  # the caller's builder: untouched


def _undriven():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 1.0)
    b.add_switch("S", "in", "out", 1e3, 1e-9)
    b.add_resistor("R", "out", "gnd", 1.0)
    return b


@pytest.mark.parametrize("engine", ["pwl", "exact"])
def test_an_undriven_switch_runs_closed_when_asked(engine):
    # legacy subject: the pre-release pwl / exact behaviour, restored by undriven='closed'
    r, _ = _run(_undriven(), engine, undriven="closed")
    assert r.diagnostics["undriven"] == {"policy": "closed", "switches": ["S"]}
    assert float(np.asarray(r.v("out"))[-1]) == pytest.approx(1.0, rel=2e-3)


@pytest.mark.parametrize("engine", ENGINES)
def test_an_undriven_switch_is_refused_on_every_engine(engine):
    # since the release (DECISIONS 26.1) every engine refuses it by name
    with pytest.raises(p.UnsupportedError, match=r"undriven='refuse'.*have no driver: S\."):
        p.simulate(_undriven(), 1e-5, **ENGINES[engine])


@pytest.mark.parametrize("engine", ENGINES)
def test_voltage_sanity_warns_on_every_engine(engine):
    _, w = _run(_rc(), engine, voltage_sanity_factor=1e-9)
    assert any("node in reached" in m for m in w)
