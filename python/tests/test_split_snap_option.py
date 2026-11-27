"""simulate(split_snap=...): the fixed engine's event-split snap band (a
fraction of dt) is a public keyword. Its default moved from 0.01 (2.1.0) to
1e-6; None keeps the default, 0.01 restores 2.1.0's behaviour.

The cell: a hard-switched capacitor (test_split_sliver_restart) gated 1e-3 dt
before grid points. With 1e-6 the split cuts a sliver there (split steps);
with 0.01 the edge is within the snap and is taken at the grid point (none)."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import pulsim as p

TB, DT = 1e-7, 1e-9
T_END = 5 * TB + 1e-12
OFF = 1e-3 * DT / TB           # every edge 1e-3 dt before its grid point


def _cell():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 100.0)
    b.add_resistor("R", "in", "sw", 10.0)
    b.add_switch("S", "sw", "gnd", 1e3, 1e-9)
    b.add_capacitor("C", "sw", "gnd", 1e-9)
    n, k = b.graph.num_switches, b.switch_index_of("S")

    def sw(t):
        m = p.SwitchStateMask(n)
        m.set(k, bool((t / TB) - math.floor(t / TB + OFF) >= 0.5 - OFF))
        return m
    return b, sw


def _sim(**kw):
    b, sw = _cell()
    kw.setdefault("discontinuity_restart", True)
    kw.setdefault("enable_substep_state_correction", True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, T_END, dt=DT, switch_fn=sw, **kw)


def _raw(snap=None):
    b, sw = _cell()
    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build_lazy(DT)
    o = p.SimulationOptions(t_start=0.0, t_end=T_END, dt=DT)
    o.discontinuity_restart = True
    o.enable_substep_state_correction = True
    if snap is not None:
        o.split_snap = snap
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.run_transient(cache, b.graph, b.pool, o, sw)


def _same(a, b):
    np.testing.assert_array_equal(np.asarray(a.times), np.asarray(b.times))
    np.testing.assert_array_equal(np.asarray(a.states), np.asarray(b.states))
    assert a.n_split_steps == b.n_split_steps


def test_not_passing_split_snap_is_the_default_bit_for_bit():
    base = _sim()
    _same(base, _sim(split_snap=None))
    _same(base, _sim(split_snap=1e-6))        # the engine default, spelled out
    assert dict(base.counters) == dict(_sim(split_snap=None).counters)
    _same(base, _raw())                       # run_transient, untouched options


def test_split_snap_0_01_restores_the_2_1_0_snap():
    new, old = _sim(), _sim(split_snap=0.01)
    assert new.n_split_steps > 0              # 1e-6: a sliver is cut at each edge
    assert old.n_split_steps == 0             # 0.01: the edge snaps to the grid
    assert not np.array_equal(np.asarray(new.states), np.asarray(old.states))


@pytest.mark.parametrize("snap", [0.01, 1e-4])
def test_simulate_equals_run_transient_bit_for_bit(snap):
    via_kw = _sim(split_snap=snap)
    _same(via_kw, _raw(snap))
    _same(via_kw, _sim(solver=p.SolverOptions(split_snap=snap)))


@pytest.mark.parametrize("bad", [0.0, -1e-3, 1e-13, 0.3, 0.5, math.nan, math.inf])
def test_out_of_range_is_refused_by_name(bad):
    with pytest.raises(p.errors.OptionsError, match="split_snap") as e:
        _sim(split_snap=bad)
    assert e.value.option == "split_snap"


@pytest.mark.parametrize("kw", [
    {"enable_substep_state_correction": False},
    {"discontinuity_restart": False, "enable_substep_state_correction": None},
], ids=["split_off", "rule_off"])
def test_split_off_is_refused_by_name(kw):
    with pytest.raises(p.errors.OptionsError, match="split_snap") as e:
        _sim(split_snap=0.01, **kw)
    assert e.value.option == "split_snap"
    assert "splitting is off" in str(e.value)


@pytest.mark.parametrize("engine", ["trbdf2", "exact"])
def test_other_engines_refuse_it_as_they_refuse_the_split(engine):
    # The sibling convention (enable_substep_state_correction): the option x
    # engine table refuses a fixed-step-only keyword with UnsupportedError.
    b, sw = _cell()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(p.UnsupportedError) as e:
            p.simulate(b, T_END, engine=engine, switch_fn=sw, split_snap=0.01)
    assert e.value.feature == "split_snap"
