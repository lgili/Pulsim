"""F-Z-3: the restart rule after a split SLIVER (an event piece much shorter
than dt). A hard-switched capacitor (C = 1 nF, R_on = 1 mOhm: a 1 ps mode,
dt = 1 ns) gated a fraction of dt before a grid point: the step is cut at the
edge and the event's restart probe spans only gamma * (the piece), so the mode
was neither damped nor projected and rang +-2.25 A on under trapezoid (max
deviation 1.022 A at 1e-5 dt, 0.362 A at 1e-3 dt). The next grid step's restart
now probes it with the event before its start. Options explicit: split_snap
is reachable only through run_transient."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import pulsim as p

TB, DT, GON, GOFF, V, R = 1e-5, 1e-9, 1e3, 1e-9, 100.0, 10.0
I_ON = V / (R + 1 / GON)


def _cap_cell(off, split=True):
    """Edges at (k + 0.5 - off) TB and (k + 1 - off) TB; off < 0: after."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", V)
    b.add_resistor("R", "in", "sw", R)
    b.add_switch("S", "sw", "gnd", GON, GOFF)
    b.add_capacitor("C", "sw", "gnd", 1e-9)
    n, k = b.graph.num_switches, b.switch_index_of("S")

    def sw(t):
        m = p.SwitchStateMask(n)
        m.set(k, bool((t / TB) - math.floor(t / TB + off) >= 0.5 - off))
        return m

    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build_lazy(DT)
    o = p.SimulationOptions(t_start=0.0, t_end=20 * TB + 1e-12, dt=DT)
    o.discontinuity_restart = True
    o.enable_substep_state_correction = split
    o.split_snap = 1e-6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.run_transient(cache, b.graph, b.pool, o, sw)
    t = np.asarray(res.times)
    i_sw = np.asarray(res.states)[:, b.node_id_of("sw")] * GON
    ph = (t / TB) % 1.0
    on = (ph > 0.52) & (ph < 0.98) & (t > 10 * TB)   # well inside the ON phase
    return np.abs(i_sw[on] - I_ON).max(), res


# The switch current is G_on * v_sw with v_sw ~ 10 mV computed on the 100 V
# source's scale: its round-off is eps * V * G_on = 2.2e-11 A; the bound
# allows 100x that (the rule alone, split off, gives 1.8e-15 A).
BOUND = 100 * np.finfo(float).eps * V * GON


@pytest.mark.parametrize("side", [1, -1], ids=["before", "after"])
@pytest.mark.parametrize("frac", [1e-7, 1e-5, 1e-3, 0.1, 0.5, 0.9])
def test_no_ringing_after_a_split_sliver(frac, side):
    dev, res = _cap_cell(side * frac * DT / TB)
    assert dev < BOUND, f"max deviation {dev:.3g} A after the edge"
    if frac >= 1e-5:   # 1e-7 dt is inside split_snap (1e-6): taken at the grid point
        assert res.n_split_steps > 0


def test_rule_alone_is_the_reference_class():
    dev, res = _cap_cell(1e-5 * DT / TB, split=False)
    assert dev < BOUND and res.n_split_steps == 0


def _two_cells(split=True, off=1e-5 * DT / TB, t_end=20 * TB + 1e-12):
    """Two hard-switched capacitor cells on one source: S1's edges at
    (k + 0.5) TB - off and k TB - off, S2's one step later, so the two
    slivers sit in consecutive grid steps."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", V)
    for j in (1, 2):
        b.add_resistor(f"R{j}", "in", f"sw{j}", R)
        b.add_switch(f"S{j}", f"sw{j}", "gnd", GON, GOFF)
        b.add_capacitor(f"C{j}", f"sw{j}", "gnd", 1e-9)
    n = b.graph.num_switches
    ks = [b.switch_index_of(f"S{j}") for j in (1, 2)]

    def sw(t):
        m = p.SwitchStateMask(n)
        for j, k in enumerate(ks):
            u = (t - j * DT) / TB
            m.set(k, bool(u - math.floor(u + off) >= 0.5 - off))
        return m

    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build_lazy(DT)
    o = p.SimulationOptions(t_start=0.0, t_end=t_end, dt=DT)
    o.discontinuity_restart = True
    o.enable_substep_state_correction = split
    o.split_snap = 1e-6
    return b, cache, o, sw


def _run(cache, b, o, sw, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.run_transient(cache, b.graph, b.pool, o, sw, **kw)


def _on_currents(b, res):
    t = np.asarray(res.times)
    x = np.asarray(res.states)
    out = []
    for j in (1, 2):
        ph = ((t - (j - 1) * DT) / TB) % 1.0
        on = (ph > 0.52) & (ph < 0.98) & (t > 10 * TB)
        out.append((t[on], x[on, b.node_id_of(f"sw{j}")] * GON))
    return out


def test_two_slivers_in_consecutive_steps_both_damped():
    b, cache, o, sw = _two_cells()
    res = _run(cache, b, o, sw)
    assert res.n_split_steps > 0
    b0, cache0, o0, sw0 = _two_cells(split=False)
    ref = _run(cache0, b0, o0, sw0)
    for (t, i), (t0, i0) in zip(_on_currents(b, res), _on_currents(b0, ref)):
        assert np.abs(i - I_ON).max() < BOUND
        # the grid samples of the rule-alone run (edges taken at the grid point)
        common, a, c = np.intersect1d(t, t0, return_indices=True)
        assert common.size > 1000
        assert np.abs(i[a] - i0[c]).max() < BOUND


def test_resume_across_the_sliver_anchor():
    """A run stopped on the step that ends with a sliver resumes it bit for
    bit, in process and from a pickled copy (the anchor is pickled: before,
    the copy dropped it and its first resumed sample was ~470 A off). A
    pickle without the field (an older build's) loads it as False."""
    import pickle

    t_end = 4 * TB + 1e-12
    b, cache, o, sw = _two_cells(t_end=t_end)
    full = _run(cache, b, o, sw)
    # the first S1 turn-on edge is 1e-14 s before the grid point 0.5 TB
    b1, cache1, o1, sw1 = _two_cells(t_end=0.5 * TB)
    first = _run(cache1, b1, o1, sw1)
    snap = first.final_snapshot
    assert snap.restart_anchor_at_end
    t0 = snap.t
    o2 = p.SimulationOptions(t_start=t0, t_end=t_end, dt=DT)
    o2.discontinuity_restart = True
    o2.enable_substep_state_correction = True
    o2.split_snap = 1e-6
    tf, xf = np.asarray(full.times), np.asarray(full.states)
    k = int(np.searchsorted(tf, t0))
    assert tf[k] == t0
    rest = _run(cache1, b1, o2, sw1, resume_from=snap)
    tr, xr = np.asarray(rest.times), np.asarray(rest.states)
    assert np.array_equal(tr, tf[k:k + tr.size]) and tr[-1] == tf[-1]
    assert np.array_equal(xr, xf[k:])
    cold = pickle.loads(pickle.dumps(snap))
    assert cold.restart_anchor_at_end
    assert cold.restart_steps_left == snap.restart_steps_left
    rest_p = _run(cache1, b1, o2, sw1, resume_from=cold)
    assert np.array_equal(np.asarray(rest_p.times), tr)
    assert np.array_equal(np.asarray(rest_p.states), xr)
    # an older build's pickle (no field 20): the neutral value, no error
    state = snap.__getstate__()
    assert len(state) == 21
    legacy = p._pulsim.SolverSnapshot.__new__(p._pulsim.SolverSnapshot)
    legacy.__setstate__(state[:20])
    assert not legacy.restart_anchor_at_end
    assert legacy.restart_steps_left == snap.restart_steps_left


def test_snapshot_pickle_carries_every_field():
    """Every SolverSnapshot field survives pickling: a field the pickle drops
    is a silent wrong answer on a cross-process resume (the anchor above).
    The invalid snapshot's reason too (a resume refuses by it)."""
    import pickle

    s = p._pulsim.SolverSnapshot()
    s.invalid_reason = "kept none: the reason"
    c = pickle.loads(pickle.dumps(s))
    assert c.invalid_reason == "kept none: the reason" and not c.valid
    assert pickle.loads(pickle.dumps(p._pulsim.SolverSnapshot())).invalid_reason == ""


def test_period_map_copies_the_anchor():
    from pulsim.steady_state import _snapshot_like

    b, cache, o, sw = _two_cells(t_end=0.5 * TB)
    snap = _run(cache, b, o, sw).final_snapshot
    assert snap.restart_anchor_at_end
    assert _snapshot_like(snap, list(snap.history)).restart_anchor_at_end
