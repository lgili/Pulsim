"""The flat device state in a snapshot (add-unified-device-interface step 3).

A snapshot carries every device state as one vector (`dev_state`) in
named blocks (`dev_layout`), resumes on either transient engine, refuses
another circuit's layout by block name, and still reads Pulsim 2.0
pickles.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import pulsim as p
from pulsim._pulsim import SolverSnapshot


def _all_kinds():
    """A Coss, a Lauritzen diode, a saturable inductor, an IGBT with a
    tail, PWL diodes and linear L / C in one circuit."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_saturable_inductor("L", "sw", "out", 100e-6, 6.0, 1e-6)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_nonlinear_capacitor("Cj", "vin", "sw", 1e-9, 1.0)
    b.add_lauritzen_diode("DL", "out", "x")
    b.add_resistor("RL", "x", "gnd", 2.0)
    b.add_pulse_voltage_source("Vg", "g", "gnd", 0.0, 15.0, 0.0, 5e-5, 0.0,
                               1e-8, 1e-8)
    b.add_voltage_source("Vdc", "dc", "gnd", 100.0)
    b.add_inductor("Lload", "dc", "c", 500e-6, i0=10.0)
    b.add_diode("Dfw", "c", "dc", 1e3, 1e-9, 0.7)
    b.add_igbt_level1("Q1", "c", "e", "g", 1.5, 0.05, 5.0, tau_tail=1e-6,
                      k_tail=0.30)
    b.add_resistor("Rsense", "e", "gnd", 1e-4)
    b.add_capacitor("Cs", "c", "gnd", 2e-9)
    n, k, T = b.graph.num_switches, b.switch_index_of("S"), 1e-5

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, bool((t / T) - np.floor(t / T + 1e-9) < 0.5 - 1e-9))
        return m
    return b, fn


def _pwl(t_end, **kw):
    b, fn = _all_kinds()
    return p.simulate(b, t_end=t_end, dt=2e-8, engine="pwl", switch_fn=fn,
                      **kw)


def _trbdf2(t_end, **kw):
    b, fn = _all_kinds()
    return p.simulate(b, t_end=t_end, engine="trbdf2", rtol=1e-6,
                      switch_fn=fn, **kw)


def test_the_snapshot_holds_every_block_in_one_vector():
    snap = _pwl(2e-5).final_snapshot
    names = list(snap.dev_layout_names)
    counts = dict(zip(names, snap.dev_layout))
    assert counts["saturable_inductor"] == 1
    assert counts["nonlinear_capacitor"] == 1
    assert counts["lauritzen_diode"] == 1
    assert counts["igbt_tail"] == 1
    assert counts["pmsm_mna"] == 0
    assert counts["linear_companion"] >= 3          # C, Lload, Cs
    # the 2.0 named fields are views of the blocks, in layout order
    joined = (list(snap.history) + list(snap.saturable_history)
              + list(snap.coss_history) + list(snap.lauritzen_history)
              + list(snap.igbt_tail_history) + list(snap.pmsm_history))
    assert joined == list(snap.dev_state)      # (order: next test)


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_each_block_holds_its_device_s_state_at_the_end(engine):
    """Checked against the run's own waveforms, not against the snapshot:
    the saturable block is (λ, i, v, ...) of L, the Coss block (v, q, i)
    of Cj, the Lauritzen block (q_M, f) of DL, and the IGBT carries tail
    charge after its turn-off."""
    res = {"pwl": _pwl, "trbdf2": _trbdf2}[engine](4.27e-5)
    snap = res.final_snapshot
    sat = list(snap.saturable_history)
    assert sat[1] == pytest.approx(np.asarray(res.i("L"))[-1], rel=1e-12)
    v_l = np.asarray(res.v("sw"))[-1] - np.asarray(res.v("out"))[-1]
    assert sat[2] == pytest.approx(v_l, rel=1e-9)
    assert sat[0] > 0.0                                # flux, i > 0
    coss = list(snap.coss_history)
    v_cj = np.asarray(res.v("vin"))[-1] - np.asarray(res.v("sw"))[-1]
    assert coss[0] == pytest.approx(v_cj, rel=1e-9, abs=1e-12)
    assert list(snap.igbt_tail_history)[0] > 0.0      # stored tail charge
    # Lauritzen q_M ~ tau * i_D near steady conduction (tau = 1e-7 s):
    # tells its block from the IGBT tail's, which is 2 wide as well
    q_m = list(snap.lauritzen_history)[0]
    assert q_m == pytest.approx(1e-7 * np.asarray(res.i("RL"))[-1], rel=2e-2)
    assert np.array_equal(np.asarray(snap.x), np.asarray(res.states)[-1])


@pytest.mark.parametrize("path", ["legacy", "interface"])
@pytest.mark.parametrize("t_split", [4.2e-5, 5.0e-5, 5.5e-5])
def test_split_fixed_step_run_is_the_continuous_one_bit_for_bit(t_split, path):
    """The snapshot carries the grid (origin, index, dt), so the resumed
    run continues the continuous run's time grid and meets every gate
    edge at the same instant. Before, the grid restarted at t_split, the
    sample times moved by ~1e-20 s and the states drifted apart after the
    next edge (8e-12 of scale). Split between edges and exactly on one
    (5e-5 on, 5.5e-5 off)."""
    # On 'interface' too: the Lauritzen and IGBT-tail local charges must
    # survive the snapshot against the UNINTERRUPTED run, not only a resume.
    full = _pwl(1e-4, device_path=path)
    first = _pwl(t_split, device_path=path)
    snap = pickle.loads(pickle.dumps(first.final_snapshot))
    assert snap.grid_dt == 2e-8 and snap.grid_origin == 0.0
    second = _pwl(1e-4, resume_from=snap, device_path=path)
    t_full = np.asarray(full.times)
    i0 = len(np.asarray(first.times)) - 1
    t2 = np.asarray(second.times)
    n = min(len(t2), len(t_full) - i0)
    assert n > 0.99 * len(t2)
    assert np.array_equal(t2[:n], t_full[i0:i0 + n])
    assert np.array_equal(np.asarray(second.states)[:n],
                          np.asarray(full.states)[i0:i0 + n])
    assert np.array_equal(np.asarray(second.final_snapshot.dev_state),
                          np.asarray(full.final_snapshot.dev_state))


def test_a_snapshot_without_a_grid_restarts_the_grid_at_its_t():
    """A 2.0 / version-3 pickle, a hand-built or a variable-step snapshot
    carries no grid: the resumed grid starts at t_start, as before (within
    1e-9 of scale of the continuous run, not bit for bit)."""
    full = _pwl(1e-4)
    snap = _pwl(4.2e-5).final_snapshot
    snap.grid_origin = float("nan")
    second = _pwl(1e-4, resume_from=snap)
    i0 = int(np.searchsorted(np.asarray(full.times), snap.t))
    x2 = np.asarray(second.states)
    x_full = np.asarray(full.states)[i0:i0 + len(x2)]
    scale = np.abs(x_full).max(axis=0) + 1e-12
    assert (np.abs(x_full - x2[:len(x_full)]) / scale).max() < 1e-9


@pytest.mark.parametrize("first_engine,second_engine",
                         [("pwl", "trbdf2"), ("trbdf2", "pwl")])
def test_a_snapshot_resumes_on_the_other_engine(first_engine, second_engine):
    """Split off any gate edge (a TR-BDF2 snapshot ON an edge is refused
    by the fixed engine: next test)."""
    run = {"pwl": _pwl, "trbdf2": _trbdf2}
    t_split = 4.77e-5             # S is off: the Coss holds ~24 V
    first = run[first_engine](t_split)
    snap = first.final_snapshot
    second = run[second_engine](1e-4, resume_from=snap)
    full = run[second_engine](1e-4)
    x2 = np.asarray(second.states)
    # starts from the snapshot's x, device state and mode bits
    assert np.array_equal(x2[0], np.asarray(snap.x))
    # each stateful kind continues from its own state across the seam,
    # checked up to the next gate edge
    for name, err in _seam_errors(second_engine, second, full).items():
        assert err <= 4e-3, name   # intact <= 7.1e-4; a dropped state > 2e-2


_RUN = {"pwl": _pwl, "trbdf2": _trbdf2}
_T_SPLIT = 4.77e-5              # S is off: the Coss holds ~24 V
_PROBES = {
    "i(L)": lambda r: np.asarray(r.i("L")),
    "v(Cj)": lambda r: np.asarray(r.v("vin")) - np.asarray(r.v("sw")),
    "i(Q1)": lambda r: np.asarray(r.i("Rsense")),
    "v(out)": lambda r: np.asarray(r.v("out")),
    "i(D1)": lambda r: np.asarray(r.i("D1")),
}


def _seam_errors(second_engine, second, full):
    """Relative error of each probe of the resumed run up to the next
    gate edge. i(D1) is compared with the TR-BDF2 continuous run: the
    fixed engine's own continuous run rings in i(D1) after the Coss
    turn-off (trapezoid, 8.39 / 6.86 A), the resumed one does not."""
    t2 = np.asarray(second.times)
    w = (t2 > _T_SPLIT) & (t2 < 4.99e-5)
    out = {}
    for name, f in _PROBES.items():
        ref = full
        if name == "i(D1)" and second_engine != "trbdf2":
            ref = _trbdf2(1e-4)
        y = f(ref)
        out[name] = (np.abs(f(second)[w] - np.interp(t2[w], np.asarray(
            ref.times), y)).max() / np.abs(y).max())
    return out


@pytest.mark.parametrize("first_engine,second_engine",
                         [("pwl", "trbdf2"), ("trbdf2", "pwl")])
def test_a_lost_coss_block_is_caught_in_both_directions(first_engine,
                                                       second_engine):
    """The seam test must see a snapshot whose Coss block was dropped.
    Measured with the block zeroed: into TR-BDF2, v(Cj) 0.82 (intact
    6e-7); into the fixed engine, i(D1) 4.1e-2 against the TR-BDF2 run
    (intact 2.1e-7) while v(Cj) stays at 2e-6 — the fixed engine moves
    the lost charge through the conducting D1."""
    first = _RUN[first_engine](_T_SPLIT)
    snap = first.final_snapshot
    snap.coss_history = [0.0] * len(list(snap.coss_history))
    second = _RUN[second_engine](1e-4, resume_from=snap)
    full = _RUN[second_engine](1e-4)
    errs = _seam_errors(second_engine, second, full)
    assert max(errs.values()) > 2e-2, errs


def test_a_trbdf2_snapshot_on_a_gate_edge_is_refused_by_the_fixed_engine():
    """TR-BDF2 ending exactly on an edge keeps the pre-edge x but has
    already replaced the device derivative slots with the post-edge ones
    (a 22.9 kA Coss discharge through the closed switch). The fixed engine
    started from that pair rang and aborted at the next edge (Newton at
    2e4 A residuals); it now refuses by name. TR-BDF2 resumes from it."""
    snap = _trbdf2(5e-5).final_snapshot
    assert snap.post_edge_derivatives
    assert not _trbdf2(4.77e-5).final_snapshot.post_edge_derivatives
    with pytest.raises(ValueError, match="gate edge"):
        _pwl(1e-4, resume_from=snap)
    again = pickle.loads(pickle.dumps(snap))
    assert again.post_edge_derivatives
    res = _trbdf2(1e-4, resume_from=again)
    assert np.isfinite(np.asarray(res.states)).all()


def test_a_snapshot_of_another_circuit_is_refused_by_block_name():
    """Same number of unknowns, different device kinds: only the device
    layout tells them apart."""
    def sat():
        b = p.CircuitBuilder()
        b.add_sine_voltage_source("V", "a", "gnd", v_dc=0.0,
                                  v_amplitude=50.0, frequency=1e3, phase=0.0)
        b.add_saturable_inductor("Ls", "a", "gnd", 1e-3, 5.0, 5e-5)
        return b

    def linear():
        b = p.CircuitBuilder()
        b.add_sine_voltage_source("V", "a", "gnd", v_dc=0.0,
                                  v_amplitude=50.0, frequency=1e3, phase=0.0)
        b.add_inductor("Ls", "a", "gnd", 1e-3)
        return b

    snap = p.simulate(sat(), t_end=1e-4, dt=1e-6, engine="pwl").final_snapshot
    for engine in ("pwl", "trbdf2"):
        with pytest.raises(ValueError, match="linear_companion"):
            p.simulate(linear(), t_end=2e-4, dt=1e-6, engine=engine,
                       resume_from=snap)


def test_a_pulsim_2_0_pickle_resumes_like_the_snapshot_it_came_from():
    first = _pwl(4.2e-5)
    snap = first.final_snapshot
    # the 2.0 layout: t, x, history, diode_on, the five stateful fields,
    # valid, stamped_mask, restart state (12 fields; 10 and 11 also read)
    old = (snap.t, list(snap.x), list(snap.history), list(snap.diode_on),
           list(snap.saturable_history), list(snap.coss_history),
           list(snap.lauritzen_history), list(snap.igbt_tail_history),
           list(snap.pmsm_history), True, list(snap.stamped_mask),
           int(snap.restart_steps_left))
    legacy = SolverSnapshot.__new__(SolverSnapshot)
    legacy.__setstate__(old)
    assert list(legacy.dev_layout) == []            # converted at resume
    # 2.0 carried no grid: compare with the same snapshot without one
    snap.grid_origin = float("nan")
    a = _pwl(1e-4, resume_from=snap)
    b = _pwl(1e-4, resume_from=legacy)
    assert np.array_equal(np.asarray(a.states), np.asarray(b.states))


def test_a_dev_state_without_its_layout_is_refused():
    snap = _pwl(4.2e-5).final_snapshot
    values = list(snap.dev_state)
    snap.history = list(snap.history)      # now a 2.0-style payload
    snap.dev_state = values                # values with no layout
    with pytest.raises(ValueError, match="no dev_layout"):
        _pwl(1e-4, resume_from=snap)


def test_a_newer_pickle_version_is_refused_by_version():
    snap = SolverSnapshot.__new__(SolverSnapshot)
    with pytest.raises(RuntimeError, match="pickle version 6"):
        snap.__setstate__(("SolverSnapshot", 6, 0.0))
    with pytest.raises(RuntimeError, match="pickle version 5"):
        snap.__setstate__(("SolverSnapshot", 5, 0.0))
    with pytest.raises(RuntimeError, match="pickle version 3"):
        snap.__setstate__(("Other", 3) + (0,) * 10)
    with pytest.raises(RuntimeError, match="pickle version 4"):
        snap.__setstate__(("SolverSnapshot", 4) + (0,) * 10)   # 12 fields


def test_a_version_3_pickle_loads_and_resumes_on_a_restarted_grid():
    """Version 3 = version 5 without the grid field: it loads, and resumes
    exactly as the same snapshot without a grid."""
    snap = _pwl(4.2e-5).final_snapshot
    v4 = snap.__getstate__()
    assert v4[1] == 5 and len(v4) == 19
    # version 5 before field 17 (the loss tracker's state) still loads
    v17 = SolverSnapshot.__new__(SolverSnapshot)
    v17.__setstate__(tuple(v4[:17]))
    assert np.array_equal(np.asarray(_pwl(1e-4, resume_from=v17).states),
                          np.asarray(_pwl(1e-4, resume_from=snap).states))
    old = SolverSnapshot.__new__(SolverSnapshot)
    old.__setstate__(("SolverSnapshot", 3) + tuple(v4[2:12]))
    assert np.isnan(old.grid_origin) and not old.post_edge_derivatives
    snap.grid_origin = float("nan")
    a = _pwl(1e-4, resume_from=snap)
    b = _pwl(1e-4, resume_from=old)
    assert np.array_equal(np.asarray(a.states), np.asarray(b.states))


@pytest.mark.parametrize("path", ["legacy", "interface"])
def test_every_pickle_layout_resumes_the_stored_charge_kinds_on_both_paths(path):
    """Every layout this Pulsim reads (2.0: 10/11/12 fields; version 3:
    12/13; version 4: 13/14/15; version 5: 17/18) loads a snapshot holding a Lauritzen
    diode and an IGBT tail taken on either device path, and resumes on
    that path exactly as the snapshot it came from (the layouts without a
    grid as the same snapshot without one)."""
    snap = _pwl(4.2e-5, device_path=path).final_snapshot
    assert list(snap.lauritzen_history)[0] != 0.0     # charge really stored
    assert list(snap.igbt_tail_history)[0] != 0.0
    v5 = snap.__getstate__()
    assert v5[1] == 5 and len(v5) == 19
    old = (snap.t, list(snap.x), list(snap.history), list(snap.diode_on),
           list(snap.saturable_history), list(snap.coss_history),
           list(snap.lauritzen_history), list(snap.igbt_tail_history),
           list(snap.pmsm_history), True, list(snap.stamped_mask),
           int(snap.restart_steps_left))
    # version 4's variants around the same core: grid only (13 fields);
    # marker + grid (14, the PR); grid + marker (+ controller) (14 / 15, the
    # device interface); marker + grid + controller (15, its last form)
    core, marker, grid = tuple(v5[2:12]), v5[12], v5[13]
    v4 = ("SolverSnapshot", 4) + core
    with_grid = [v4 + (grid,), v4 + (marker, grid), v4 + (grid, marker),
                 v4 + (grid, marker, None), v4 + (marker, grid, None),
                 v5[:17], v5[:18], v5]
    no_grid = [("SolverSnapshot", 3) + core,
               ("SolverSnapshot", 3) + core + (marker,),
               old, old[:11], old[:10]]

    def resume(s):
        return np.asarray(_pwl(1e-4, resume_from=s, device_path=path).states)

    def load(state):
        s = SolverSnapshot.__new__(SolverSnapshot)
        s.__setstate__(state)
        return s

    ref = resume(snap)
    for state in with_grid:
        assert np.array_equal(resume(load(state)), ref), len(state)
    snap.grid_origin = float("nan")
    ref = resume(snap)
    for state in no_grid:
        assert np.array_equal(resume(load(state)), ref), len(state)
