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
    assert joined == list(snap.dev_state)


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_each_block_holds_its_device_s_state_at_the_end(engine):
    """Checked against the run's own waveforms, not against the snapshot:
    the saturable block is (λ, i, v, ...) of L, the Coss block (v, q, i)
    of Cj, and the IGBT carries tail charge after its turn-off."""
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
    assert np.array_equal(np.asarray(snap.x), np.asarray(res.states)[-1])


def test_split_fixed_step_run_matches_the_continuous_one():
    """Not bit for bit: the resumed run's time grid starts from t_split,
    so its sample times differ from the continuous run's by ~1e-20 s and
    a gate edge lands a rounding apart (measured: states equal up to the
    edge at 5e-5, then within 8e-12)."""
    full = _pwl(1e-4)
    first = _pwl(4.2e-5)
    snap = pickle.loads(pickle.dumps(first.final_snapshot))
    second = _pwl(1e-4, resume_from=snap)
    t_full = np.asarray(full.times)
    x_full = np.asarray(full.states)
    t2 = np.asarray(second.times)
    x2 = np.asarray(second.states)
    # compare on the time points the two grids share (to 1e-15 s)
    k = np.clip(np.searchsorted(t_full, t2), 1, len(t_full) - 1)
    k -= (t2 - t_full[k - 1]) < (t_full[k] - t2)
    common = np.abs(t_full[k] - t2) < 1e-15
    assert common.mean() > 0.99
    scale = np.abs(x_full).max(axis=0) + 1e-12
    assert (np.abs(x_full[k[common]] - x2[common]) / scale).max() < 1e-9


@pytest.mark.parametrize("first_engine,second_engine",
                         [("pwl", "trbdf2"), ("trbdf2", "pwl")])
def test_a_snapshot_resumes_on_the_other_engine(first_engine, second_engine):
    """Split off any gate edge: a TR-BDF2 run that ends exactly ON an edge
    stores the edge sample with the pre-edge solve, and the fixed engine
    resumed from it aborts at the next edge (found while writing this
    test; the same on the build before step 3)."""
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
    # checked up to the next gate edge. Measured with one block of the
    # snapshot zeroed (relative error of the probe): saturable 0.39 on
    # i(L); IGBT tail 0.066 / 0.0041 on i(Q1); Coss 0.82 on v(Cj) into
    # TR-BDF2 but not on these probes into the fixed engine (it moves a
    # branch current there); intact <= 7.1e-4.
    t2 = np.asarray(second.times)
    tf = np.asarray(full.times)
    w = (t2 > t_split) & (t2 < 4.99e-5)
    probes = {
        "i(L)": lambda r: np.asarray(r.i("L")),
        "v(Cj)": lambda r: np.asarray(r.v("vin")) - np.asarray(r.v("sw")),
        "i(Q1)": lambda r: np.asarray(r.i("Rsense")),
        "v(out)": lambda r: np.asarray(r.v("out")),
    }
    for name, f in probes.items():
        y_full = f(full)
        err = np.abs(f(second)[w] - np.interp(t2[w], tf, y_full)).max()
        assert err <= 2e-3 * np.abs(y_full).max(), name


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
    with pytest.raises(RuntimeError, match="pickle version 4"):
        snap.__setstate__(("SolverSnapshot", 4, 0.0))
