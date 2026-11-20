"""Recorders fed IN THE LOOP on the exact engine (add-probes-and-recorders R5,
task 0b.11; add-exact-propagator-strategy session 26).

The event-driven integrator (the route diodes, loss ports, controllers and
resume take) drives `record::LoopFeed` from its step hooks through
`dsed/loop_feed_adapter.hpp`: the scheduler's reduced states become full MNA
vectors (`recover_full`), an event reported inside the step it ends becomes an
event row at its instant after that step's sample. The feed sees every
accepted step whatever `store_every` keeps, and its state travels in the
snapshot (field 18), so a resumed run continues the record.

What is exact and what is sampled: every kept row and both limits of every
event row are the engine's exact states at those instants (event rows carry
resolution 'exact'); window and period statistics are the recorder's exact
integrals of the straight line between consecutive accepted states — sampled,
like every engine's. They agree with trbdf2 at tight tolerance to the accuracy
of that line (checked below, time-weighted).

Not in the loop on the exact engine (the stored-array pass, named in
`result.recorder_feed['why']`): the default 'auto' integrator of a diode-free
circuit (PEDSimulatorAuto has no step hooks), the junction temperature loop
(its kick corrects the state after the hooks), a fresh run with t_start.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest

import pulsim as p
from pulsim._recorders import record_result

from .test_device_losses_kernel import _vsi
from .test_exact_resume import _bridge_case, _ccm, _dcm
from .test_recorders import _rectifier

T_UNIT = 1e-6


def _run(make, t_end, recs, **kw):
    b, fn = make()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end, switch_fn=fn, engine="exact", recorders=recs, **kw)


def _buck_rec(t_end, keep="every"):
    return [p.Recorder("r", ["v(vout)", "i(L)", "v(sw)", "i(Q1)", "i(D1)", "v(vout)*i(L)"],
                       keep=keep, windows=[(0.25 * t_end, t_end)], period=10e-6)]


def _bridge_rec(t_end, keep="every"):
    return [p.Recorder("r", ["v(dp,dn)", "i(Ls)", "i(D1)", "i(D3)", "v(dp,dn)*i(Ls)"],
                       keep=keep, windows=[(0.5 * t_end, t_end)], period=20e-3)]


def _vsi_rec(t_end, keep="every"):
    return [p.Recorder("r", ["i(La)", "i(Lb)", "v(a,b)", "v(n)", "i(INV__HSa)"],
                       keep=keep, windows=[(0.25 * t_end, t_end)], period=1e-3)]


def _rect():
    return _rectifier(), None


def _ccm_losses():
    """The loss port (fixed Tj) and the feed on the same hooks."""
    b, fn = _ccm()
    ax = dict(v_axis=[0.0, 400.0, 800.0], i_axis=[0.0, 30.0, 60.0], tj_axis=[25.0, 150.0])
    rng = np.random.default_rng(7)
    for n in ("Q1", "D1"):
        b.set_device_losses(n, Tj=125.0,
                            E_on=p.LossTable(**ax, energy=rng.uniform(1e-3, 5e-3, (3, 3, 2))),
                            E_off=p.LossTable(**ax, energy=rng.uniform(1e-3, 5e-3, (3, 3, 2))))
    return b, fn


def _rect_rec(t_end, keep="every"):
    return [p.Recorder("r", ["v(dp,dn)", "i(D1)", "i(D4)", "v(dp,dn)*i(RL)"],
                       keep=keep, windows=[(7.3e-3, 33.1e-3)], period=10e-3)]


def _pulse_rc():
    """A pulse source's corners change no mask (no event hook): the record
    still has them as event rows, and its corner at t_end as the left limit."""
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("Vp", "in", "gnd", 0.0, 10.0, 0.0, 1e-4, 2e-4, 0.0, 0.0)
    b.add_resistor("R", "in", "a", 100.0)
    b.add_capacitor("C", "a", "gnd", 1e-7)
    b.add_diode("D1", "a", "o", 1e3, 1e-9)
    b.add_resistor("RL", "o", "gnd", 1e3)
    return b, None


def _pulse_rec(t_end, keep="every"):
    return [p.Recorder("r", ["v(in)", "v(a)", "v(o)", "v(in)*v(a)"],
                       keep=keep, windows=[(0.0, t_end), (1.5e-4, 4.5e-4)], period=2e-4)]


CASES = {
    "pulse_corners": (_pulse_rc, 6e-4, _pulse_rec),
    "rectifier": (_rect, 40e-3, _rect_rec),
    "buck_ccm_losses": (_ccm_losses, 200e-6, _buck_rec),
    "buck_ccm": (_ccm, 200e-6, _buck_rec),
    "buck_dcm": (_dcm, 300e-6, _buck_rec),
    "bridge": (_bridge_case, 40e-3, _bridge_rec),
    "vsi": (_vsi, 2e-3, _vsi_rec),
}


def _same(a, b, rel=1e-12):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape
    scale = max(1.0, float(np.abs(b).max(initial=0.0)))
    np.testing.assert_allclose(a, b, rtol=0, atol=rel * scale)


@pytest.mark.parametrize("name", list(CASES))
def test_in_loop_recorders_match_the_stored_pass_on_the_same_run(name):
    """Kept rows and event rows against the stored-array pass over the same
    run's stored samples (store_every=1: every accepted step is stored), at
    1e-12. Window and period statistics are the stored pass's line only where
    the loop had no closed form (none here: since the time-varying sources'
    closed forms, pulse_corners, rectifier and bridge are closed too). Where
    every piece is closed (F-B2-1) the integrals (avg, rms) are the
    trajectory's, checked against a run on a finer grid (dt = 1e-7, or
    t_end / 2e4 on the 40 ms runs: at dt = 1e-7 the bridge's 3.5e7 pieces
    add up 1.4e-10 of rounding, its bus average 1.9e-12 off the mpmath truth
    against 6.6e-14 on the default grid) instead; min / max are still the samples', but a window or period
    edge inside a step takes the closed form's value there, not the chord's
    (measured <= 4.4e-11 of the scale here, so 1e-9)."""
    make, t_end, rec = CASES[name]
    recs = rec(t_end)
    res = _run(make, t_end, recs)
    assert res.recorder_feed["mode"] == "in-loop", res.recorder_feed
    cf = res.recorder_feed["closed_form"]
    assert cf["n_closed"] == 0 or cf["n_line"] == 0, cf
    closed = cf["n_closed"] > 0
    other = (_run(make, t_end, rec(t_end), dt=max(1e-7, t_end / 2e4)).recorded["r"] if closed
             else record_result(res, recs)["r"])
    mm = 1e-9 if closed else 1e-11
    # the closed forms' rounding adds up over the fine grid's pieces: worst
    # v(vout)*i(L)'s rms on buck_dcm, 1.7e-11 at dt = 1e-7 (3021 steps)
    ia = 1e-10 if closed else 1e-11
    loop = res.recorded["r"]
    stored = record_result(res, recs)["r"]
    _same(loop.times, stored.times)
    for pr in loop.probes:
        _same(loop[pr], stored[pr])
    _same(loop.event_t, stored.event_t)
    assert list(loop.event_resolution) == list(stored.event_resolution)
    assert set(loop.event_resolution) <= {"exact"}
    if name == "pulse_corners":   # v(in) is 10 V on half of every period: rms sqrt(50)
        assert len(loop.event_t) >= 5
        np.testing.assert_allclose(loop.windows["v(in)"]["rms"][0], np.sqrt(50.0), rtol=1e-12)
    for pr in loop.probes:
        _same(loop.event_left[pr], stored.event_left[pr])
        _same(loop.event_right[pr], stored.event_right[pr])
        for f in ("min", "max"):
            _same(loop.windows[pr][f], stored.windows[pr][f], mm)
            _same(loop.periods[pr][f], stored.periods[pr][f], mm)
        for f in ("integral", "avg", "rms"):
            _same(loop.windows[pr][f], other.windows[pr][f], ia)
            _same(loop.periods[pr][f], other.periods[pr][f], ia)


@pytest.mark.parametrize("name,t_split", [("buck_ccm", 100e-6), ("buck_dcm", 150e-6),
                                          ("vsi", 1e-3)])
def test_a_resumed_run_continues_the_record(name, t_split):
    """Field 18 on the exact engine: a run split at a gate edge and resumed
    gives the uninterrupted run's windows, periods and kept rows after the
    split (1e-12; bit-identical on this machine)."""
    make, t_end, rec = CASES[name]
    u = _run(make, t_end, rec(t_end))
    a = _run(make, t_split, rec(t_end))
    assert len(a.final_snapshot.recorder_state) > 0
    snap = pickle.loads(pickle.dumps(a.final_snapshot))
    r = _run(make, t_end, rec(t_end), resume_from=snap)
    assert r.recorder_feed["mode"] == "in-loop" and "continued" in r.recorder_feed["resume"]
    ru, rr = u.recorded["r"], r.recorded["r"]
    _same(rr.times, ru.times)
    _same(rr.event_t, ru.event_t)
    for pr in ru.probes:
        _same(rr[pr], ru[pr])
        _same(rr.event_right[pr], ru.event_right[pr])
        for f in ("integral", "avg", "rms", "min", "max"):
            _same(rr.windows[pr][f], ru.windows[pr][f], 1e-11)
            _same(rr.periods[pr][f], ru.periods[pr][f], 1e-11)


def test_window_statistics_agree_with_trbdf2_at_tight_tolerance():
    """Time-weighted window average and RMS of the buck's v(vout), i(L) on the
    exact engine (in the loop) and on trbdf2 at rtol 1e-10. The exact
    engine's statistics are each segment's closed form (F-B2-1), so the
    default grid (one exact step per half period) suffices; the line between
    its states had put v(vout)'s average 2.7e-3 V off."""
    t_end = 200e-6
    recs = [p.Recorder("w", ["v(vout)", "i(L)"], keep="none", windows=[(100e-6, 200e-6)])]
    ex = _run(_ccm, t_end, recs)
    assert ex.recorder_feed["closed_form"]["n_line"] == 0
    assert ex.recorder_feed["mode"] == "in-loop"
    b, fn = _ccm()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tr = p.simulate(b, t_end, switch_fn=fn, engine="trbdf2", rtol=1e-10, atol=1e-12,
                        recorders=recs)
    for pr, scale in (("v(vout)", 12.0), ("i(L)", 1.2)):
        for f in ("avg", "rms"):
            e, t = ex.recorded["w"].windows[pr][f][0], tr.recorded["w"].windows[pr][f][0]
            assert abs(e - t) < WINDOW_BOUND * scale, (pr, f, e, t)


WINDOW_BOUND = 5e-6   # trbdf2's own error at rtol 1e-10 (was measured at dt = 0.1 us)


def test_the_junction_loop_keeps_the_stored_pass_and_says_why():
    from .test_exact_junction_loop import _buck
    b, fn, _ = _buck()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, 50e-6, switch_fn=fn, engine="exact",
                         recorders=[p.Recorder("r", ["v(vout)"])])
    assert res.recorder_feed["mode"] == "stored-arrays"
    assert "junction temperature loop" in res.recorder_feed["why"]
