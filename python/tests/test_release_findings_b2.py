"""Release finding F-4-4: recorder window integrals on restart-rule steps.

The fixed engine's restart rule (``discontinuity_restart=True``) takes
three SDIRK(2,2) steps after an event; such a step is not a trapezoid
between its ends. Group B gave each restart step its LINES (the start
right limit 2·w1·X1 + (1 − 2·w1)·X2, the events inside the step), whose
trapezoid is the integrator's own quadrature. The in-loop recorder is fed
those lines, with the capacitor's companion current at each line point
(the stage currents' same combination), and the stored-array pass reads
the same lines through ``pulsim.with_event_limits``.

Truths: the window charge of a capacitor is C·Δv (an identity of the
integrator: round-off for windows whose ends are step ends); the buck's
energy balance from recorder outputs (Tellegen; supplied = dissipated +
stored change); the in-loop and the stored-array feeds agree; the exact
engine's window statistics on a piecewise-linear circuit.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import pulsim as p

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_recorder_device_readings as m  # noqa: E402

RULE = dict(engine="pwl", discontinuity_restart=True,
            enable_substep_state_correction=True)


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _run(fac, recs, **kw):
    b, fn, t_end, _, _ = fac()
    extra = {} if fn is None else {"switch_fn": fn}
    return b, _quiet(p.simulate, b, t_end, recorders=recs, **RULE, **extra, **kw)


@pytest.mark.parametrize("circ", ["buck", "rectifier"])
def test_window_charge_is_c_dv_at_step_ends(circ):
    """Windows whose ends are stored samples: ∫i_C = C·(v_C(b) − v_C(a))
    to round-off, with the samples' own (left-limit) capacitor voltages."""
    fac = {"buck": m._buck, "rectifier": m._rectifier}[circ]
    _, _, t_end, _, caps = fac()
    (cap, (nodes, c)), = caps.items()
    n = round(t_end / 1e-6)
    ks = [(int(0.07 * n), int(0.29 * n)), (int(0.31 * n), int(0.97 * n)), (1, n)]
    b, fn, _, _, _ = fac()
    extra = {} if fn is None else {"switch_fn": fn}
    res0 = _quiet(p.simulate, b, t_end, dt=1e-6, **RULE, **extra)
    t = np.asarray(res0.times)
    wins = [(t[a], t[z]) for a, z in ks]
    _, res = _run(fac, [p.Recorder("r", [f"i({cap})"], keep="none", windows=wins)], dt=1e-6)
    assert res.recorder_feed["mode"] == "in-loop", res.recorder_feed
    np.testing.assert_array_equal(np.asarray(res.times), t)   # the same run
    na, _, nb = nodes.partition(",")
    v = np.asarray(res.v(na)) - (np.asarray(res.v(nb)) if nb else 0.0)
    q = res.recorded["r"].windows[f"i({cap})"]["integral"]
    for k, (a, z) in enumerate(ks):
        dq = c * (v[z] - v[a])
        scale = c * np.max(np.abs(v))
        assert abs(q[k] - dq) <= 1e-11 * scale, (k, q[k], dq, (q[k] - dq) / dq)


@pytest.mark.parametrize("circ", ["buck", "rectifier"])
def test_window_charge_over_mid_step_windows(circ):
    """test_recorder_device_readings' mid-step windows under the rule: the
    residue is the two partial steps at the window ends (2e-4)."""
    fac = {"buck": m._buck, "rectifier": m._rectifier}[circ]
    _, _, t_end, _, caps = fac()
    (cap, (nodes, c)), = caps.items()
    wins = [(0.07 * t_end + 1.234e-9, 0.29 * t_end + 4.321e-9),
            (0.31 * t_end + 2.2e-9, 0.97 * t_end)]
    vq = f"v({nodes})"
    _, res = _run(fac, [p.Recorder("r", [f"i({cap})", vq], windows=wins)], dt=1e-6)
    _, res2 = _run(fac, [p.Recorder("r", [vq], keep="times",
                                    times=[x for w in wins for x in w])], dt=1e-6)
    q = res.recorded["r"].windows[f"i({cap})"]["integral"]
    ends = res2.recorded["r"][vq]
    for k in range(len(wins)):
        dq = c * (ends[2 * k + 1] - ends[2 * k])
        assert abs(q[k] - dq) <= 2e-4 * abs(dq) + 1e-12, (k, q[k], dq)


def _energy(dt):
    b, fn, t_end, _, _ = m._buck()
    A, B = m.A, m.B
    res = _quiet(p.simulate, b, t_end, switch_fn=fn, dt=dt, **RULE, recorders=[
        p.Recorder("e", m._BUCK_P, keep="none", windows=[(A, B)]),
        p.Recorder("s", ["v(out)", "i(L)"], keep="times", times=[A, B])])
    e = {q: res.recorded["e"].windows[q]["integral"][0] for q in m._BUCK_P}
    s = res.recorded["s"]
    stored = (0.5 * 100e-6 * np.diff(s["i(L)"] ** 2)[0]
              + 0.5 * 10e-6 * np.diff(s["v(out)"] ** 2)[0])
    supplied = -e["p(V)"]
    return e, stored, supplied


def test_buck_energy_balance_from_recorder_outputs():
    """supplied − dissipated − stored change: 3.6e-2 of supplied before
    (1.4e-5 without the rule); Tellegen on the same records."""
    e, stored, supplied = _energy(1e-6)
    dissipated = e["p(S1)"] + e["p(D1)"] + e["p(R)"]
    resid = supplied - dissipated - stored
    assert abs(resid) <= 1e-4 * supplied, resid / supplied
    assert abs(sum(e.values())) <= 1e-9 * supplied
    react = e["p(L)"] + e["p(C)"] - stored
    assert abs(react) <= 1e-4 * supplied, react / supplied


def test_in_loop_and_stored_array_feeds_agree():
    """The same recorder over the same rule run: fed in the loop, and
    fed from the stored arrays (inner events and right limits as rows)."""
    from pulsim._recorders import record_result
    b, fn, t_end, _, _ = m._buck()
    probes = ["v(out)", "i(L)", "i(R)", "v(sw)", "i(S1)", "i(D1)", "v(out)*i(L)"]
    wins = [(m.A, m.B), (0.05e-3 + 7e-10, 0.95e-3)]
    recs = [p.Recorder("w", probes, keep="every", windows=wins, period=1e-4)]
    res = _quiet(p.simulate, b, t_end, switch_fn=fn, dt=1e-6, **RULE, recorders=recs)
    assert res.recorder_feed["mode"] == "in-loop", res.recorder_feed
    assert len(res.inner_events["times"]) > 0       # the split's events are fed
    loop = res.recorded["w"]
    arr = _quiet(record_result, res, recs)["w"]
    np.testing.assert_array_equal(loop.times, arr.times)
    np.testing.assert_array_equal(loop.event_t, arr.event_t)
    for q in probes:
        scale = max(np.max(np.abs(loop[q])), 1e-300)
        np.testing.assert_allclose(loop[q], arr[q], rtol=0, atol=1e-12 * scale)
        for f in ("integral", "avg", "rms"):
            a, z = loop.windows[q][f], arr.windows[q][f]
            np.testing.assert_allclose(a, z, rtol=1e-12, atol=1e-15 * scale, err_msg=(q, f))
            a, z = loop.periods[q][f], arr.periods[q][f]
            np.testing.assert_allclose(a, z, rtol=1e-12, atol=1e-15 * scale, err_msg=(q, f))


def test_window_statistics_against_the_exact_engine():
    """The buck is piecewise linear: the exact engine's in-loop windows
    are the truth; the rule run's lines are second order."""
    b, fn, t_end, _, _ = m._buck()
    wins = [(m.A, m.B)]
    probes = ["i(L)", "v(out)", "p(R)"]
    ex = _quiet(p.simulate, b, t_end, switch_fn=fn, engine="dsed", output_rtol=1e-9,
                recorders=[p.Recorder("w", probes, keep="none", windows=wins)])
    errs = []
    for dt in (1e-6, 5e-7):
        b, fn, _, _, _ = m._buck()
        r = _quiet(p.simulate, b, t_end, switch_fn=fn, dt=dt, **RULE,
                   recorders=[p.Recorder("w", probes, keep="none", windows=wins)])
        errs.append([abs(r.recorded["w"].windows[q]["integral"][0]
                         / ex.recorded["w"].windows[q]["integral"][0] - 1) for q in probes])
    errs = np.asarray(errs)
    print("exact-engine window errors", errs, file=sys.stderr)
    assert np.all(errs[0] < 1e-3), errs
    assert np.all(errs[0] / np.maximum(errs[1], 1e-300) > 3.0), errs   # second order
