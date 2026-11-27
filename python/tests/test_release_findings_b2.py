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


@pytest.mark.parametrize("circ", [
    pytest.param("buck", marks=pytest.mark.xfail(strict=True, reason=(
        "release finding F-4-4: under control_tick='state' (the release default: a gate edge is "
        "placed inside its step) the recorder's window charge of the buck's capacitor misses "
        "C dv of the stored samples by 1.07e-6 of scale in two of three windows (-1.99242230e-05 C "
        "against -1.99243747e-05 C); 1.9e-16 with control_tick='label'"))),
    "rectifier"])
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


def _mid_step_residue(fac, dt):
    _, _, t_end, _, caps = fac()
    (cap, (nodes, c)), = caps.items()
    wins = [(0.07 * t_end + 1.234e-9, 0.29 * t_end + 4.321e-9),
            (0.31 * t_end + 2.2e-9, 0.97 * t_end)]
    vq = f"v({nodes})"
    _, res = _run(fac, [p.Recorder("r", [f"i({cap})"], keep="none", windows=wins)], dt=dt)
    _, res2 = _run(fac, [p.Recorder("r", [vq], keep="times",
                                    times=[x for w in wins for x in w])], dt=dt)
    q = res.recorded["r"].windows[f"i({cap})"]["integral"]
    ends = res2.recorded["r"][vq]
    dq = np.array([c * (ends[2 * k + 1] - ends[2 * k]) for k in range(len(wins))])
    return np.abs(q - dq) / np.abs(dq)


def test_window_charge_over_mid_step_windows():
    """test_recorder_device_readings' mid-step windows under the rule. A
    window end INSIDE a step reads the step's line there: no identity on
    any step that is not a trapezoid. Rectifier: no end lies in a restart
    step, the trapezoidal residue (2e-4 as on the trapezoidal engine).
    Buck: window 0 starts 1.2 ns into the restart step after the 70 us
    gate edge, and window 1 ends AT the 970 us edge, where keep='times'
    reads the right limit; both read the restart line's start value
    (2·w1·X1 + (1 − 2·w1)·X2, the right limit to O(h²)) as v_C: second
    order (7.4e-4 / 1.7e-3 at 1 us, 1.9e-4 / 4.2e-4 at 0.5 us). A true
    right limit there needs the consistent start of release group A
    (fixes/group-b2.md)."""
    r = _mid_step_residue(m._rectifier, 1e-6)
    assert np.all(r <= 2e-4), r
    r1, r2 = _mid_step_residue(m._buck, 1e-6), _mid_step_residue(m._buck, 5e-7)
    assert np.all(r1 / r2 > 3.0) and np.all(r1 <= 2e-3), (r1, r2)


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
    """supplied − dissipated − stored change: 3.57e-2 of supplied before,
    3.0e-5 now (1.4e-5 without the rule), second order (7.6e-6 at dt/2):
    the stored change reads v, i at B, 4 ns into a restart step, on the
    line's start value (test_window_charge_over_mid_step_windows).
    Tellegen on the same records."""
    e, stored, supplied = _energy(1e-6)
    dissipated = e["p(S1)"] + e["p(D1)"] + e["p(R)"]
    resid = supplied - dissipated - stored
    assert abs(resid) <= 1e-4 * supplied, resid / supplied
    e2, stored2, supplied2 = _energy(5e-7)
    resid2 = supplied2 - e2["p(S1)"] - e2["p(D1)"] - e2["p(R)"] - stored2
    assert abs(resid / supplied) > 3.0 * abs(resid2 / supplied2), (resid, resid2)
    assert abs(sum(e.values())) <= 1e-9 * supplied
    react = e["p(L)"] + e["p(C)"] - stored
    assert abs(react) <= 1e-4 * supplied, react / supplied


def test_in_loop_and_stored_array_feeds_agree():
    """The same recorder over the same rule run: fed in the loop, and
    fed from the stored arrays. The samples agree; the stored pass's event
    rows are the result's right limits, STATE VALUES at the events only
    (F-SB-1), a subset of the loop's jump rows (the loop's lines start at
    every restart step, at the quadrature start). The windows therefore
    differ: the loop's are the integrator's quadrature, the stored pass's
    the trapezoid of state values (second order; not compared here)."""
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
    jumps = loop.event_resolution != "within_dt"
    assert np.all(jumps), "a lined step's commutation is its jump rows, no bracket"
    assert 0 < arr.event_t.size < loop.event_t.size
    assert np.all(np.isin(arr.event_t, loop.event_t[jumps]))
    for q in probes:
        scale = max(np.max(np.abs(loop[q])), 1e-300)
        np.testing.assert_allclose(loop[q], arr[q], rtol=0, atol=1e-12 * scale)


def test_window_integrals_converge_second_order():
    """Products (v·i) included: the rule run's window integrals against a
    tight variable-step reference (trbdf2 rtol 1e-10, which agrees with
    the rule run at dt/16 to 3e-8 on i(L), v(out), p(R)), at dt and dt/2:
    second order (ratio >= 3; measured 3.6-6.0, -> 4 down to dt/8;
    p(R) and v(out) are 1e-7 off already at dt and reach the asymptote
    later). The exact engine is not used here: on this
    buck it differs from both by 1e-4 on i(L) (a model difference of its
    own, see fixes/group-b2.md)."""
    b, fn, t_end, _, _ = m._buck()
    wins = [(m.A, m.B)]
    probes = ["p(V)", "p(L)", "p(C)", "p(S1)", "p(D1)", "i(L)"]
    rec = [p.Recorder("w", probes, keep="none", windows=wins)]
    ref = _quiet(p.simulate, b, t_end, switch_fn=fn, engine="trbdf2", rtol=1e-10,
                 atol=1e-13, recorders=rec)
    x = {q: ref.recorded["w"].windows[q]["integral"][0] for q in probes}
    # energies against the supplied energy, the charge against itself
    scale = {q: abs(x["i(L)"] if q == "i(L)" else x["p(V)"]) for q in probes}
    errs = []
    for dt in (1e-6, 5e-7):
        b, fn, _, _, _ = m._buck()
        r = _quiet(p.simulate, b, t_end, switch_fn=fn, dt=dt, **RULE, recorders=rec)
        errs.append([abs(r.recorded["w"].windows[q]["integral"][0] - x[q]) / scale[q]
                     for q in probes])
    errs = np.asarray(errs)
    print("window errors (relative)", errs, file=sys.stderr)
    assert np.all(errs[0] < 1e-3), errs
    assert np.all(errs[0] / np.maximum(errs[1], 1e-300) > 3.0), errs


def test_stateful_reading_takes_restart_steps_whole():
    """A stateful device's reading reads the committed histories (no value
    inside a step): its feed takes restart steps whole (no line jumps), and
    KCL against the series resistor holds at every sample (the reviewer's
    finding: line points read end-of-step histories)."""
    b = m._NEWTON["nonlinear_capacitor"]()
    res = _quiet(p.simulate, b, 2e-3, dt=2e-6, **RULE,
                 recorders=[p.Recorder("r", ["i(Rs)", "i(X)"])])
    assert res.recorder_feed["mode"] == "in-loop", res.recorder_feed
    assert res.recorder_feed["n_jumps"] == 0, res.recorder_feed
    assert res.n_restart_steps > 0
    a = res.recorded["r"]
    scale = np.max(np.abs(a["i(Rs)"]))
    floor = 8 * 1e-6 * m.EPS * 5.0 / np.diff(a.times)
    assert np.all(np.abs(a["i(Rs)"] - a["i(X)"])[1:] <= 1e-6 * scale + floor)
