"""update-number-changing-release: the exact engine's output-grid rule
(``simulate(engine='exact', output_rtol=...)``, default off).

Without dt the exact engine sampled every 10 us (the default step cap). With
``output_rtol`` each exact step is cut so that linear interpolation between
consecutive samples is within output_rtol of every state, and the event
projection removes only the parasitic cluster (the 10 us cap was projecting a
real ~100 ns transient: ll06_buck_vcswitch's sample at D1's turn-off was 1.1e-2
of the span off the mpmath truth).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import pulsim

BENCH = Path(__file__).resolve().parents[2] / "benchmarks"


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _bench():
    pytest.importorskip("yaml")
    sys.path[:0] = [str(BENCH), str(BENCH / "tools")]
    import v1_reader
    return v1_reader


def _run(case, **kw):
    b, sw, obs = case.build()
    args = dict(engine="exact", switch_fn=sw, **kw)
    if obs is not None:
        args["step_observer"] = obs
    return pulsim.simulate(b, case.t_end, **args)


def test_ll06_d1_turn_off_matches_the_truth_without_dt():
    """Blocker 1: the right limit at a located diode turn-off is the state of
    the run, not the 10 us cap's projection of it (mpmath truth)."""
    pytest.importorskip("mpmath")
    v1 = _bench()
    import differential_runner as dr
    import exact_scoreboard as sb
    import exact_truth as et
    unit = "switching/buck/ll06_buck_vcswitch.yaml"
    case = v1.load(BENCH / "circuits" / unit)
    b, sw, _ = case.build()
    x0 = case._dc_start(b, sw)
    b, sw, _ = case.build()
    r = pulsim.simulate(b, case.t_end, engine="exact", switch_fn=sw, initial_state=x0,
                        output_rtol=1e-4)
    cols = dr.observables_of(case)
    sol = et.Solver(et.Circuit(BENCH / "circuits" / unit), sb.truth_dps(unit)).run()
    g, _, _ = sb.graded_columns(sol, cols)
    rec = sb.grade(sol, dr.traces(r, cols), g, sb.spans(sol, g), case.dt, 1e-3,
                   self_score=False)
    # 1.1e-2 at the PR head without the rule; 9.5e-12 with the netlist's dt
    assert rec["exact"] < 1e-7, rec["departure"]
    assert rec["exact_max_pt"] < 1e-5, rec["departure"]


def test_a_located_turn_off_on_an_exact_zero_is_past_the_crossing():
    """boost_classic at output_rtol=1e-4: the root search hit an exact 0 of
    Drect's ON current at a state the OFF mode read 1e-7 V forward; the
    decision kept the diode ON and the law check refused the run
    (DiodeModeInconsistentError at t = 1.06e-3). Under the rule an exact 0 is
    no sign: the event state is past the crossing."""
    v1 = _bench()
    case = v1.load(BENCH / "circuits" / "switching/boost/boost_classic.yaml")
    import next_defaults_rehearsal as rehearsal
    # the run a script without dt gets (the rehearsal's own runner)
    r = rehearsal._no_dt_run(case, "dsed", output_rtol=1e-4)
    assert np.all(np.diff(np.asarray(r.times)) >= 0)
    assert r.diagnostics["output_grid"]["n_output_capped"] == 0


def test_the_rule_bounds_linear_interpolation():
    """Every chord between consecutive samples is within output_rtol of the
    exact trajectory (checked at the chords' midpoints and quarter points
    against a run that samples there), on a damped resonant tank (complex
    eigenvalues) with a sine source (oscillator states)."""
    def build():
        b = pulsim.CircuitBuilder()
        b.add_sine_voltage_source("V", "a", "gnd", 0.0, 10.0, 2e3)
        b.add_resistor("R", "a", "b", 2.0)
        b.add_inductor("L", "b", "c", 100e-6)
        b.add_capacitor("C", "c", "gnd", 1e-6)
        b.add_resistor("Rl", "c", "gnd", 50.0)
        return b
    rtol = 1e-4
    r = pulsim.simulate(build(), 2e-3, engine="exact", output_rtol=rtol)
    t = np.asarray(r.times)
    assert np.all(np.diff(t) > 0)
    og = r.diagnostics["output_grid"]
    assert (og["n_output_capped"], og["n_output_unbounded"], og["output_budget_t"]) == (0, 0, None)
    # the trajectory at the chords' interior points, from runs whose grid holds them
    q = np.sort(np.concatenate([t[:-1] + f * np.diff(t) for f in (0.25, 0.5, 0.75)]))
    dense = pulsim.simulate(build(), 2e-3, engine="exact", dt=2e-8)
    td = np.asarray(dense.times)
    for name in ("c", "b"):
        y = np.asarray(r.v(name))
        yd = np.asarray(dense.v(name))
        scale = np.abs(yd).max()
        # the dense run's own chords are far finer: its interpolant is the truth here
        err = np.abs(np.interp(q, t, y) - np.interp(q, td, yd)).max() / scale
        assert err < 4 * rtol, (name, err)
    # without the rule every step end (1 us here) is stored; the rule stores what its
    # tolerance needs (a step end its chord covers is not stored, F-X-4)
    r0 = pulsim.simulate(build(), 2e-3, engine="exact")
    assert len(t) < len(r0.times)


def test_default_and_explicit_dt_keep_todays_samples():
    """output_rtol=None is today's run; a dt keeps today's samples whatever
    output_rtol says. Recorders no longer drop it (F-B2-1: their statistics
    are each segment's closed form, so the grid moves only the samples)."""
    def build():
        b = pulsim.CircuitBuilder()
        b.add_voltage_source("V", "a", "gnd", 1.0)
        b.add_resistor("R", "a", "b", 1.0)
        b.add_capacitor("C", "b", "gnd", 1e-5)
        return b
    base = pulsim.simulate(build(), 1e-4, engine="exact")
    assert "output_grid" not in base.diagnostics
    a = pulsim.simulate(build(), 1e-4, engine="exact", dt=1e-6)
    c = pulsim.simulate(build(), 1e-4, engine="exact", output_rtol=1e-4, dt=1e-6)
    np.testing.assert_array_equal(np.asarray(a.times), np.asarray(c.times))
    np.testing.assert_array_equal(np.asarray(a.states), np.asarray(c.states))
    rec = {"recorders": [pulsim.Recorder("vb", probes=["v(b)"])]}
    c = pulsim.simulate(build(), 1e-4, engine="exact", output_rtol=1e-4, **rec)
    assert "output_grid" in c.diagnostics
    on = pulsim.simulate(build(), 1e-4, engine="exact", output_rtol=1e-4)
    assert len(on.times) > len(base.times)
    for bad in (0.0, -1.0, float("inf"), True):
        with pytest.raises(ValueError, match="output_rtol"):
            pulsim.simulate(build(), 1e-4, engine="exact", output_rtol=bad)


def test_output_rtol_is_refused_by_name_off_the_exact_engine():
    b = pulsim.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_resistor("R", "a", "gnd", 1.0)
    for eng, kw in (("pwl", {"dt": 1e-6}), ("trbdf2", {})):
        with pytest.raises(pulsim.UnsupportedError) as ei:
            pulsim.simulate(b, 1e-5, engine=eng, output_rtol=1e-4, **kw)
        assert ei.value.feature == "output_rtol"


def test_events_keep_both_limits_and_order():
    """Every event instant is a sample (both limits where the state jumps,
    as today) and the samples stay in time order."""
    b = pulsim.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9, 0.0)
    b.add_inductor("L", "sw", "out", 100e-6)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", 50.0)
    T = 1e-5
    fn = pulsim.make_pwm_switch_fn(1 / T, 0.3, b.switch_index_of("S"), b.graph.num_switches, 0.0)
    r0 = pulsim.simulate(b, 40 * T, engine="exact", switch_fn=fn)
    r1 = pulsim.simulate(b, 40 * T, engine="exact", switch_fn=fn, output_rtol=1e-4)
    t1 = np.asarray(r1.times)
    assert np.all(np.diff(t1) >= 0)
    ev = np.asarray(r1.event_times)
    assert ev.size >= 60
    # each event instant is a sample of the run
    assert np.abs(t1[np.searchsorted(t1, ev).clip(0, t1.size - 1)] - ev).max() <= 1e-15
    # the rule moves samples, not decisions: the gate edges are the same
    assert np.asarray(r0.event_times).size >= 60


@pytest.mark.parametrize("case", ["buck", "vsi"])
def test_the_loss_port_does_not_depend_on_the_output_grid(case):
    """The exact engine's loss integrals are exact per interval: cutting the
    intervals by the rule leaves every energy and commutation count where it
    was (batch 14n measured <= 1.8e-13 relative; asserted at 1e-10 for the
    platforms: event instants are located by root search)."""
    from .test_exact_loss_port import E_OFF, E_ON, _buck, _vsi_case
    make, t_end = {"buck": (_buck, 202.5e-6), "vsi": (_vsi_case, 2.03e-3)}[case]
    runs = []
    for kw in ({}, {"output_rtol": 1e-5}):
        b, fn, names = make()
        for n in names:
            b.set_device_losses(n, Tj=125.0, E_on=E_ON, E_off=E_OFF)
        runs.append(pulsim.simulate(b, t_end, switch_fn=fn, engine="exact", **kw))
    off, on = runs
    assert len(on.times) > 2 * len(off.times)        # the rule really cut the steps
    for n in names:
        a, c = off.device_energy(n), on.device_energy(n)
        assert (a["n_on"], a["n_off"]) == (c["n_on"], c["n_off"]), n
        for k in ("E_on", "E_off", "E_cond_electrical", "E_impulse"):
            assert c[k] == pytest.approx(a[k], rel=1e-10, abs=1e-18), (n, k)


def _bridges(n):
    b = pulsim.CircuitBuilder()
    for k in range(n):
        for ph, ang in (("a", 0.0), ("b", -120.0), ("c", 120.0)):
            b.add_sine_voltage_source(f"V{k}{ph}", f"s{k}{ph}", "gnd", 0.0, 100.0, 50.0,
                                      np.deg2rad(ang + 6.0 * k))
            b.add_inductor(f"L{k}{ph}", f"s{k}{ph}", f"x{k}{ph}", 0.2e-3)
            b.add_diode(f"Dt{k}{ph}", f"x{k}{ph}", "p", 1e3, 1e-9, 0.0)
            b.add_diode(f"Db{k}{ph}", "n", f"x{k}{ph}", 1e3, 1e-9, 0.0)
    b.add_capacitor("C", "p", "n", 2e-3)
    b.add_resistor("R", "p", "n", 5.0)
    b.add_resistor("Rg", "n", "gnd", 1e6)
    return b


def test_a_parasitic_mode_at_rounding_does_not_set_the_output_step():
    """Two paralleled diode bridges: the 1 Mohm bus tie and the 1 nS OFF diodes make modes up
    to ~5e12 /s that the event projection leaves at rounding. The chord bound of a32b97ed read
    lambda^2 x rounding as curvature and cut every interval to a constant ~1e-8 s step (20 ms:
    14078 samples over the 10 us grid at 1e-5; ten banks ran > 8 min). Each mode's contribution
    now saturates at its amplitude (update-number-changing-release DECISIONS 5): 647 extra samples measured; bound 2x."""
    off = pulsim.simulate(_bridges(2), 0.02, engine="exact")
    on = pulsim.simulate(_bridges(2), 0.02, engine="exact", output_rtol=1e-5)
    t = np.asarray(on.times)
    assert np.all(np.diff(t) >= 0)
    og = on.diagnostics["output_grid"]
    assert (og["n_output_capped"], og["n_output_unbounded"], og["output_budget_t"]) == (0, 0, None)
    assert len(t) - len(off.times) <= 1300


def test_the_run_budget_stops_the_rule_with_one_warning(monkeypatch):
    """Past the run's sample budget the rule stops cutting (the default grid), names the time in
    diagnostics['output_grid']['output_budget_t'] and warns once; event instants stay samples."""
    from pulsim import _dsed_dispatch
    b = pulsim.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, 10.0, 2e3)
    b.add_resistor("R", "a", "b", 2.0)
    b.add_inductor("L", "b", "c", 100e-6)
    b.add_capacitor("C", "c", "gnd", 1e-6)
    b.add_resistor("Rl", "c", "gnd", 50.0)
    full = pulsim.simulate(b, 2e-3, engine="exact", output_rtol=1e-4)
    monkeypatch.setattr(_dsed_dispatch, "OUTPUT_BUDGET", 50)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = pulsim.simulate(b, 2e-3, engine="exact", output_rtol=1e-4)
    budget = [x for x in w if "sample budget" in str(x.message)]
    assert len(budget) == 1 and budget[0].category is RuntimeWarning
    tb = r.diagnostics["output_grid"]["output_budget_t"]
    assert tb is not None and 0.0 < tb < 2e-3
    t = np.asarray(r.times)
    assert np.all(np.diff(t) > 0)
    assert len(t) > len(full.times)   # past the budget every step end is stored again
    # past the budget the default grid of a run without the rule (the last step ends at t_end)
    grid = np.diff(np.asarray(pulsim.simulate(b, 2e-3, engine="exact").times)).max()
    assert np.diff(t[t > tb])[:-1].min() >= 0.99 * grid


def test_the_first_sample_after_an_event_is_placed_by_the_rule():
    """F-X-7's sample past an event sat a fixed 2 ps out: a switch closing on an RC
    (tau = C / (1/R + g_on) = 49.75 ps, resolved: above 1e-6 of the 5 us interval)
    leaves the chord 19.7 x the rule's bound there. It is now the rule's chord from the
    right limit, at most 2 ps and at least 64 eps max(1, |t|); against the closed form."""
    F, V, R, C, gon, goff, rtol = 1e5, 100.0, 10.0, 1e-9, 20.0, 1e-9, 1e-5
    b = pulsim.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", V)
    b.add_resistor("R", "in", "sw", R)
    b.add_switch("S", "sw", "gnd", gon, goff)
    b.add_capacitor("C", "sw", "gnd", C)
    sw = pulsim.make_pwm_switch_fn(F, 0.5, b.switch_index_of("S"), b.graph.num_switches, 0.5 / F)
    r = pulsim.simulate(b, 4 / F, engine="exact", switch_fn=sw, output_rtol=rtol)
    t, v = np.asarray(r.times), np.asarray(r.v("sw"))
    assert np.all(np.diff(t) > 0)
    og = r.diagnostics["output_grid"]
    assert (og["n_output_capped"], og["n_output_unbounded"]) == (0, 0)
    tol = rtol * (v.max() - v.min())   # the rule's bound: rtol x the row's range (~100 V)
    for k, te in enumerate(np.arange(1, 8) * 0.5 / F):
        i = int(np.argmin(np.abs(t - te)))
        assert t[i] == pytest.approx(te, rel=1e-9)
        g = gon if k % 2 == 0 else goff
        tau, vinf = C / (1 / R + g), V / R / (1 / R + g)
        h1 = t[i + 1] - t[i]
        assert 64 * np.finfo(float).eps * max(1.0, t[i]) <= h1 <= 2e-12 * (1 + 1e-9)
        s = np.linspace(t[i], t[i + 1], 201)
        exact = vinf + (v[i] - vinf) * np.exp(-(s - t[i]) / tau)
        chord = np.interp(s, t[i:i + 2], v[i:i + 2])
        assert np.abs(chord - exact).max() <= 1.02 * tol, (k, h1)   # 0.988 measured
        if g == gon:
            # control: the fixed 2 ps chord on the same closed form (19.7 x)
            s2 = np.linspace(t[i], t[i] + 2e-12, 201)
            e2 = vinf + (v[i] - vinf) * np.exp(-(s2 - t[i]) / tau)
            c2 = np.interp(s2, s2[[0, -1]], e2[[0, -1]])
            assert np.abs(c2 - e2).max() > 10 * tol
            # samples per transient: the rule's chord count over A e^{-s/tau} is
            # 2 sqrt(A / (8 tol)) = 224 (230 measured): bounded, no collapse
            assert ((t > t[i]) & (t <= t[i] + 20 * tau)).sum() <= 300


def test_the_analytic_gate_reads_the_projected_flow():
    """DECISIONS 8: the rule's promise is against the flow with the parasitic cluster
    projected at events, and each projection is recorded; the rehearsal's analytic
    gate reads the closed form with the recorded mode removed. hard_switched_rc: the
    closing switch gives C a 1 ps mode (tau = C / (1/R + g_on)); F-X-7 stores the
    projected value 2 ps past each closing, where the unprojected form reads 13.54 V."""
    _bench()
    import metrics
    import reference_status as rs
    from analytic import BY_ID
    c = BY_ID["hard_switched_rc"]
    with rs.extra_kwargs(dict(output_rtol=1e-5)):
        r = c.run("dsed")
    t, y = np.asarray(r.times), np.asarray(c.observe(r))
    assert np.all(np.diff(t) >= 0)
    P = r.diagnostics["output_grid"]["projections"]
    assert len(P) == 5
    ceil = rs.ANALYTIC[c.id]["dsed"]["ceiling"]
    kw = dict(abs_scale=c.abs_scale, exclude_t0=c.exclude_t0)
    # control: the records ignored, the cell disagrees (0.0676)
    assert metrics.err_rms(t, y, c.exact, **kw) > 1e3 * ceil
    ref = metrics.projected_reference(c.exact, P)
    assert ref.n_applied == 5
    assert metrics.err_rms(t, y, ref, **kw) <= ceil
    # the projected closed form inside the window is the closed switch's settled value
    vinf = 100.0 / 10.0 / (1 / 10.0 + 1e3)
    k = int(np.searchsorted(t, P[0]["t"], side="right"))
    assert t[k] - P[0]["t"] < 40 * P[0]["tau"]
    assert ref(t[k:k + 1])[0] == pytest.approx(vinf, rel=1e-6)
    # a record whose tau the closed form does not decay at is not applied, nor is none
    assert metrics.projected_reference(c.exact, [dict(p, tau=10 * p["tau"]) for p in P]) is c.exact
    assert metrics.projected_reference(c.exact, ()) is c.exact
    # the caller's parasitic bound: a record above it is not applied (rehearsal: 1e-6 of the span)
    assert metrics.projected_reference(c.exact, P, tau_max=0.5 * P[0]["tau"]) is c.exact
    assert metrics.projected_reference(c.exact, P, tau_max=1e-6 * c.t_end).n_applied == 5
    # a duplicate record a few ulps later is the same mode: removed once
    dup = P + [dict(P[0], t=float(np.nextafter(P[0]["t"], 1.0)))]
    assert metrics.projected_reference(c.exact, dup).n_applied == 5
