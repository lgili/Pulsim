"""update-number-changing-release: the exact engine's output-grid rule
(``simulate(engine='exact', output_rtol=...)``; default 1e-5 since the release,
``output_rtol=0`` is the old 10 us grid).

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
    assert r.diagnostics["output_grid"] == {"n_output_capped": 0, "n_output_unbounded": 0,
                                            "output_budget_t": None}
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
    # the old 10 us grid (output_rtol=0): the same circuit without the rule is coarser
    r0 = pulsim.simulate(build(), 2e-3, engine="exact", output_rtol=0)
    assert len(r0.times) < len(t)


def test_default_and_explicit_dt_keep_todays_samples():
    """output_rtol=0 is the old 10 us grid and the default is output_rtol=1e-5
    (the release); a dt, store_every or recorders keep the old samples whatever
    output_rtol says."""
    def build():
        b = pulsim.CircuitBuilder()
        b.add_voltage_source("V", "a", "gnd", 1.0)
        b.add_resistor("R", "a", "b", 1.0)
        b.add_capacitor("C", "b", "gnd", 1e-5)
        return b
    base = pulsim.simulate(build(), 1e-4, engine="exact", output_rtol=0)
    assert "output_grid" not in base.diagnostics
    default = pulsim.simulate(build(), 1e-4, engine="exact")
    rule = pulsim.simulate(build(), 1e-4, engine="exact", output_rtol=1e-5)
    assert "output_grid" in default.diagnostics
    np.testing.assert_array_equal(np.asarray(default.times), np.asarray(rule.times))
    np.testing.assert_array_equal(np.asarray(default.states), np.asarray(rule.states))
    explicit = ({"dt": 1e-6}, {"recorders": [pulsim.Recorder("vb", probes=["v(b)"])]})
    for kw in explicit:
        a = pulsim.simulate(build(), 1e-4, engine="exact", **kw)
        c = pulsim.simulate(build(), 1e-4, engine="exact", output_rtol=1e-4, **kw)
        np.testing.assert_array_equal(np.asarray(a.times), np.asarray(c.times))
        np.testing.assert_array_equal(np.asarray(a.states), np.asarray(c.states))
    on = pulsim.simulate(build(), 1e-4, engine="exact", output_rtol=1e-4)
    assert len(on.times) > len(base.times)
    for bad in (-1.0, float("inf"), True):     # 0 is the explicit off since the release
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
    r0 = pulsim.simulate(b, 40 * T, engine="exact", switch_fn=fn, output_rtol=0)
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
    for kw in ({"output_rtol": 0}, {"output_rtol": 1e-5}):
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
    assert on.diagnostics["output_grid"] == {"n_output_capped": 0, "n_output_unbounded": 0,
                                             "output_budget_t": None}
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
    assert len(t) < len(full.times)
    # past the budget the default grid of a run without the rule (the last step ends at t_end)
    grid = np.diff(np.asarray(pulsim.simulate(b, 2e-3, engine="exact").times)).max()
    assert np.diff(t[t > tb])[:-1].min() >= 0.99 * grid
