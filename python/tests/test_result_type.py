"""One result type on every engine (add-probes-and-recorders D3, sections
1, 2, 4, 0b.8 and p() readings): class identity, grid kind and run
statistics, the result's time-weighted measures, t_eval rows, and the
power of a two-terminal device."""
import numpy as np
import pytest

import pulsim as p

from .test_recorders import RUNS, _rc, _rectifier

ENGINES = sorted(RUNS)


@pytest.mark.parametrize("eng", ENGINES)
def test_every_engine_returns_a_simulation_result(eng):
    res = p.simulate(_rc(), t_end=1e-3, **RUNS[eng])
    assert isinstance(res, p.SimulationResult)
    assert res.stats.engine == {"exact": "dsed"}.get(eng, eng)
    assert res.grid == (p.GridKind.Uniform if eng == "pwl"
                        else p.GridKind.Accepted)
    if eng == "exact":
        assert res.n_accept == res.stats.n_accept > 0
        assert res.n_events == res.stats.n_mode_events
        assert res.cpu_time_seconds == res.stats.cpu_time_seconds
        with pytest.raises(AttributeError):
            res.n_accept = 3          # read-only, over res.stats
    else:
        # the exact engine's counters stay absent elsewhere, so
        # getattr(res, "n_accept", None) still tells the engines apart
        assert getattr(res, "n_accept", None) is None
    if eng == "trbdf2":
        assert res.stats.n_accept == res._trbdf2_stats["n_accept"] > 0
    t = np.asarray(res.times)
    assert t.ndim == 1 and np.asarray(res.states).shape[0] == t.size
    assert not res.empty() and res.num_steps() == t.size


def test_an_empty_result_is_constructible_and_filled_by_adapters():
    r = p.SimulationResult()
    assert r.num_steps() == 0 and r.switch_states is None
    r._adapter_fill(np.array([0.0, 1.0]), np.array([[1.0, 2.0], [3.0, 4.0]]),
                    np.array([[True, False, True], [False, False, True]]))
    np.testing.assert_array_equal(r.states, [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(r.switch_states,
                                  [[True, False, True], [False, False, True]])
    with pytest.raises(Exception, match="already filled"):
        r._adapter_fill(np.zeros(1), np.zeros((1, 2)))
    with pytest.raises(Exception, match="unknown RunStats key"):
        r._set_run_info("pwl", p.GridKind.Uniform, {"n_bogus": 1})


@pytest.mark.parametrize("eng", ENGINES)
def test_measures_equal_time_average_on_the_same_samples(eng):
    res = p.simulate(_rectifier(), t_end=40e-3, **RUNS[eng])
    a, b = 7.3e-3, 33.1e-3
    got = res.avg("v(dp,dn)", a, b)
    from pulsim._result_views import with_right_limits
    w = with_right_limits(res)
    t = np.asarray(w.times)
    v = np.asarray(w.v("dp")) - np.asarray(w.v("dn"))
    assert got == pytest.approx(p.time_average(t, v, a, b), rel=1e-12)
    assert res.rms("v(dp,dn)", a, b) == pytest.approx(
        p.time_rms(t, v, a, b), rel=1e-12)
    assert res.integral("v(dp,dn)", a, b) == pytest.approx(got * (b - a),
                                                           rel=1e-12)
    lo, hi = res.minmax("v(dp,dn)", a, b)
    assert lo <= got <= hi and hi == pytest.approx(100.0, rel=1e-3)
    # whole-run defaults, and at(): a sample is itself, mid-step is linear
    assert res.avg("v(dp,dn)") == pytest.approx(
        p.time_average(t, v), rel=1e-12)
    traw = np.asarray(res.times)
    k = traw.size // 2
    if eng != "exact":   # (the exact engine's sample k may be an event row)
        assert res.at("v(dp)", traw[k]) == pytest.approx(
            float(np.asarray(res.v("dp"))[k]), rel=1e-15, abs=1e-300)
    tm = 0.5 * (traw[k] + traw[k + 1])
    vals = res.at("v(dp)", np.array([traw[1], tm]))
    assert vals.shape == (2,)
    assert abs(vals[1] - res.at("v(dp)", tm)) == 0.0


def test_at_an_exact_engine_event_is_the_right_limit():
    res = p.simulate(_rectifier(), t_end=40e-3, engine="exact")
    idx = np.asarray(res.right_limit_index)
    assert idx.size
    k = int(idx[len(idx) // 2])
    t = float(np.asarray(res.times)[k])
    rl = np.asarray(res.right_limit_states)[len(idx) // 2]
    j = res._builder.node_id_of("dp")
    assert res.at("v(dp)", t) == rl[j]


@pytest.mark.parametrize("eng", ENGINES)
def test_times_rows_and_power_readings(eng):
    ts = (0.1e-3, 0.37e-3 + 1.234e-9, 1.0e-3)
    rec = p.Recorder("r", ["v(out)", "p(R1)", "p(V1)"], keep="times",
                     times=ts, windows=[(0.0, 1e-3)])
    res = p.simulate(_rc(), t_end=1e-3, recorders=[rec], **RUNS[eng])
    r = res.recorded["r"]
    np.testing.assert_array_equal(r.times, ts)
    assert r.interpolated.dtype == bool and r.interpolated.size == 3
    assert not r.interpolated[-1]          # the last sample
    # linear on the stored samples (the dense-output line), no new error
    t = np.asarray(res.times)
    vout = np.asarray(res.v("out"))
    for k, tk in enumerate(ts):
        assert r["v(out)"][k] == pytest.approx(np.interp(tk, t, vout),
                                               rel=1e-12)
    # p(R) = (V - v)^2 / R per sample; the source delivers what R and C take
    vr = np.asarray(res.v("in")) - np.asarray(res.v("out"))
    full = p.Recorder("f", ["p(R1)", "p(V1)", "v(in,out)*i(R1)"])
    rf = p.simulate(_rc(), t_end=1e-3, recorders=[full],
                    **RUNS[eng]).recorded["f"]
    np.testing.assert_allclose(rf["p(R1)"], rf["v(in,out)*i(R1)"], rtol=0,
                               atol=0)
    np.testing.assert_allclose(rf["p(R1)"][: t.size], vr * vr / 1e3,
                               rtol=1e-12, atol=1e-15)
    # the source's absorbed power is minus what it delivers: -V*i
    assert np.all(rf["p(V1)"][1:] < 0)
    w = r.windows
    # energy delivered = energy in R + energy stored in C (0.5*C*v^2)
    e_src = -w["p(V1)"]["integral"][0]
    e_r = w["p(R1)"]["integral"][0]
    v_end = float(np.asarray(res.v("out"))[-1])
    assert e_src == pytest.approx(e_r + 0.5e-6 * v_end * v_end, rel=2e-3)


@pytest.mark.parametrize("probe,match", [
    ("p(C1)", r"p\(C1\) of a capacitor"),
    ("i(C1)*2", r"i\(C1\) of a capacitor"),
])
def test_loop_only_device_readings_are_refused_on_the_exact_engine(probe, match):
    # read in the loop on pwl / trbdf2 (DECISIONS 20); the exact engine is
    # fed from its stored arrays, which hold no such reading
    with pytest.raises(p.UnsupportedError, match=match):
        p.simulate(_rc(), t_end=1e-3, engine="exact",
                   recorders=[p.Recorder("x", [probe])])


def test_engine_warnings_with_recorders_point_at_the_callers_line():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # (integrator= on trbdf2 used to warn; the option table now refuses
        # it by name. An undriven switch warned next, until the release made
        # undriven='refuse' the default and an explicit policy silent; the
        # warning here is the split's: an opaque gate edge off the grid)
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "in", "gnd", 1.0)
        b.add_switch("S", "in", "out", 1e3, 1e-9)
        b.add_resistor("R", "out", "gnd", 1.0)
        n = b.graph.num_switches

        def gate(t):
            m = p.SwitchStateMask(n)
            m.set(0, t >= 3.5e-6)
            return m
        p.simulate(b, t_end=1e-5, dt=1e-6, engine="pwl", switch_fn=gate,
                   recorders=[p.Recorder("x", ["v(out)"])])
    mine = [x for x in w if "could not be located" in str(x.message)]
    assert mine and all(x.filename == __file__ for x in mine), \
        [(x.filename, x.lineno) for x in mine]
