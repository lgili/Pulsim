"""Kernel probes and recorders (add-probes-and-recorders) against
independent truths: RC closed forms over windows clipped mid-step, a
rectified sine with commutations inside the window, the floating-bus
bridge's mpmath average (benchmarks/tools/bridge_ground_truth.py), and
pulsim.time_average on the same samples."""
import math

import numpy as np
import pytest

import pulsim as p

from .test_bridge_ground_truth import TRUTH, _bridge


V, R, C = 10.0, 1e3, 1e-6
TAU = R * C
A, B = 0.37e-3 + 1.234e-9, 2.71e-3 + 4.321e-9   # mid-step on every grid


def _rc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "in", "gnd", V)
    b.add_resistor("R1", "in", "out", R)
    b.add_capacitor("C1", "out", "gnd", C)
    return b


def _rc_truth(a, b):
    ea, eb = math.exp(-a / TAU), math.exp(-b / TAU)
    avg = V - V * TAU * (ea - eb) / (b - a)
    ms = V * V * ((b - a) - 2 * TAU * (ea - eb)
                  + 0.5 * TAU * (ea * ea - eb * eb)) / (b - a)
    return avg, math.sqrt(ms)


RUNS = {
    "pwl": dict(engine="pwl", dt=1e-6),
    "trbdf2": dict(engine="trbdf2", rtol=1e-8, atol=1e-11),
    "exact": dict(engine="exact"),
}
# The run's own accuracy on the RC charge. pwl starts first order from the
# inconsistent t = 0 state (owner note fixed-engine-first-order-start): its
# first step lags by V*dt/(2*tau), and that offset decays with tau, so the
# window average is off by ~1.3e-3 V at dt = 1e-6 (bound V*dt/tau = 1e-2).
# pwl / trbdf2 run through the kernel Simulation object, whose step hooks
# feed the recorders while the engine runs; the exact engine runs this
# diode-free RC on integrator 'auto', which has no step hooks: stored arrays.
FEEDING = {"pwl": "in-loop", "trbdf2": "in-loop", "exact": "stored-arrays"}
# a circuit with diodes routes the exact engine to its event-driven
# integrator, which feeds the recorders in the loop too (session 26); the
# diode-free RC above runs on 'auto' (no step hooks: the stored arrays)
FEEDING_DIODES = dict(FEEDING, exact="in-loop")
RC_BOUND = {"pwl": V * 1e-6 / TAU, "trbdf2": 2e-5 * V, "exact": 2e-5 * V}


@pytest.mark.parametrize("eng", sorted(RUNS))
def test_rc_charge_window_clipped_mid_step(eng):
    rec = p.Recorder("rc", ["v(out)", "v(in,out)/1e3"], windows=[(A, B)])
    res = p.simulate(_rc(), t_end=3e-3, recorders=[rec], **RUNS[eng])
    r = res.recorded["rc"]
    assert r.feeding == FEEDING[eng]
    avg, rms = _rc_truth(A, B)
    w = r.windows["v(out)"]
    bound = RC_BOUND[eng]
    assert abs(w["avg"][0] - avg) <= bound, (w["avg"][0], avg)
    assert abs(w["rms"][0] - rms) <= bound, (w["rms"][0], rms)
    # the resistor current: (V - v) / R, average = (V - avg) / R
    wi = r.windows["v(in,out)/1e3"]
    assert abs(wi["avg"][0] - (V - avg) / R) <= bound / R
    print(f"RC {eng}: avg err {w['avg'][0] - avg:+.3e} V, "
          f"rms err {w['rms'][0] - rms:+.3e} V")
    # pulsim.time_average / time_rms on the same samples: round-off
    t, v = np.asarray(res.times), np.asarray(res.v("out"))
    assert w["avg"][0] == pytest.approx(p.time_average(t, v, A, B), rel=1e-9)  # kernel vs numpy: two summation paths
    assert w["rms"][0] == pytest.approx(p.time_rms(t, v, A, B), rel=1e-9)  # kernel vs numpy: two summation paths
    # monotone charge: min at the clipped start, max at the clipped end
    assert w["t_min"][0] == A and w["t_max"][0] == B
    assert w["min"][0] == pytest.approx(np.interp(A, t, v), rel=1e-9)  # kernel vs numpy: two summation paths


def _rectifier():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "s", "gnd", 0.0, 100.0, 50.0)
    for nm, a, c in (("D1", "s", "dp"), ("D2", "gnd", "dp"),
                     ("D3", "dn", "s"), ("D4", "dn", "gnd")):
        b.add_diode(nm, a, c, 1e3, 1e-9)
    b.add_resistor("RL", "dp", "dn", 100.0)
    b.add_resistor("Rtie", "dn", "gnd", 1e6)
    return b


@pytest.mark.parametrize("eng", sorted(RUNS))
def test_rectified_sine_with_commutations_inside_the_window(eng):
    # the window holds the 10, 20 and 30 ms zero crossings (4 diodes
    # commute at each) and starts and ends mid-half-wave
    a, b = 7.3e-3, 33.1e-3
    rec = p.Recorder("bus", ["v(dp,dn)", "i(D1)"], windows=[(a, b)],
                     period=10e-3, offset=5e-3)
    res = p.simulate(_rectifier(), t_end=40e-3, recorders=[rec],
                     **RUNS[eng])
    r = res.recorded["bus"]
    w = 2 * math.pi * 50.0
    k = 100.0 / (100.0 + 2e-3)       # two conducting diodes of 1 mOhm

    def F(t):   # integral of |sin(wt)| from 0
        n = math.floor(t * 100.0)
        return (2 * n + (1 - math.cos(w * t - n * math.pi))) / w

    truth = 100.0 * k * (F(b) - F(a)) / (b - a)
    got = r.windows["v(dp,dn)"]["avg"][0]
    # the diodes' dead band (|v_s| below the g_off/g_on knee) is ~1e-9 of
    # the window; the rest is the engine's own accuracy
    assert abs(got - truth) <= 5e-3, (got, truth)
    t = np.asarray(res.times)
    vb = np.asarray(res.v("dp")) - np.asarray(res.v("dn"))
    assert got == pytest.approx(p.time_average(t, vb, a, b), rel=1e-9)   # two summation paths
    # three complete half-periods [5+10k, 15+10k] ms of 2/pi * peak
    per = r.periods["v(dp,dn)"]
    assert per["avg"].size == 3
    assert np.allclose(per["avg"], 2 / math.pi * 100.0 * k, atol=5e-3)
    assert np.allclose(per["max"], 100.0 * k, atol=5e-3)


def test_both_limits_at_the_exact_engines_events():
    rec = p.Recorder("bus", ["v(dp,dn)", "v(s)"], keep="none")
    res = p.simulate(_rectifier(), t_end=40e-3, engine="exact",
                     recorders=[rec])
    r = res.recorded["bus"]
    idx = np.asarray(res.right_limit_index)
    assert idx.size > 0 and r.event_t.size == idx.size
    t = np.asarray(res.times)
    np.testing.assert_array_equal(r.event_t, t[idx])
    vb = np.asarray(res.v("dp")) - np.asarray(res.v("dn"))
    np.testing.assert_array_equal(r.event_left["v(dp,dn)"], vb[idx])
    rl = np.asarray(res.right_limit_states)
    jdp, jdn = (res._builder.node_id_of(n) for n in ("dp", "dn"))
    np.testing.assert_array_equal(r.event_right["v(dp,dn)"],
                                  rl[:, jdp] - rl[:, jdn])
    assert r.times.size == 0


def test_bridge_bus_average_against_the_mpmath_truth():
    """30-40 ms bus average, window clipped exactly, on every engine;
    printed as the validation table."""
    rec = p.Recorder("bus", ["v(dp) - v(dn)"], keep="none",
                     windows=[(30e-3, 40e-3)])
    runs = {"pwl dt=1e-6": (dict(engine="pwl", dt=1e-6), 1e-5),
            "trbdf2 rtol=1e-7": (dict(engine="trbdf2", rtol=1e-7,
                                      atol=1e-10), 1e-3),
            "exact": (dict(engine="exact"), 1e-6)}
    rows = []
    for label, (kw, bound) in runs.items():
        res = p.simulate(_bridge(np.zeros(6), "conductance"), t_end=40e-3,
                         recorders=[rec], **kw)
        got = res.recorded["bus"].windows["v(dp) - v(dn)"]["avg"][0]
        t = np.asarray(res.times)
        vb = np.asarray(res.v("dp")) - np.asarray(res.v("dn"))
        ta = p.time_average(t, vb, 30e-3, 40e-3)
        rows.append((label, got, got - TRUTH, got - ta))
        assert abs(got - TRUTH) <= bound, (label, got)
        assert got == pytest.approx(ta, rel=1e-9)   # two summation paths
    for row in rows:
        print("%-18s %.12f  err %+.3e  vs time_average %+.1e" % row)


def test_decimation_keeps_rows_not_accuracy():
    full = p.Recorder("f", ["v(out)"], windows=[(A, B)])
    dec = p.Recorder("d", ["v(out)"], keep="decimate", every=7,
                     windows=[(A, B)], period=1e-3, offset=0.5e-3)
    res = p.simulate(_rc(), t_end=3e-3, engine="trbdf2", recorders=[full, dec])
    f, d = res.recorded["f"], res.recorded["d"]
    assert d.times.size == (f.times.size + 6) // 7
    np.testing.assert_array_equal(d.times, f.times[::7])
    np.testing.assert_array_equal(d["v(out)"], f["v(out)"][::7])
    assert d.windows["v(out)"]["avg"][0] == f.windows["v(out)"]["avg"][0]
    assert d.periods["v(out)"]["avg"].size == 2      # [0.5, 1.5], [1.5, 2.5] ms


@pytest.mark.parametrize("kw,match", [
    (dict(engine="pwl", dt=1e-6, resume_from=object()), "resume_from"),
])
def test_unhonourable_options_are_refused_by_name(kw, match):
    with pytest.raises(ValueError, match=match):
        p.simulate(_rc(), t_end=1e-3, recorders=[p.Recorder("x", ["v(out)"])],
                   **kw)


@pytest.mark.parametrize("rec,match", [
    (p.Recorder("x", ["v(out)"], keep="times"), "times"),
    (p.Recorder("x", ["v(out)"], keep="times", times=(0.5e-3, 2e-3)),
     "inside"),
    (p.Recorder("x", ["v(out"]), "probe"),
    (p.Recorder("x", ["v(out)"], windows=[(1.5e-3, 2e-3)]), r"windows\[0\]"),
    (p.Recorder("x", ["v(out)"], keep="some"), "keep"),
    (p.Recorder("x", ["v(out)"], period=-1.0), "period"),
    (p.Recorder("x", ["v(out)"], period=1e-4, offset=float("nan")),
     "offset"),
])
def test_bad_recorders_are_refused_before_the_run(rec, match):
    with pytest.raises(ValueError, match=match):
        p.simulate(_rc(), t_end=1e-3, dt=1e-6, engine="pwl", recorders=[rec])


def test_unknown_node_names_the_reading():
    with pytest.raises(KeyError, match="nope"):
        p.simulate(_rc(), t_end=1e-3, dt=1e-6, engine="pwl",
                   recorders=[p.Recorder("x", ["v(nope)"])])


# ---- in-loop feeding (task 0b.6) and event semantics (item 2) ----------

def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_mosfet("Q1", "vin", "sw")
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 10.0)
    fn = p.make_pwm_switch_fn(100e3, 0.5, b.switch_index_of("Q1"),
                              b.graph.num_switches)
    return b, fn


_STATS = ("integral", "avg", "rms", "min", "t_min", "max", "t_max")
_CASES = {
    "rc": (_rc, 3e-3, ["v(out)", "v(in,out)/1e3", "i(R1)", "p(R1)", "i(V1)"],
           [(A, B)], None, 0.0),
    "rect": (_rectifier, 40e-3, ["v(dp,dn)", "v(s)", "i(RL)"],
             [(7.3e-3, 33.1e-3)], 10e-3, 5e-3),
    "bridge": (lambda: _bridge(np.zeros(6), "conductance"), 40e-3,
               ["v(dp) - v(dn)", "i(Ls)"], [(30e-3, 40e-3)], None, 0.0),
    # PWM gate edges: trbdf2 lands them and commits the post-edge probe
    "buck": (_buck, 1e-3, ["v(vout)", "i(L)", "p(RL)"],
             [(0.2e-3 + 3.3e-9, 0.9e-3 - 1.7e-9)], 1e-5, 0.0),
}
_LOOP_RUNS = {"pwl": dict(engine="pwl", dt=1e-6),
              "trbdf2": dict(engine="trbdf2", rtol=1e-7, atol=1e-10)}


@pytest.mark.parametrize("case", sorted(_CASES))
@pytest.mark.parametrize("eng", sorted(_LOOP_RUNS))
def test_in_loop_feeding_equals_the_stored_array_pass(eng, case):
    """The same run read two ways: the hooks while it ran, and the pass over
    its stored arrays. On pwl the samples are the same (bit-identical on
    this machine; 1e-9 across platforms). trbdf2 stores the settled
    post-event probe as an extra sample where the in-loop record starts the
    next line from the same settled state as an event row: the rectifier's
    windows differ there by ~5e-10 relative."""
    from pulsim._recorders import record_result
    fac, te, probes, win, per, off = _CASES[case]
    rec = p.Recorder("r", probes, windows=win, period=per, offset=off)
    built = fac()
    b0, fn = built if isinstance(built, tuple) else (built, None)
    res = p.simulate(b0, t_end=te, recorders=[rec], switch_fn=fn,
                     **_LOOP_RUNS[eng])
    a = res.recorded["r"]
    b = record_result(res, [rec])["r"]
    assert a.feeding == "in-loop" and b.feeding == "stored-arrays"
    assert res.recorder_feed["mode"] == "in-loop"
    assert res.recorder_feed["n_dropped"] == 0
    # trbdf2 stores its settled post-event probe as a sample: the line from
    # the left limit to it (stored) vs from the settled state (in-loop)
    rtol = 1e-6 if eng == "trbdf2" and case in ("rect", "buck") else 1e-9
    for q in probes:
        for st in (lambda r: r.windows[q], lambda r: r.periods[q]):
            sa, sb = st(a), st(b)
            if sb["rms"].size == 0:
                continue
            # a scale floor: a sine's period integral is physically zero
            scale = float(np.max(sb["rms"]))
            span = float(np.max(sb["t1"] - sb["t0"]))
            # min / max (and their times) are limit-aware: a period that
            # starts at a settled event starts on its right limit in-loop,
            # while the stored pass has no event row there (the DCM buck's
            # inductor current: -1.46e-8 A left, -5.5e-11 A right at 0.24 ms)
            fields = _STATS if rtol == 1e-9 else ("integral", "avg", "rms")
            for f in fields:
                atol = rtol * scale * (span if f == "integral" else 1.0)
                if f.startswith("t_"):
                    atol = rtol * span
                np.testing.assert_allclose(sa[f], sb[f], rtol=rtol,
                                           atol=atol, err_msg=f"{q} {f}")
    if eng == "pwl" or case not in ("rect", "buck"):
        np.testing.assert_array_equal(a.times, b.times)
        for q in probes:
            np.testing.assert_allclose(a[q], b[q], rtol=1e-9, atol=1e-12)


def test_store_every_gives_the_full_runs_statistics():
    """A decimated run's recorders see every accepted step: the windows,
    periods and event rows of store_every=50 equal those of the full run
    (the same step sequence), and a statistics-only recorder keeps no row."""
    mk = lambda: p.Recorder("s", ["v(dp,dn)", "i(RL)"], keep="none",  # noqa: E731
                            windows=[(7.3e-3, 33.1e-3)], period=10e-3,
                            offset=5e-3)
    full = p.simulate(_rectifier(), t_end=40e-3, engine="pwl", dt=1e-6,
                      recorders=[mk()])
    dec = p.simulate(_rectifier(), t_end=40e-3, engine="pwl", dt=1e-6,
                     store_every=50, recorders=[mk()])
    assert len(full.times) == 40001 and len(dec.times) == 801
    np.testing.assert_array_equal(full.times[::50], dec.times)
    a, b = full.recorded["s"], dec.recorded["s"]
    assert b.feeding == "in-loop" and b.times.size == 0
    for q in a.probes:
        for f in _STATS:
            np.testing.assert_array_equal(a.windows[q][f], b.windows[q][f])
            np.testing.assert_array_equal(a.periods[q][f], b.periods[q][f])
        np.testing.assert_array_equal(a.event_right[q], b.event_right[q])
    np.testing.assert_array_equal(a.event_t, b.event_t)
    assert a.periods["v(dp,dn)"]["avg"].size == 3


def test_store_every_without_in_loop_feeding_is_refused_by_name():
    # a VCCS has no in-loop current law (the stored pass reads it)
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "in", "gnd", 1.0)
    b.add_resistor("R1", "in", "gnd", 10.0)
    b.add_vccs("G1", "in", "gnd", "a", "gnd", 0.1)
    b.add_resistor("R2", "a", "gnd", 10.0)
    with pytest.raises(ValueError, match=r"store_every=2.*could not be fed in the "
                       r"loop.*i\(G1\).*no in-loop current law"):
        p.simulate(b, t_end=1e-5, dt=1e-6, engine="pwl", store_every=2,
                   recorders=[p.Recorder("x", ["i(G1)"])])


# the 11 commutations' (i_L, v_C) at the event, from
# benchmarks/tools/bridge_ground_truth.py run(params(), dps=30)
_TRUTH_STATES = [
    (3.329968943405992e-07, 332.9968976609098),
    (-2.9589476677125644e-07, 295.89476677125634),
    (-0.0002765293634114608, 276.2531103011596),
    (-0.0003444643908292179, 344.12027056718364),
    (-3.1633577693666694e-07, 316.33577693675653),
    (2.931086676394856e-07, 293.10866763948553),
    (3.3529305742994196e-07, 335.2930755044125),
    (-3.0898050256049306e-07, 308.98050256049294),
    (-0.0002874449184521423, 287.1577606914508),
    (-0.0003389918909732532, 338.6532377471501),
    (-3.118711722901159e-07, 311.8711722901191),
]


@pytest.mark.parametrize("eng", ["pwl", "trbdf2", "exact"])
def test_event_rows_against_the_bridge_truth(eng):
    """Both limits where the engine has them, flagged: 'exact' (the exact
    engine's right limits), 'settled' (trbdf2's post-event state at the
    instant), 'within_dt' (a bracket: the left limit at event_t, the end of
    the first step under the new mode at event_t_right; pwl always, trbdf2
    where its settle did not report). Never a fabricated right limit."""
    from .test_bridge_ground_truth import EVENTS
    kw = dict(_LOOP_RUNS, exact=dict(engine="exact"))[eng]
    rec = p.Recorder("e", ["i(Ls)", "v(dp,dn)"], keep="none")
    res = p.simulate(_bridge(np.zeros(6), "conductance"), t_end=40e-3,
                     recorders=[rec], **kw)
    r = res.recorded["e"]
    res_set = set(r.event_resolution.tolist())
    assert res_set <= {"exact": {"exact"}, "pwl": {"within_dt"},
                       "trbdf2": {"settled", "within_dt"}}[eng], res_set
    assert np.all(r.event_t_right >= r.event_t)
    sett = r.event_resolution != "within_dt"
    assert np.all(r.event_t_right[sett] == r.event_t[sett])
    found = 0
    for (t, _), (iL, vC) in zip(EVENTS, _TRUTH_STATES):
        if eng == "exact":
            k = int(np.argmin(np.abs(r.event_t - t)))
            assert abs(r.event_t[k] - t) <= 1e-10
            for side in (r.event_left, r.event_right):
                assert abs(side["v(dp,dn)"][k] - vC) <= 1e-6
            assert abs(r.event_left["i(Ls)"][k] - iL) <= 1e-9
            found += 1
        elif eng == "pwl":
            # a bracket holds the truth (the stored engine's grid is 1 us)
            hit = (r.event_t <= t + 1e-9) & (t <= r.event_t_right + 1e-9)
            assert hit.any(), t
            k = int(np.nonzero(hit)[0][0])
            assert abs(r.event_left["v(dp,dn)"][k] - vC) <= 1e-2
            assert abs(r.event_right["v(dp,dn)"][k] - vC) <= 1e-2
            found += 1
        else:
            # trbdf2 lands every commutation within ~30 ns here; the left
            # limit is the landing, the right one settled or a bracket
            k = int(np.argmin(np.abs(r.event_t - t)))
            assert abs(r.event_t[k] - t) <= 2e-7, (t, r.event_t[k])
            assert abs(r.event_left["v(dp,dn)"][k] - vC) <= 1e-2
            if r.event_resolution[k] == "settled":
                assert abs(r.event_right["v(dp,dn)"][k] - vC) <= 1e-2
            found += 1
    assert found == 11, found


@pytest.mark.parametrize("eng", sorted(RUNS))
def test_stream_delivers_the_recorders_kept_rows(eng):
    # task 8.3: Recorder(stream=NativeLiveStream) pushes each kept row into
    # the stream's ring; the consumer reads the rows the recording holds
    rec_probes = ["v(dp,dn)", "i(D1)"]
    st = p.NativeLiveStream(capacity=200_000, decimate=1)
    rec = p.Recorder("bus", rec_probes, keep="every", stream=st)
    res = p.simulate(_rectifier(), t_end=20e-3, recorders=[rec], **RUNS[eng])
    r = res.recorded["bus"]
    assert r.feeding == FEEDING_DIODES[eng]
    assert st.channel_names == rec_probes
    got = st.get_new_samples()
    assert got is not None
    t_s, x_s = got
    assert r.times.size > 10
    np.testing.assert_array_equal(t_s, r.times)
    for j, pr in enumerate(rec_probes):
        np.testing.assert_array_equal(x_s[:, j], r.values[pr])
    # a second run re-attaches: the consumer reads that run's rows only
    res2 = p.simulate(_rectifier(), t_end=10e-3, recorders=[rec], **RUNS[eng])
    t2, _ = st.get_new_samples()
    np.testing.assert_array_equal(t2, res2.recorded["bus"].times)
