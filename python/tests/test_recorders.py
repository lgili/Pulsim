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
RC_BOUND = {"pwl": V * 1e-6 / TAU, "trbdf2": 2e-5 * V, "exact": 2e-5 * V}


@pytest.mark.parametrize("eng", sorted(RUNS))
def test_rc_charge_window_clipped_mid_step(eng):
    rec = p.Recorder("rc", ["v(out)", "v(in,out)/1e3"], windows=[(A, B)])
    res = p.simulate(_rc(), t_end=3e-3, recorders=[rec], **RUNS[eng])
    r = res.recorded["rc"]
    assert r.feeding == "stored-arrays"
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
    (dict(store_every=2, dt=1e-6, engine="pwl"), "store_every"),
    (dict(engine="pwl", dt=1e-6, resume_from=object()), "resume_from"),
])
def test_unhonourable_options_are_refused_by_name(kw, match):
    with pytest.raises(ValueError, match=match):
        p.simulate(_rc(), t_end=1e-3, recorders=[p.Recorder("x", ["v(out)"])],
                   **kw)


@pytest.mark.parametrize("rec,match", [
    (p.Recorder("x", ["i(C1)"]), r"i\(C1\) of a capacitor"),
    (p.Recorder("x", ["v(out)"], keep="times"), "times"),
    (p.Recorder("x", ["v(out)"], keep="times", times=(0.5e-3, 2e-3)),
     "inside"),
    (p.Recorder("x", ["v(out"]), "probe"),
    (p.Recorder("x", ["v(out)"], windows=[(0.5e-3, 2e-3)]), r"windows\[0\]"),
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
