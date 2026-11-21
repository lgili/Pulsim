"""Integration 14m: the recorder's step sink and 128-row batches (probes
0b.10, DECISIONS 24) against what its branch never saw: the kernel's device
readings (`res.i` of a Level-1 MOSFET, batch 14l), `integrator='esdirk43'`,
and a run that crosses a batch boundary with events inside. A recorder only
observes: the run's arrays are the same bits with it on and off, and a
keep='every' row is the committed state of its step, each step exactly once.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p

_RUNS = {"pwl": dict(engine="pwl", dt=2e-6),
         "trbdf2": dict(engine="trbdf2", rtol=1e-7, atol=1e-10),
         "esdirk43": dict(engine="trbdf2", rtol=1e-7, atol=1e-10, integrator="esdirk43")}


def _sim(b, t_end, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end, **kw)


def _mosfet():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdd", "in", "gnd", 10.0)
    b.add_resistor("Rs", "in", "mid", 10.0)
    b.add_sine_voltage_source("Vg", "g", "gnd", 5.0, 5.0, 1e3)
    b.add_mosfet_level1("X", "mid", "gnd", "g", 2.0, 3.0, with_body_diode=False)
    return b


@pytest.mark.parametrize("probes", [["v(mid)"], ["v(mid)", "i(X)"]],
                         ids=["batched", "device-reading"])
@pytest.mark.parametrize("eng", ["pwl", "trbdf2"])
def test_mosfet_res_i_is_the_same_with_a_recorder(eng, probes):
    # ["v(mid)"] alone takes the batched feed on pwl; with i(X) the feed
    # takes the general path. res.i comes from the kernel law either way.
    r0 = _sim(_mosfet(), 2e-3, **_RUNS[eng])
    r1 = _sim(_mosfet(), 2e-3, **_RUNS[eng], recorders=[p.Recorder("r", probes)])
    assert r1.recorder_feed["mode"] == "in-loop", r1.recorder_feed
    np.testing.assert_array_equal(np.asarray(r1.times), np.asarray(r0.times))
    i0, i1 = np.asarray(r0.i("X")), np.asarray(r1.i("X"))
    assert np.max(np.abs(i0)) > 1e-3            # the device conducts
    np.testing.assert_array_equal(i1, i0)
    a = r1.recorded["r"]
    np.testing.assert_array_equal(a.times, np.asarray(r1.times))
    np.testing.assert_array_equal(a["v(mid)"], np.asarray(r1.v("mid")))
    if "i(X)" in probes:
        # two code paths (the loop's reading, the kernel's res.i)
        np.testing.assert_allclose(a["i(X)"][1:], i1[1:], rtol=1e-9,
                                   atol=1e-12 * np.max(np.abs(i1)))


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 24.0)
    b.add_switch("S1", "in", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "out", 100e-6)
    b.add_capacitor("C", "out", "gnd", 10e-6)
    b.add_resistor("R", "out", "gnd", 5.0)
    fn = p.make_pwm_switch_fn(20e3, 0.4, b.switch_index_of("S1"), b.graph.num_switches)
    return b, fn


@pytest.mark.parametrize("eng", sorted(_RUNS))
def test_rows_across_batch_boundaries_with_events(eng):
    # 1 ms of a 20 kHz buck: hundreds of rows (several 128-row batches on
    # pwl) with commutations inside the batches and at the boundaries. Every
    # accepted step is one row, in order, equal to the committed state.
    b0, fn0 = _buck()
    r0 = _sim(b0, 1e-3, switch_fn=fn0, **_RUNS[eng])
    b1, fn1 = _buck()
    r1 = _sim(b1, 1e-3, switch_fn=fn1, **_RUNS[eng],
              recorders=[p.Recorder("r", ["v(out)", "v(sw)"], keep="every")])
    feed = r1.recorder_feed
    assert feed["mode"] == "in-loop", feed
    assert feed["n_events"] >= 20, feed
    t = np.asarray(r1.times)
    assert t.size > 3 * 128
    np.testing.assert_array_equal(t, np.asarray(r0.times))
    np.testing.assert_array_equal(np.asarray(r1.states), np.asarray(r0.states))
    a = r1.recorded["r"]
    assert np.all(np.diff(a.times) > 0)
    np.testing.assert_array_equal(a.times, t)
    np.testing.assert_array_equal(a["v(out)"], np.asarray(r1.v("out")))
    np.testing.assert_array_equal(a["v(sw)"], np.asarray(r1.v("sw")))
