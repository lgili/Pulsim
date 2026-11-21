"""A recorder (fed in the loop through the step hooks) and a loss port (the
in-kernel loss tracker) on one run: each is an observer of the other.

The recorded values with a loss port equal those of the recorder alone, and
the losses with a recorder equal those of the loss port alone (same binary,
same path: bit-identical on this machine). Loss data here is synthetic.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p

from .test_device_losses_variable_step import T_END, _rl_chopper

_RUNS = {"pwl": dict(engine="pwl", dt=1e-7),
         "trbdf2": dict(engine="trbdf2", rtol=1e-6, atol=1e-9)}
_PROBES = ["v(out)", "i(L)", "p(R)"]


def _rec():
    return p.Recorder("r", _PROBES, windows=[(0.2e-3, 0.9e-3)], period=50e-6)


def _run(losses, recorder, eng, thermal=None):
    b, fn = _rl_chopper(losses=losses, thermal=thermal)
    kw = dict(recorders=[_rec()]) if recorder else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, T_END, switch_fn=fn, **_RUNS[eng], **kw)


def _same_record(a, b):
    np.testing.assert_array_equal(a.times, b.times)
    for q in _PROBES:
        np.testing.assert_array_equal(a[q], b[q])
        for sa, sb in ((a.windows, b.windows), (a.periods, b.periods)):
            assert sa[q]["rms"].size > 0, q     # a statistic was really taken
            assert sorted(sa[q]) == sorted(sb[q])
            for f, v in sa[q].items():
                np.testing.assert_array_equal(v, sb[q][f], err_msg=f"{q} {f}")


def _same_losses(a, b):
    for n in ("S1", "D1"):
        assert a.device_energy(n) == b.device_energy(n), n
        assert a.loss_events(n).tobytes() == b.loss_events(n).tobytes(), n
    assert a.heat_injected == b.heat_injected
    assert a.heat_pending == b.heat_pending


@pytest.mark.parametrize("eng", sorted(_RUNS))
def test_recorder_and_loss_port_observe_each_other_not(eng):
    rec_only = _run(False, True, eng)
    port_only = _run(True, False, eng)
    both = _run(True, True, eng)
    assert both.recorder_feed["mode"] == "in-loop"
    assert both.recorder_feed["n_dropped"] == 0
    _same_record(both.recorded["r"], rec_only.recorded["r"])
    _same_losses(both, port_only)
    e = both.device_energy("S1")
    assert e["E_on"] > 0.0 and e["E_off"] > 0.0 and e["E_cond_electrical"] > 0.0


@pytest.mark.parametrize("eng", sorted(_RUNS))
def test_a_recorder_leaves_a_heat_injecting_port_untouched(eng):
    """With a junction the port feeds heat into the circuit: the recorder
    still changes nothing (states, losses, heat)."""
    port_only = _run(True, False, eng, thermal=(5.0, 1e-4))
    both = _run(True, True, eng, thermal=(5.0, 1e-4))
    assert both.recorder_feed["mode"] == "in-loop"
    assert both.recorder_feed["n_dropped"] == 0
    assert np.asarray(both.times).tobytes() == np.asarray(port_only.times).tobytes()
    assert np.asarray(both.states).tobytes() == np.asarray(port_only.states).tobytes()
    _same_losses(both, port_only)
    assert both.heat_injected > 0.0


@pytest.mark.parametrize("eng", ["pwl", "trbdf2", "esdirk43"])
def test_a_recorder_reads_the_junction_it_heats(eng):
    """The in-loop recorder (the batched step feed) probing the junction
    node itself: the T_j trace is the same with the recorder on or off,
    and every recorded sample is a value of that trace at its time
    (add-in-kernel-event-losses DECISIONS 17 puts the heat's right limit
    in the thermal capacitor's history: the recorder must see committed
    states only)."""
    run = dict(_RUNS.get(eng, _RUNS["trbdf2"]))
    if eng == "esdirk43":
        run["integrator"] = "esdirk43"
    b0, fn0 = _rl_chopper(losses=True, thermal=(5.0, 1e-4))
    b1, fn1 = _rl_chopper(losses=True, thermal=(5.0, 1e-4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        off = p.simulate(b0, T_END, switch_fn=fn0, **run)
        on = p.simulate(b1, T_END, switch_fn=fn1, **run,
                        recorders=[p.Recorder("tj", ["v(tj)"], keep="every")])
    assert on.recorder_feed["mode"] == "in-loop"
    assert on.recorder_feed["n_dropped"] == 0
    assert np.asarray(on.times).tobytes() == np.asarray(off.times).tobytes()
    tj = np.asarray(off.v("tj"))
    assert np.asarray(on.v("tj")).tobytes() == tj.tobytes()
    assert tj.max() > 26.0                     # the junction really heats
    seen = {}
    for t, v in zip(np.asarray(off.times), tj):
        seen.setdefault(float(t), set()).add(float(v))
    rec = on.recorded["tj"]
    rt, rv = np.asarray(rec.times), np.asarray(rec["v(tj)"])
    assert rt.size > 100
    for t, v in zip(rt, rv):
        assert float(v) in seen.get(float(t), ()), (eng, t, v)
