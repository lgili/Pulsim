"""Task 6.2: a discrete Gate sink. A comparator ticking at 100 kHz drives a
buck switch through `MixedDomainBlockChain.gate`; the gate's edges are
the graph's publication instants, so trbdf2 lands every switch change on
a tick and pwl+split splits at them. Shooting refuses the gate (it
follows a controller)."""
from __future__ import annotations

import numpy as np
import pytest

import pulsim as p

T = 1e-5


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    b.add_switch("Q", "in", "sw", 1e2, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e2, 1e-9)
    b.add_inductor("L", "sw", "out", 1e-4)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", 10.0)
    return b


def _bang_bang(b):
    ch = p.MixedDomainBlockChain(rate=p.Rate(T))
    # HIGH below 12 V - 0.1, LOW above 12 V + 0.1: the comparator's output
    # is the inverse of "out too high"
    ch.add("err", p.Subtract(), inputs=dict(a=12.0, b="node:out"), output="e")
    ch.add("cmp", p.Comparator(threshold=0.0, hysteresis=0.2),
           inputs=dict(input_value="channel:e"), output="on")
    obs = ch.make_step_observer(b)
    return obs, ch.gate(b, {"on": "Q"})


def _changes(res, b):
    """(sample before, sample at) of each recorded change of Q."""
    bits = np.asarray(res.switch_states)[:, b.switch_index_of("Q")]
    t = np.asarray(res.times)
    k = np.flatnonzero(bits[1:] != bits[:-1]) + 1
    return t[k - 1], t[k]


def test_trbdf2_lands_every_gate_change_on_a_tick():
    b = _buck()
    obs, gate = _bang_bang(b)
    res = p.simulate(b, 2e-3, engine="trbdf2", step_observer=obs,
                     switch_fn=gate)
    before, at = _changes(res, b)
    assert len(at) > 20                   # it regulates: many edges
    # each change happens on a tick k*T: the step that records it starts
    # on that tick (trbdf2 records the new state at the step's end)
    k = np.floor(at / T + 1e-9)
    assert np.abs(before - k * T).max() <= 1e-12
    v = np.asarray(res.v("out"))
    tt = np.asarray(res.times)
    tail = v[tt > 1e-3]
    assert 11.0 < tail.mean() < 13.0


def test_pwl_split_and_trbdf2_agree_on_the_gate():
    out = {}
    for eng, kw in (("trbdf2", dict(rtol=1e-8)),
                    ("pwl", dict(dt=T / 20, discontinuity_restart=True,
                                 enable_substep_state_correction=True))):
        b = _buck()
        obs, gate = _bang_bang(b)
        res = p.simulate(b, 1e-3, engine=eng, step_observer=obs,
                         switch_fn=gate, **kw)
        out[eng] = float(np.asarray(res.v("out"))[-1])
    assert out["pwl"] == pytest.approx(out["trbdf2"], rel=1e-3)


def test_shooting_refuses_the_gate():
    b = _buck()
    obs, gate = _bang_bang(b)
    with pytest.raises(p.UnsupportedError):
        p.periodic_steady_state(b, period=T, switch_fn=gate, engine="pwl",
                                dt=T / 100)
    sf = p.make_combined_switch_fn(b.graph.num_switches, [gate])
    assert sf.has_edges and sf.follows_controller


def test_gate_needs_a_built_graph():
    ch = p.MixedDomainBlockChain(rate=p.Rate(T))
    with pytest.raises(ValueError, match="make_step_observer"):
        ch.gate(_buck(), {"on": "Q"})


def test_gate_refuses_an_unknown_channel():
    b = _buck()
    obs, _ = _bang_bang(b)
    ch = obs._graph
    with pytest.raises(ValueError, match="no block writes"):
        ch.gate(b, {"onn": "Q"})
