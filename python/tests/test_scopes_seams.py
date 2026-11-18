"""Integration batch 12, seam F: a circuit whose power stage sits in a
subsystem scope runs bit for bit as its hand-flattened twin on every
engine, with a C block, after a YAML round trip and in the Evaluator."""

import warnings

import numpy as np
import pytest

import pulsim as p
from pulsim.design import Evaluator, SteadySpec

T = 1e-5


def _filter_def():
    f = p.define_subsystem("LCFilter", ports=("sw", "out"), params={"L": 100e-6})

    @f.body
    def _(s):
        s.add_inductor("L", "sw", "out", s.p.L)
        s.add_capacitor("C", "out", "gnd", 47e-6)
        s.add_resistor("RL", "out", "gnd", 10.0)
    return f


def _root(b):
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)


def scoped():
    b = p.CircuitBuilder()
    _root(b)
    _filter_def().instantiate(b, "pwr", sw="sw", out="out")
    return b


def flat():
    b = p.CircuitBuilder()
    _root(b)
    b.add_inductor("pwr/L", "sw", "out", 100e-6)
    b.add_capacitor("pwr/C", "out", "gnd", 47e-6)
    b.add_resistor("pwr/RL", "out", "gnd", 10.0)
    return b


def _pwm(c):
    return p.make_pwm_switch_fn(1 / T, 0.5, 0, c.num_switches)


def _run(c, engine, **kw):
    kw.setdefault("switch_fn", _pwm(c))
    if engine == "pwl":
        kw.setdefault("dt", 1e-7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(c, 1e-4, engine=engine, **kw)


def _same(a, b):
    np.testing.assert_array_equal(np.asarray(a.times), np.asarray(b.times))
    np.testing.assert_array_equal(np.asarray(a.states), np.asarray(b.states))


@pytest.mark.parametrize("engine", ("pwl", "trbdf2", "exact"))
def test_scoped_power_stage_is_its_flat_twin(engine):
    s, f = scoped().build(), flat().build()
    assert s.nodes == f.nodes and s.components() == f.components()
    _same(_run(s, engine), _run(f, engine))


@pytest.mark.parametrize("engine", ("pwl", "trbdf2"))
def test_scoped_circuit_after_a_yaml_round_trip_is_the_twin(engine):
    s = p.Circuit.from_yaml(scoped().build().to_yaml())
    assert s.scopes()[0]["path"] == "pwr"
    _same(_run(s, engine), _run(flat().build(), engine))


def _with_c_block(b):
    def step(t, dt, inp, out, state):
        out[0] = 0.5 * inp[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        p.add_c_block(b, inputs=[("v", "out")], outputs=[("v", "mon", "gnd")],
                      dt=T, fn=step)
    b.add_resistor("Rmon", "mon", "gnd", 1e3)
    return b


@pytest.mark.parametrize("engine", ("pwl", "trbdf2"))
def test_scoped_circuit_with_a_c_block_is_the_twin(engine):
    s, f = _with_c_block(scoped()).build(), _with_c_block(flat()).build()
    rs, rf = _run(s, engine), _run(f, engine)
    _same(rs, rf)
    assert np.max(np.abs(np.asarray(rs.v("mon")))) > 0.1   # the block ran


def test_scoped_circuit_in_the_evaluator_gives_the_twin_records():
    def ev(c, field):
        return Evaluator(circuit=c, fields={"L": field}, switch_fn=_pwm(c),
                         period=T, loads=("pwr/RL",), output="out",
                         engine="pwl", options={"dt": T / 100},
                         steady=SteadySpec(method="transient"))
    rs = ev(scoped().build(), "pwr/L.L").evaluate({"L": 220e-6})
    rf = ev(flat().build(), "pwr/L.L").evaluate({"L": 220e-6})
    assert rs.status == rf.status
    assert rs.kpis == rf.kpis and rs.losses == rf.losses
