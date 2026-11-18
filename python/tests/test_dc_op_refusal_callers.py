"""The DC operating point's refusal of a source-inductor loop reaches every
caller merged since the TR-BDF2 track branched (integration batch 9): the
circuit IR, a design evaluation (a ``refused`` record with the four fields,
never ``failed`` and never a crash), and the control graph's C++ chain path
of ``simulate(start_from_dc_op=True)``."""

import pytest

import pulsim as p
from pulsim.design import Evaluator, SteadySpec

T = 1e-5


def _loop(_design=None):
    # 48 V across two series inductors (ll16's Vin / Lleak / Lm): no DC point exists
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 48.0)
    b.add_inductor("L1", "vin", "d", 2e-6)
    b.add_inductor("L2", "d", "gnd", 158e-6)
    b.add_resistor("R", "d", "gnd", 10.0)
    return b


def test_the_circuit_ir_is_refused():
    with pytest.raises(p.UnsupportedError) as ei:
        p.compute_dc_op(_loop().build())
    assert (ei.value.feature, ei.value.device) == ("dc_operating_point", "V")


def test_a_design_whose_dc_start_is_refused_is_a_refused_record():
    ev = Evaluator(_loop, period=T, loads=("R",), output="d", engine="pwl",
                   options={"dt": T / 100, "start_from_dc_op": True},
                   steady=SteadySpec(method="transient"))
    rec = ev.evaluate({})
    assert rec.status == "refused"
    assert rec.refusal[1:3] == ("dc_operating_point", "V")
    assert rec.refusal[0] == "pwl"


def test_the_control_graph_chain_path_lets_the_refusal_through():
    b = _loop()
    ch = p.MixedDomainBlockChain(rate=p.Rate(1e-4))
    ch.add("g", p.Gain(k=0.5), inputs=dict(x="d"), output="y")
    obs = ch.make_step_observer(b)
    assert getattr(obs, "_cxx_chain", None) is not None      # the C++ chain path
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(b, 1e-4, engine="pwl", dt=1e-6, step_observer=obs,
                   start_from_dc_op=True)
    assert ei.value.feature == "dc_operating_point"
