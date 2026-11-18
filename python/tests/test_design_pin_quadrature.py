"""p_in on the fixed-step engine (add-design-optimization-workflow
DECISIONS 29; probe probes/pin_quadrature.py).

A DCM buck on `steady=SteadySpec(method='pss')`, `engine='pwl'` at T/200
read p_in 1.0 % high: the sample at a gate edge holds the left limit,
and the source current, integrated straight across the edge, charged
half a step of the switch's peak current to the step after it. The
states are exact (p_out within 1e-7); the efficiency never read p_in.
Each side of the edge is now integrated separately."""

import pytest

import pulsim as p
from pulsim.design import Evaluator, SteadySpec

T = 1e-5
DESIGN = dict(L=20e-6, R=50.0)


def buck(d):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "x", d["L"])
    b.add_resistor("RLs", "x", "vout", 0.01)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", d["R"])
    return b, p.make_pwm_switch_fn(1 / T, 0.5, b.switch_index_of("S"),
                                   b.graph.num_switches)


@pytest.fixture(scope="module")
def exact():
    r = Evaluator(buck, period=T, loads=("RL",),
                  steady=SteadySpec(method="pss")).evaluate(DESIGN)
    assert r.counters["route"] == "pss/dsed"
    return r


def _pwl(n, method="pss"):
    return Evaluator(buck, period=T, loads=("RL",), engine="pwl",
                     options={"dt": T / n},
                     steady=SteadySpec(method=method)).evaluate(DESIGN)


def test_p_in_on_the_fixed_step_pss_route_is_second_order(exact):
    e = {}
    for n in (200, 800):
        r = _pwl(n)
        assert r.status == "ok" and r.counters["route"] == "pss/pwl"
        e[n] = abs(r.kpis["p_in"] / exact.kpis["p_in"] - 1)
        # the balance closes (it was 9.9e-3 at T/200)
        assert abs(r.counters["energy_residual"]) < 1e-6
    assert e[200] < 1e-6                     # measured 1.5e-7 (was 1.0e-2)
    assert e[200] / e[800] > 8               # measured 16: O(h^2)


def test_the_efficiency_never_read_p_in(exact):
    # the claim checked: efficiency is p_out / (p_out + losses), equal to
    # the exact route's within 1e-8 at T/200
    r = _pwl(200)
    assert abs(r.kpis["efficiency"] - exact.kpis["efficiency"]) < 1e-8


def test_the_transient_route_keeps_its_own_balance():
    # there the step ending at an edge ran with the new mask at its end
    # (the engine's first order, docs/known-limitations.md): the samples'
    # trapezoid is that step's energy; no correction, the balance closes
    r = _pwl(200, method="transient")
    assert r.status == "ok"
    assert abs(r.counters["energy_residual"]) < 1e-6
