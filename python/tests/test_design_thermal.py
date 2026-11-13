"""`pulsim.design`: the junction-temperature fixed point (design D4,
tasks 3.1-3.2 of add-design-optimization-workflow).

Oracle: with a loss linear in temperature, ``P = P0 (1 + a (T - 25))``,
and ``T = T_amb + R_th P``, the fixed point is
``T* = (T_amb + R_th P0 (1 - 25 a)) / (1 - R_th P0 a)``, which exists
only for ``R_th P0 a < 1``.
"""

import pytest

import pulsim as p
from pulsim.design import Evaluator, SteadySpec, ThermalSpec, as_objectives

from .test_design_evaluation import CCM, T, buck

P0, ALPHA, T_AMB = 10.0, 0.004, 40.0


def linear_loss(device, p_sim, t_j):
    return P0 * (1.0 + ALPHA * (t_j - 25.0))


def _ev(r_th, **kw):
    return Evaluator(buck, period=T, loads=("RL",), output="vout",
                     steady=SteadySpec(method="pss"), thermal=ThermalSpec(
                         r_th={"S": r_th}, t_amb=T_AMB,
                         loss_at=linear_loss, **kw))


def test_fixed_point_equals_the_closed_form():
    r_th = 2.0
    t_star = (T_AMB + r_th * P0 * (1 - 25 * ALPHA)) / (1 - r_th * P0 * ALPHA)
    assert t_star == pytest.approx(63.043478, abs=1e-6)
    rec = _ev(r_th).evaluate(CCM)
    assert rec.status == "ok", rec.reason
    assert rec.t_j["S"] == pytest.approx(t_star, abs=0.01)
    assert rec.kpis["t_j.S"] == rec.t_j["S"]
    assert rec.counters["thermal_increment"] < 0.1
    assert 2 <= rec.counters["thermal_iterations"] <= 20
    # the record's loss is the loss AT the junction temperature
    assert sum(rec.losses["S"].values()) == pytest.approx(
        linear_loss("S", None, rec.t_j["S"]), rel=1e-12)


def test_runaway_is_infeasible():
    r_th = 1.2 / (P0 * ALPHA)            # R_th P0 a = 1.2: no fixed point
    rec = _ev(r_th).evaluate(CCM)
    assert rec.status == "infeasible"
    assert rec.violations["thermal_runaway"] > 0
    assert rec.reason.startswith("thermal_runaway")
    assert rec.kpis == {}                 # no number for an unsettled T_j
    assert list(as_objectives(rec, ["efficiency"], penalty=3.0)) == [3.0]


def test_runaway_without_crossing_the_limit_is_caught_by_the_budget():
    rec = _ev(1.2 / (P0 * ALPHA), t_limit=1e9, max_iter=5).evaluate(CCM)
    assert rec.status == "infeasible"
    assert rec.counters["thermal_iterations"] == 5
    assert "fixed point" in rec.reason


def test_no_temperature_dependence_means_one_simulation():
    plain = Evaluator(buck, period=T, loads=("RL",),
                      steady=SteadySpec(method="pss")).evaluate(CCM)
    rec = _ev(2.0).evaluate(CCM)
    assert rec.counters["thermal_iterations"] >= 2
    assert rec.counters["simulations"] == plain.counters["simulations"]


def hot_buck(d):
    """The switch's on-resistance follows its junction temperature."""
    tj = d.get("_T_j", {}).get("S", 25.0)
    r_on = 0.05 * (1.0 + 0.006 * (tj - 25.0))
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", 1.0 / r_on, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "x", 100e-6)
    b.add_resistor("RLs", "x", "vout", 0.01)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 1.0)
    return b, p.make_pwm_switch_fn(1 / T, 0.5, b.switch_index_of("S"),
                                   b.graph.num_switches)


def test_temperature_dependent_circuit_is_resimulated_to_its_fixed_point():
    r_th = 10.0
    ev = Evaluator(hot_buck, period=T, loads=("RL",),
                   steady=SteadySpec(method="pss"), thermal=ThermalSpec(
                       r_th={"S": r_th}, t_amb=T_AMB,
                       temperature_dependent=True))
    rec = ev.evaluate({})
    assert rec.status == "ok", rec.reason
    p_s = sum(rec.losses["S"].values())
    # T_j = T_amb + R_th P(T_j), to the loop's 0.1 K
    assert rec.t_j["S"] == pytest.approx(T_AMB + r_th * p_s, abs=0.1)
    one = Evaluator(hot_buck, period=T, loads=("RL",),
                    steady=SteadySpec(method="pss")).evaluate({})
    assert rec.counters["simulations"] > one.counters["simulations"]
    # hotter switch, more loss than at 25 C
    assert p_s > sum(one.losses["S"].values()) * 1.1
    assert "_T_j" not in rec.design


def test_unknown_thermal_device_is_a_configuration_error():
    ev = Evaluator(buck, period=T, steady=SteadySpec(method="pss"),
                   thermal=ThermalSpec(r_th={"Q9": 1.0}))
    with pytest.raises(KeyError, match="Q9"):
        ev.evaluate(CCM)
