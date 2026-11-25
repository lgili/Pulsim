"""Kernel/API gaps of the optimisation track: ``pulsim.ConvergenceError``
raised at the sources, and ``pulsim.periodic_steady_state_supported`` on
the refusal path of ``periodic_steady_state``."""

import pytest

import pulsim as p
from pulsim import design

from . import _pss_reference as pref

T = pref.T


def _buck(lauritzen=False, shockley=False):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", pref.GON, pref.GOFF)
    if lauritzen:
        b.add_lauritzen_diode("D1", "gnd", "sw", tau=1e-7, T_M=1e-8)
    elif shockley:
        b.add_shockley_diode("D1", "gnd", "sw")
    else:
        b.add_diode("D1", "gnd", "sw", pref.GON, pref.GOFF)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 50.0)
    return b, p.make_pwm_switch_fn(1 / T, 0.3, b.switch_index_of("S"),
                                   b.graph.num_switches)


def test_convergence_error_is_the_base_of_every_non_convergence():
    assert issubclass(p.ConvergenceError, RuntimeError)
    assert issubclass(p.SimulationAborted, p.ConvergenceError)
    assert not issubclass(p.UnsupportedError, p.ConvergenceError)


def test_pss_non_convergence_is_typed_and_still_a_value_error():
    b, sf = _buck()
    with pytest.raises(p.ConvergenceError, match="no periodic orbit") as e:
        p.periodic_steady_state(b, period=T, switch_fn=sf, max_period_runs=1)
    assert isinstance(e.value, ValueError)          # what it was before
    assert not isinstance(e.value, p.UnsupportedError)


def test_pss_configuration_errors_are_not_convergence_errors():
    b, sf = _buck()
    with pytest.raises(ValueError) as e:
        p.periodic_steady_state(b, period=-1.0, switch_fn=sf)
    assert not isinstance(e.value, p.ConvergenceError)


def test_supported_is_true_on_the_exact_path():
    b, sf = _buck()
    assert p.periodic_steady_state_supported(b, T, sf) == (True, "")


@pytest.mark.parametrize("kw", [dict(lauritzen=True), dict(shockley=True)])
def test_supported_gives_the_refusal_without_running(kw):
    b, sf = _buck(**kw)
    ok, why = p.periodic_steady_state_supported(b, T, sf)
    assert not ok and why.startswith("periodic_steady_state: ")
    with pytest.raises(p.UnsupportedError) as e:
        p.periodic_steady_state(b, period=T, switch_fn=sf)
    assert str(e.value) == why


def test_design_route_uses_the_typed_refusal():
    def lauritzen(d):
        return _buck(lauritzen=True)

    r = design.Evaluator(lauritzen, period=T, output="vout").evaluate({})
    assert r.status == "ok"
    assert r.counters["steady_method"] == "transient"
    assert r.counters["steady_route_reason"].startswith("pss refused: ")
