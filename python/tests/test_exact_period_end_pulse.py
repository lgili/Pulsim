"""A gate pulse 1e-7 of the period long on the exact engine, at the period's end and at its start.

The run-end tie of the dsed schedulers moved an edge within 1e-12 s (an absolute floor) of
t_end onto t_end. A one-period run of the periodic-orbit kernel ends on the period, so the
falling edge of D = 1 - 1e-7 at 100 kHz (1e-12 s before T) vanished: the period map had no
edge and the evaluator 0.0 W of switching loss (optimisation DECISIONS 31; the same pulse
shifted to the period start gave 2.70 W). The tie is now a few ulps.
"""
import numpy as np
import pytest

import pulsim as p
from pulsim.design import Evaluator, SteadySpec
from .test_design_thermal_kernel import SPECS, T


def _buck(d, phase):
    def build(_x=None):
        b = p.CircuitBuilder()
        b.add_voltage_source("Vin", "vin", "gnd", 24.0)
        b.add_switch("S", "vin", "sw", 1e3, 1e-9)
        b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
        b.add_inductor("L", "sw", "x", 100e-6)
        b.add_resistor("RLs", "x", "vout", 0.01)
        b.add_capacitor("C", "vout", "gnd", 47e-6)
        b.add_resistor("RL", "vout", "gnd", 1.0)
        return b, p.make_pwm_switch_fn(1 / T, d, b.switch_index_of("S"),
                                       b.graph.num_switches, phase * T)
    return build


def _p_sw(d, phase):
    rec = Evaluator(_buck(d, phase), period=T, loads=("RL",), output="vout",
                    steady=SteadySpec(method="pss"), loss_options=SPECS).evaluate({})
    assert rec.status == "ok", rec.reason
    return rec.losses["S"]["switching"]


@pytest.mark.parametrize("d", [
    pytest.param(1e-7, marks=pytest.mark.xfail(
        strict=True, reason="release finding F-5-3: mode_selection='auto' (LCP) on dsed drops a 1 ps "
        "gate pulse: buck v_out 2.4e-8 V (leakage only) vs 2.37e-6 V = 0.987 D Vin (settle)")),
    1 - 1e-7])
def test_a_short_pulse_counts_wherever_it_sits_in_the_period(d):
    # the same periodic waveform, sectioned at two phases: the same commutations
    at_edge, shifted = _p_sw(d, 0.0), _p_sw(d, 0.5)
    assert at_edge > 0.5                                   # 0.80 W / 2.70 W, was 0.0 at 1 - 1e-7
    assert at_edge == pytest.approx(shifted, rel=1e-9)     # measured <= 1e-15


@pytest.mark.parametrize("t_end", [3 * T, T + T + T])   # a product, an accumulated sum
@pytest.mark.parametrize("engine", ["dsed", "exact"])
def test_the_last_edge_of_a_run_keeps_its_instant(engine, t_end):
    b, fn = _buck(1 - 1e-7, 0.0)()
    r = p.simulate(b, t_end, engine=engine, switch_fn=fn)
    ev = np.asarray(r.event_times)
    for k in (1, 2, 3):
        # every falling edge 1 ps before its period end, the last one included
        assert np.any(np.abs(ev - (k * T - 1e-12)) < 1e-16), (k, ev)
    # the run ends on the gate's edge at t_end, not a sliver past it
    assert float(r.times[-1]) == pytest.approx(t_end, abs=1e-18)
