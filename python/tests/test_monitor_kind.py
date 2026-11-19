"""A declared monitor does not change the trajectory (add-unified-event-queue,
the ``Monitor`` event kind).

A ``pulsim.control.Monitor`` only reads. On trbdf2 and dsed its instants
neither end a step nor re-sample the gate, so the run is bit-identical to
the run without it; on pwl it ticks on the grid, which never moved
anything either. The monitor still gets ceil(span / T) calls.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import pulsim as p


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    b.add_switch("Q", "in", "sw", 1e2, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e2, 1e-9)
    b.add_inductor("L", "sw", "out", 1e-3)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", 10.0)
    return b


def _run(engine, **kw):
    b = _buck()
    fn = p.make_pwm_switch_fn(20e3, 0.4, b.switch_index_of("Q"), b.graph.num_switches)
    extra = {"dt": 1e-6} if engine == "pwl" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, 2e-3, engine=engine, switch_fn=fn, **extra, **kw)


@pytest.mark.parametrize("engine", ["pwl", "trbdf2", "dsed"])
def test_a_monitor_only_run_is_bit_identical(engine):
    seen = []
    T = 1e-4 / 3          # off every grid and every PWM edge
    ref = _run(engine)
    mon = _run(engine, step_observer=p.control.Monitor(
        lambda t, x: seen.append(t), T))
    assert np.asarray(mon.times).tobytes() == np.asarray(ref.times).tobytes()
    assert np.asarray(mon.states).tobytes() == np.asarray(ref.states).tobytes()
    assert len(seen) == math.ceil(2e-3 / T) == 60
    assert list(mon.control_tick_counts) == [60]


@pytest.mark.parametrize("engine", ["trbdf2", "dsed"])
def test_the_same_observer_undeclared_lands_its_ticks(engine):
    # control: without the declaration the ticks land (the run changes),
    # so the identity above is the monitor kind's doing
    ref = _run(engine)
    obs = _run(engine, step_observer=lambda t, x: None, controller_period=1e-4 / 3)
    assert np.asarray(obs.times).tobytes() != np.asarray(ref.times).tobytes()


def test_monitor_rejects_a_bad_period():
    with pytest.raises(ValueError):
        p.control.Monitor(lambda t, x: None, 0.0)


@pytest.mark.parametrize("engine", ["trbdf2", "dsed"])
def test_a_monitor_faster_than_the_steps_keeps_every_call(engine):
    # monitors never cap a step, so one step spans many instants: each
    # instant is still called once (review: the count was short before)
    seen = []
    T = 1e-6
    ref = _run(engine)
    mon = _run(engine, step_observer=p.control.Monitor(lambda t, x: seen.append(t), T))
    assert len(seen) == 2000           # k*1e-6 < 2e-3 for k = 0..1999
    assert np.asarray(mon.times).tobytes() == np.asarray(ref.times).tobytes()


def test_a_monitor_on_a_diode_free_dsed_circuit_asks_for_rk45():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 1.0)
    b.add_resistor("R", "in", "out", 1.0)
    b.add_capacitor("C", "out", "gnd", 1e-6)
    m = p.control.Monitor(lambda t, x: None, 1e-6)
    # the refusal needs the 'auto' integrator: without the output rule (output_rtol=0)
    with pytest.raises(p.UnsupportedError, match="with a Monitor"):
        p.simulate(b, 1e-5, engine="dsed", step_observer=m, output_rtol=0)
    ref = p.simulate(b, 1e-5, engine="dsed", integrator="rk45")
    mon = p.simulate(b, 1e-5, engine="dsed", integrator="rk45", step_observer=m)
    assert np.asarray(mon.states).tobytes() == np.asarray(ref.states).tobytes()
    # the release default (output_rtol=1e-5) already runs rk45: the Monitor is
    # accepted and leaves the run bit-identical
    ref = p.simulate(b, 1e-5, engine="dsed")
    mon = p.simulate(b, 1e-5, engine="dsed", step_observer=m)
    assert np.asarray(mon.times).tobytes() == np.asarray(ref.times).tobytes()
    assert np.asarray(mon.states).tobytes() == np.asarray(ref.states).tobytes()


def test_sensitivities_with_controllers_refused_on_dsed():
    b = _buck()
    fn = p.make_pwm_switch_fn(20e3, 0.4, b.switch_index_of("Q"), b.graph.num_switches)
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(b, 1e-4, engine="dsed", switch_fn=fn, sensitivities=["R.value"],
                   step_observer=lambda t, x: None, controller_period=1e-5)
    assert ei.value.feature == "sensitivities"
