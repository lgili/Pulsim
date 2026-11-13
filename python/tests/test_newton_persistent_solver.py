"""update-newton-persistent-solver: per-run Newton counters on the result.

Counters only, no wall-clock assertion (task 7.2)."""
import numpy as np
import pulsim as p


def _shockley_buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("Q1", "vin", "sw", 1e3, 1e-9)
    b.add_shockley_diode("D1", "gnd", "sw", 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 10.0)
    fn = p.make_pwm_switch_fn(100e3, 0.5, b.switch_index_of("Q1"), b.graph.num_switches)
    return b, fn


def test_newton_counters_one_analysis_per_run():
    b, fn = _shockley_buck()
    res = p.simulate(b, switch_fn=fn, t_end=2e-4, dt=1e-7, engine="pwl")
    nw = res.newton
    trace = np.asarray(nw.iterations_per_solve)
    assert nw.solves > 0
    assert trace.dtype == np.uint16
    assert trace.size == nw.solves
    assert int(trace.sum()) == nw.iterations
    assert nw.analyses == 1                       # one pattern for the run
    assert nw.factorizations == nw.iterations     # one refactor per iteration
    assert nw.lm_attempts == 0


def test_newton_counters_zero_without_newton():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_resistor("R", "a", "b", 1.0)
    b.add_capacitor("C", "b", "gnd", 1e-6)
    res = p.simulate(b, t_end=1e-5, dt=1e-7, engine="pwl")
    assert res.newton.solves == 0
    assert np.asarray(res.newton.iterations_per_solve).size == 0
