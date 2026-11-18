"""How the exact engine samples a discontinuity (add-exact-propagator-strategy DECISIONS,
"Samples at a discontinuity"): a stored sample at an event is the LEFT limit, the RIGHT limit
is recorded beside it (``right_limit_*``, source jumps included) and
``with_right_limits`` places it one ulp later; the final sample at a corner landing on t_end is
the left limit; a mode too fast for the output step is sampled after the event that excites it."""
import numpy as np
import pytest

import pulsim as p
from pulsim._result_views import with_right_limits

ENGINES = ["dsed", "exact"]


def _pwm_rc():
    b = p.CircuitBuilder()
    b.add_pwm_voltage_source("Vg", "in", "gnd", 10.0, 0.0, 1e4, 0.5)   # edges every 50 us
    b.add_resistor("R", "in", "out", 100.0)
    b.add_capacitor("C", "out", "gnd", 1e-7)                            # tau = 10 us
    return b


@pytest.mark.parametrize("engine", ENGINES)
def test_a_source_edge_is_two_sided(engine):
    r = p.simulate(_pwm_rc(), 2e-4, engine=engine, dt=1e-6)
    t = np.asarray(r.times)
    k = int(np.flatnonzero(np.abs(t - 5e-5) < 1e-15)[0])
    assert np.asarray(r.v("in"))[k] == 10.0                  # left limit: still high
    two = with_right_limits(r)
    t2 = np.asarray(two.times)
    assert np.all(np.diff(t2) > 0)
    j = int(np.flatnonzero(t2 == t[k])[0])
    assert t2[j + 1] == np.nextafter(t[k], np.inf)
    assert np.asarray(two.v("in"))[j + 1] == 0.0             # right limit: the new level
    vout = np.asarray(two.v("out"))
    assert vout[j + 1] == pytest.approx(vout[j], abs=1e-12)  # the capacitor does not jump
    # closed form of the first half period: v = 10 (1 - exp(-t / 10 us))
    assert vout[j] == pytest.approx(10.0 * (1 - np.exp(-5.0)), rel=1e-9)


@pytest.mark.parametrize("engine", ENGINES)
def test_the_last_sample_on_a_corner_is_the_left_limit(engine):
    # t_end = 1e-4 is a rising edge of the gate source: the run ends there, so the new level is
    # never integrated and the stored sample keeps the old one
    r = p.simulate(_pwm_rc(), 1e-4, engine=engine, dt=1e-6)
    assert np.asarray(r.times)[-1] == pytest.approx(1e-4, abs=1e-18)
    assert np.asarray(r.v("in"))[-1] == 0.0


def _rl_turn_off():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_switch("S", "vin", "a", 1e6, 1e-12)
    b.add_resistor("R1", "a", "b", 10.0)
    b.add_inductor("L", "b", "gnd", 1e-3)
    b.add_resistor("Rbig", "a", "gnd", 1e5)
    fn = p.make_pwm_switch_fn(1e3, 0.5, b.switch_index_of("S"), b.graph.num_switches)
    return b, fn


@pytest.mark.parametrize("engine", ENGINES)
def test_a_fast_mode_after_an_event_is_sampled(engine):
    b, fn = _rl_turn_off()
    r = p.simulate(b, 6e-4, engine=engine, dt=1e-6, switch_fn=fn)
    t, i = np.asarray(r.times), np.asarray(r.i("L"))
    t_off = 5e-4
    k = int(np.flatnonzero(np.abs(t - t_off) < 1e-15)[0])
    i0 = i[k]
    assert i0 > 0.9                                   # the on-state current, ~1 A
    tau = 1e-3 / (10.0 + 1e5)                         # ~10 ns, a hundredth of the output step
    inside = (t > t_off) & (t < t_off + 1e-6)
    assert inside.sum() >= 8                          # samples inside the first output step
    s = np.linspace(t_off, t_off + 2e-6, 4001)
    exact = i0 * np.exp(-(s - t_off) / tau)
    # the linear interpolant through the samples follows the exponential (one step of 1 us
    # drew it as a ramp, 0.5 A off at mid-step)
    assert np.abs(np.interp(s, t, i) - exact).max() < 0.05 * i0
