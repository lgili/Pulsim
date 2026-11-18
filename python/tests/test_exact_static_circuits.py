"""A static circuit (no capacitor, no inductor) on the exact engine: nothing to propagate, each
sample an algebraic solve of its mode, events still landed (add-exact-propagator-strategy
DECISIONS, static circuits)."""
import numpy as np
import pytest

import pulsim as p

ENGINES = ["dsed", "exact"]


@pytest.mark.parametrize("engine", ENGINES)
def test_a_divider_is_its_closed_form(engine):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R1", "in", "out", 1e3)
    b.add_resistor("R2", "out", "gnd", 3e3)
    r = p.simulate(b, 1e-3, engine=engine, dt=1e-5)
    assert np.abs(np.asarray(r.v("out")) - 7.5).max() < 1e-12
    assert np.abs(np.asarray(r.i("V")) + 10.0 / 4e3).max() < 1e-15


@pytest.mark.parametrize("engine", ENGINES)
def test_a_gated_resistor_switches_on_its_edges(engine):
    g_on, g_off, R = 1e3, 1e-9, 10.0
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S", "in", "out", g_on, g_off)
    b.add_resistor("R", "out", "gnd", R)
    fn = p.make_pwm_switch_fn(1e4, 0.3, b.switch_index_of("S"), b.graph.num_switches)
    r = p.simulate(b, 3e-4, engine=engine, dt=1e-6, switch_fn=fn)
    t, v = np.asarray(r.times), np.asarray(r.v("out"))
    on = 10.0 * g_on * R / (1.0 + g_on * R)
    off = 10.0 * g_off * R / (1.0 + g_off * R)
    phase = np.mod(t, 1e-4)
    inside_on = (phase > 1e-9) & (phase < 3e-5 - 1e-9)
    inside_off = (phase > 3e-5 + 1e-9) & (phase < 1e-4 - 1e-9)
    assert np.abs(v[inside_on] - on).max() < 1e-12 * on
    assert np.abs(v[inside_off] - off).max() < 1e-12
    # the turn-off edge at 30 us is a stored sample: left limit on, right limit off
    k = int(np.flatnonzero(np.abs(t - 3e-5) < 1e-15)[0])
    assert v[k] == pytest.approx(on, rel=1e-12)
    j = list(np.asarray(r.right_limit_index)).index(k)
    assert np.asarray(r.right_limit_states)[j, b.node_id_of("out")] == pytest.approx(off, abs=1e-12)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_diode_on_an_algebraic_node_commutes_at_the_source_zero(engine):
    # sine -> R -> diode to ground: the diode's predicate is an algebraic quantity driven by the
    # source alone; its events are the source's zero crossings, k * 10 ms
    g_on, g_off, R = 1e3, 1e-9, 100.0
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 5.0, 50.0)
    b.add_resistor("R", "in", "out", R)
    b.add_diode("D", "out", "gnd", g_on, g_off)
    r = p.simulate(b, 0.04, engine=engine, dt=1e-4)
    t, v = np.asarray(r.times), np.asarray(r.v("out"))
    vin = 5.0 * np.sin(2 * np.pi * 50.0 * t)
    ron, roff = 1.0 / g_on, 1.0 / g_off
    closed = np.where(vin > 0, vin * ron / (R + ron), vin * roff / (R + roff))
    away = np.abs(np.mod(t + 1e-6, 0.01) - 1e-6) > 1e-6       # not at a zero crossing
    assert np.abs(v[away] - closed[away]).max() < 1e-9
    ev = np.asarray(r.event_times)
    for t0 in (0.01, 0.02, 0.03):
        assert np.abs(ev - t0).min() < 1e-9
