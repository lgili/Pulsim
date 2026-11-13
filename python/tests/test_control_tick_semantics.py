"""Controller ticks: one rule on every engine (fix-control-tick-semantics).

Source j with period T_j ticks at t_start + k*T_j while that instant is
< t_end: ceil(span / T_j) times, the first at t_start. Before this
change: two ClosedLoops on the variable-step engine raised TypeError;
loops at two rates on trbdf2 all ticked at the fastest rate (400 PI
updates instead of 40, V_out 12.091 V instead of 13.033 V);
`controller_period` was ignored on pwl (2001 observer calls instead of
20); the fixed engine's throttle lost ticks (198 of 200).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p


def _two_bucks():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    for s in ("a", "b"):
        b.add_switch(f"Q{s}", "in", f"sw{s}", 1e2, 1e-9)
        b.add_diode(f"D{s}", "gnd", f"sw{s}", 1e2, 1e-9)
        b.add_inductor(f"L{s}", f"sw{s}", f"out{s}", 1e-3)
        b.add_capacitor(f"C{s}", f"out{s}", "gnd", 47e-6)
        b.add_resistor(f"R{s}", f"out{s}", "gnd", 10.0)
    return b


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    b.add_switch("Q", "in", "sw", 1e2, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e2, 1e-9)
    b.add_inductor("L", "sw", "out", 1e-3)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", 10.0)
    return b


def _rlc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "in", "gnd", 12.0)
    b.add_resistor("Rs", "in", "a", 1.0)
    b.add_inductor("L1", "a", "out", 1e-3)
    b.add_capacitor("C1", "out", "gnd", 10e-6)
    b.add_resistor("RL", "out", "gnd", 10.0)
    return b


def _loop(b, out, sw, freq):
    return p.control.bind_pi_to_switch(
        b, pi=p.PIController(Kp=0.02, Ki=20.0, output_min=0.05,
                             output_max=0.95),
        measured=lambda x, i=b.node_id_of(out): x[i], setpoint=12.0,
        switch=sw, freq=freq)


def _sim(b, t_end, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end, **kw)


def test_two_closed_loops_without_dt():
    b = _two_bucks()
    la, lb = _loop(b, "outa", "Qa", 20e3), _loop(b, "outb", "Qb", 2e3)
    r = _sim(b, 10e-3, closed_loops=[la, lb])
    assert r.engine_used == "trbdf2"
    assert (len(la.duty_history), len(lb.duty_history)) == (200, 20)
    assert list(r.control_tick_counts) == [200, 20]


def test_one_loop_plus_a_user_switch_fn_without_dt():
    b = _two_bucks()
    la = _loop(b, "outa", "Qa", 20e3)
    user = p.make_pwm_switch_fn(2e3, 0.5, 1, b.graph.num_switches)
    _sim(b, 10e-3, closed_loops=[la], switch_fn=user)
    assert len(la.duty_history) == 200


def _multi_rate(monitor):
    b = _buck()
    lp = _loop(b, "out", "Q", 2e3)
    calls = []
    kw = {}
    if monitor:
        kw = dict(step_observer=lambda t, x: calls.append(t),
                  controller_period=50e-6)
    r = _sim(b, 20e-3, closed_loops=[lp], engine="auto", **kw)
    t, v = np.asarray(r.times), np.asarray(r.v("out"))
    return r, lp, calls, p.time_average(t, v, t_from=15e-3)


def test_each_source_ticks_at_its_own_rate_on_trbdf2():
    r, lp, calls, v_mon = _multi_rate(True)
    assert r.engine_used == "trbdf2"
    assert (len(lp.duty_history), len(calls)) == (40, 400)   # was 400, 400
    _, _, _, v_alone = _multi_rate(False)
    assert v_alone == pytest.approx(13.033, abs=0.01)
    assert v_mon == pytest.approx(v_alone, abs=0.01)         # was 12.091


@pytest.mark.parametrize("engine", ("pwl", "trbdf2"))
def test_controller_period_is_honoured_on_both_engines(engine):
    calls = []
    kw = {"dt": 1e-6} if engine == "pwl" else {}
    _sim(_rlc(), 2e-3, engine=engine, controller_period=1e-4,
         step_observer=lambda t, x: calls.append(t), **kw)
    assert len(calls) == 20                                  # pwl was 2001
    assert calls[0] == 0.0
    np.testing.assert_allclose(calls, np.arange(20) * 1e-4, atol=1e-12)


def test_fixed_engine_loses_no_tick():
    b = _two_bucks()
    la, lb = _loop(b, "outa", "Qa", 20e3), _loop(b, "outb", "Qb", 2e3)
    r = _sim(b, 10e-3, closed_loops=[la, lb], engine="pwl", dt=1e-6)
    assert (len(la.duty_history), len(lb.duty_history)) == (200, 20)
    assert list(r.control_tick_counts) == [200, 20]          # was 198, 20
    assert la.duty_history[0][0] == 0.0


def test_tick_instants_on_trbdf2_are_exact():
    b = _buck()
    lp = _loop(b, "out", "Q", 2e3)
    _sim(b, 10e-3, closed_loops=[lp], engine="trbdf2")
    T = 1.0 / 2e3
    times = np.array([t for t, _ in lp.duty_history])
    assert times[0] == 0.0
    np.testing.assert_allclose(times, np.arange(len(times)) * T,
                               atol=1e-12 * T, rtol=0)


def test_fixed_engine_off_grid_period_warns_and_counts():
    calls = []
    dt = 1e-6
    T = 2.5 * dt
    with pytest.warns(UserWarning, match="not a multiple of dt"):
        r = p.simulate(_rlc(), 1e-4, dt=dt, engine="pwl",
                       controller_period=T,
                       step_observer=lambda t, x: calls.append(t))
    assert len(calls) == 40      # instants 0, 2.5, ..., 97.5 µs (< t_end)
    assert 0.0 < r.control_tick_jitter_max < dt
    with pytest.raises(ValueError, match="shorter than dt"):
        _sim(_rlc(), 1e-4, dt=dt, engine="pwl", controller_period=0.5 * dt,
             step_observer=lambda t, x: None)


@pytest.mark.parametrize("engine", ("pwl", "trbdf2"))
def test_coincident_ticks_fire_in_registration_order(engine):
    order = []

    class _Src:
        def __init__(self, j, period):
            self.j, self.period = j, period

        def __call__(self, t, x):
            order.append((round(t * 1e6), self.j))

    b = _rlc()
    loops = [p.ClosedLoop(switch_fn=None, step_observer=_Src(j, per),
                          duty_history=[], error_history=[], period=per)
             for j, per in ((0, 1e-3), (1, 5e-4))]
    kw = {"dt": 1e-6} if engine == "pwl" else {}
    _sim(b, 2e-3, engine=engine, closed_loops=loops,
         step_observer=_Src(2, 2.5e-4), controller_period=2.5e-4, **kw)
    at_0 = [j for us, j in order if us == 0]
    at_1ms = [j for us, j in order if us == 1000]
    assert at_0 == [0, 1, 2] and at_1ms == [0, 1, 2]


def test_controller_period_without_observer_is_refused():
    with pytest.raises(ValueError, match="without a step_observer"):
        _sim(_rlc(), 1e-4, dt=1e-6, controller_period=1e-5)
