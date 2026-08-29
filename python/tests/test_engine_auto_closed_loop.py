"""Closed-loop control on the variable-step engine.

v2.0 Phase 3, the audit's "observers no DSED — cadência
sincronizada a eventos". A digital controller samples at k·T_ctrl.
The fixed engine can only THROTTLE an every-step observer on that
period: the tick lands on whichever step first crosses the
boundary, the phase drifts with the grid, and — the part that
actually changes the answer — the PWM edge the controller commands
is quantized to the grid. `engine='auto'` schedules the tick
instants and lands them (and the gate edges) exactly.
"""

import numpy as np
import pytest

import pulsim as ps

FREQ = 10e3
T_CTRL = 1.0 / FREQ


def _buck(vin=12.0, R_L=8.0):
    b = ps.CircuitBuilder()
    b.add_voltage_source("V1", "vin", "gnd", vin)
    b.add_mosfet_with_body_diode("Q1", "vin", "sw",
                                  R_on=1e-3, R_off=1e9, V_F=0.7)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9, V_th=0.7)
    b.add_inductor("L1", "sw", "vout", 220e-6)
    b.add_capacitor("Cout", "vout", "gnd", 220e-6)
    b.add_resistor("R_L", "vout", "gnd", R_L)
    return b


def _pi_buck(setpoint=5.0):
    b = _buck()
    pi = ps.PIController(Kp=0.08, Ki=40.0,
                          output_min=0.05, output_max=0.95)
    duty_get, obs, hist = ps.bind_pi_to_duty_callable(
        b, pi=pi, measured=lambda x: x[b.node_id_of("vout")],
        setpoint=setpoint, freq=FREQ)
    q = b.switch_index_of("Q1")
    n = b.graph.num_switches

    def switch_fn(t):
        m = ps.SwitchStateMask(n)
        if (t % T_CTRL) / T_CTRL < duty_get():
            m.set(q, True)
        return m

    return b, switch_fn, obs, hist


def _tail_mean(res, node, from_t):
    t = np.asarray(res.times)
    v = np.asarray(res.v(node))
    m = t > from_t
    return float(np.trapezoid(v[m], t[m]) / (t[m][-1] - t[m][0]))


def test_buck_pi_reaches_setpoint_without_choosing_dt():
    b, sf, obs, hist = _pi_buck()
    res = ps.simulate(b, t_end=20e-3, switch_fn=sf,
                       step_observer=obs, engine="auto")
    v_ss = _tail_mean(res, "vout", 15e-3)
    assert v_ss == pytest.approx(5.0, rel=0.05)
    # Ideal duty is Vout/Vin = 0.4167; the PI lands near it.
    assert 0.35 < hist[-1][1] < 0.5
    # And it agrees with the FINEST fixed-step run (dt = 2e-8,
    # 1e6 steps): 4.972616 V.
    assert v_ss == pytest.approx(4.9726, abs=0.02)


def test_controller_ticks_are_exact_and_none_are_lost():
    """The cadence is scheduled, not throttled.

    The fixed engine drifts (198 ticks instead of 200 over 20 ms
    at 10 kHz, because `last_tick = t` re-anchors to the grid),
    and the loop's own throttle would reject a third of exactly
    scheduled ticks to floating point — hence the unthrottled
    `.tick` entry point this path uses.
    """
    b, sf, obs, hist = _pi_buck()
    res = ps.simulate(b, t_end=20e-3, switch_fn=sf,
                       step_observer=obs, engine="auto")
    assert res._trbdf2_stats["n_ctrl_ticks"] == 200
    assert len(hist) == 200
    spacing = np.diff(np.array([h[0] for h in hist]))
    assert np.abs(spacing - T_CTRL).max() < 1e-15


def test_realized_duty_matches_the_commanded_duty():
    """What the circuit actually SEES.

    A fixed grid quantizes the commanded PWM edge: at dt = 2 µs on
    a 100 µs period the realized duty missed the command by 12386
    ppm, and even 1e6 steps at dt = 20 ns left 157 ppm. Exact edge
    landing makes it a rounding error.
    """
    b, sf, obs, hist = _pi_buck()
    res = ps.simulate(b, t_end=20e-3, switch_fn=sf,
                       step_observer=obs, engine="auto")
    t = np.asarray(res.times)
    v_sw = np.asarray(res.v("sw"))
    sel = (t >= 19.9e-3) & (t <= 20e-3)          # the last period
    ts, vs = t[sel], v_sw[sel]
    dts = np.diff(ts)
    v_mid = 0.5 * (vs[1:] + vs[:-1])
    realized = float(np.sum(dts[v_mid > 6.0]) / np.sum(dts))
    commanded = hist[-1][1]
    assert abs(realized - commanded) < 1e-5      # measured 4e-7


def test_closed_loop_handle_drives_the_engine():
    """A ClosedLoop carries its own period and switch_fn — the
    caller passes it and nothing else."""
    b = _buck()
    pi = ps.PIController(Kp=0.08, Ki=40.0,
                          output_min=0.05, output_max=0.95)
    loop = ps.bind_pi_to_switch(
        b, pi=pi, measured=lambda x: x[b.node_id_of("vout")],
        setpoint=5.0, switch="Q1", freq=FREQ)
    assert loop.period == pytest.approx(T_CTRL)
    res = ps.simulate(b, t_end=20e-3, closed_loops=[loop],
                       engine="auto")
    assert _tail_mean(res, "vout", 15e-3) == pytest.approx(
        5.0, rel=0.05)
    assert res._trbdf2_stats["n_ctrl_ticks"] == 200
    assert len(loop.duty_history) == 200


def test_setpoint_step_is_tracked():
    """A reference step mid-run: the loop must follow it, and the
    tick that sees the new setpoint is the one scheduled at the
    instant it changes."""
    def sp(t):
        return 3.0 if t < 10e-3 else 6.0

    b = _buck()
    pi = ps.PIController(Kp=0.08, Ki=40.0,
                          output_min=0.05, output_max=0.95)
    duty_get, obs, hist = ps.bind_pi_to_duty_callable(
        b, pi=pi, measured=lambda x: x[b.node_id_of("vout")],
        setpoint=sp, freq=FREQ)
    q = b.switch_index_of("Q1")
    n = b.graph.num_switches

    def switch_fn(t):
        m = ps.SwitchStateMask(n)
        if (t % T_CTRL) / T_CTRL < duty_get():
            m.set(q, True)
        return m

    res = ps.simulate(b, t_end=20e-3, switch_fn=switch_fn,
                       step_observer=obs, engine="auto")
    t = np.asarray(res.times)
    v = np.asarray(res.v("vout"))
    before = float(np.mean(v[(t > 8e-3) & (t < 10e-3)]))
    after = float(np.mean(v[t > 18e-3]))
    assert before == pytest.approx(3.0, rel=0.08)
    assert after == pytest.approx(6.0, rel=0.08)


def test_observer_without_a_cadence_refuses():
    """There is no fixed grid to ride here, so 'every step' is not
    a cadence a controller can be written against."""
    b = _buck()
    n = b.graph.num_switches

    def sf(t):
        m = ps.SwitchStateMask(n)
        m.set(b.switch_index_of("Q1"), (t % T_CTRL) / T_CTRL < 0.4)
        return m

    with pytest.raises(ValueError, match="controller_period"):
        ps.simulate(b, t_end=1e-3, switch_fn=sf,
                     step_observer=lambda t, x: None,
                     engine="auto")


def test_bare_observer_with_explicit_period():
    b = _buck()
    n = b.graph.num_switches
    seen = []

    def sf(t):
        m = ps.SwitchStateMask(n)
        m.set(b.switch_index_of("Q1"), (t % T_CTRL) / T_CTRL < 0.4)
        return m

    ps.simulate(b, t_end=1e-3, switch_fn=sf,
                 step_observer=lambda t, x: seen.append(t),
                 controller_period=50e-6, engine="auto")
    assert len(seen) == 20
    assert np.abs(np.diff(seen) - 50e-6).max() < 1e-15


def test_fixed_engine_semantics_are_unchanged():
    """The throttled observer and its history are exactly what
    they were — the new `.tick` is an added entry point, not a
    behaviour change."""
    b, sf, obs, hist = _pi_buck()
    res = ps.simulate(b, t_end=20e-3, dt=2e-6, switch_fn=sf,
                       step_observer=obs)
    assert _tail_mean(res, "vout", 15e-3) == pytest.approx(
        4.985423, abs=1e-4)
    assert len(hist) == 198        # the drift, still there
