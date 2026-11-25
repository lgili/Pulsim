"""Snapshot and resume of a run holding control graphs (pwl).

A graph keeps its tick counters and block states across simulate()
calls on the same builder: a run that goes on from the previous t_end
(resume_from) continues the tick grid (BlockChain.rearm_if_restarted
only restarts it when time goes back), and the C-block graph is
compiled once per builder layout, not once per run. Resumed in the
middle of a controller period, the second segment matches the
uninterrupted run: duty history and tick counts exactly, states as
exactly as an open-loop resume (the snapshot itself).
"""

from __future__ import annotations

import warnings

import numpy as np

import pulsim as p

DT = 1e-6


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    b.add_switch("Q", "in", "sw", 1e2, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e2, 1e-9)
    b.add_inductor("L", "sw", "out", 1e-3)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", 10.0)
    return b


def _loop(b):
    return p.bind_pi_to_switch(
        b, pi=p.PIController(Kp=0.02, Ki=20.0, output_min=0.05,
                             output_max=0.95),
        measured="out", setpoint=12.0, switch="Q", freq=10e3)


def _tail(full, second):
    tf = np.asarray(full.times)
    xf = np.asarray(full.states)[tf >= second.times[0] - 0.5 * DT]
    xs = np.asarray(second.states)
    n = min(len(xf), len(xs))
    return xf[:n], xs[:n]


def test_closed_loop_resumed_mid_period_continues_the_run():
    t1, t2 = 1.53e-3, 3e-3          # t1: 0.3 of a 100 us period in
    bf = _buck()
    lf = _loop(bf)
    full = p.simulate(bf, t2, dt=DT, closed_loops=[lf])
    b = _buck()
    lp = _loop(b)
    first = p.simulate(b, t1, dt=DT, closed_loops=[lp])
    second = p.simulate(b, t2, dt=DT, closed_loops=[lp],
                        resume_from=first.final_snapshot)
    # duties bit for bit; the tick labels are t1 + n*dt on the resumed
    # time grid against n*dt (19 * 1e-4 reads 0.0018999999999999998)
    assert [d for _, d in lp.duty_history] == [d for _, d in lf.duty_history]
    assert len(lp.duty_history) == 30
    np.testing.assert_allclose([t for t, _ in lp.duty_history],
                               [t for t, _ in lf.duty_history],
                               rtol=0, atol=1e-15)
    xf, xs = _tail(full, second)
    assert np.abs(xf - xs).max() <= 1e-9 * np.abs(xf).max()


def test_c_block_resumed_mid_period_continues_the_run():
    def rc():
        b = p.CircuitBuilder()
        b.add_resistor("R", "u", "v", 1e3)
        b.add_capacitor("C", "v", "gnd", 1e-6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            h = p.add_c_block(
                b, inputs=[("v", "v")], outputs=[("v", "u", "gnd")],
                dt=1e-4, fn=lambda t, dt, i, o, s: o.__setitem__(
                    0, 2.0 * (1.0 - i[0])))
        return b, h
    bf, hf = rc()
    full = p.simulate(bf, 2e-3, dt=DT)
    b, h = rc()
    first = p.simulate(b, 0.73e-3, dt=DT)
    n1 = h.n_fires
    second = p.simulate(b, 2e-3, dt=DT, resume_from=first.final_snapshot)
    assert n1 + h.n_fires == hf.n_fires == 20
    xf, xs = _tail(full, second)
    assert np.abs(xf - xs).max() <= 1e-9 * np.abs(xf).max()
