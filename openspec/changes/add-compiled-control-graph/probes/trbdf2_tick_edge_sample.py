"""Reproducer: trbdf2 records no post-edge sample at a gate edge that is
also a controller tick.

core/include/pulsim/solver/trbdf2_transient.hpp, `restart_after_gate_event`:
after a landed gate edge the engine commits a probe solve h_p later under
the post-edge mask and records it, so the trace shows the edge as a jump.
It refuses that commit when `t + h_p >= t_next_tick - h_min` ("it would
fire dtp late"). A closed loop's period start IS a tick (the PI runs at
kT and the carrier restarts at kT), so t_next_tick == t at every rising
edge: the slots are refreshed, nothing is recorded, and the trace ramps
the switch node from its pre-edge value to the post-edge one across the
whole first post-edge step. That step's length follows rtol, so V(sw) of
the closed-loop references (trbdf2 rtol 1e-8) does not converge with rtol
while the states do. Open loop (no tick at the edge): both edges keep
their post-edge sample.

Run: PYTHONPATH=<repo>/python python trbdf2_tick_edge_sample.py
Measured (macOS arm64, 2026-11-18), 2 ms, 100 kHz buck, duty ~0.5:
  open loop    rising edges with a post-edge sample 199/199, falling 200/200
  closed loop  rising edges with a post-edge sample   0/199, falling 200/200
"""
import numpy as np

import pulsim

T = 1e-5


def buck():
    b = pulsim.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 10.0)
    b.add_switch("S", "vin", "sw", 100.0, 1e-6)
    b.add_diode("D", "gnd", "sw", 100.0, 1e-6)
    b.add_inductor("L", "sw", "out", 100e-6)
    b.add_capacitor("C", "out", "gnd", 100e-6)
    b.add_resistor("R", "out", "gnd", 10.0)
    return b


def count(r):
    """(rising with a post-edge sample, rising), same for falling: an
    edge's post-edge sample sits within 1 ns of its pre-edge one."""
    t = np.asarray(r.times)
    s = np.asarray(r.switch_states)[:, 0]
    k = np.flatnonzero(s[1:] != s[:-1])
    near = (t[k + 1] - t[k]) < 1e-9
    up = s[k + 1]
    return (int(np.sum(near & up)), int(np.sum(up)),
            int(np.sum(near & ~up)), int(np.sum(~up)))


b = buck()
r = pulsim.simulate(b, 2e-3, engine="trbdf2",
                    switch_fn=pulsim.make_pwm_switch_fn(1 / T, 0.5, 0, 1))
print("open loop    rising %d/%d  falling %d/%d" % count(r))

b = buck()
pi = pulsim.PIController(Kp=0.02, Ki=40.0, output_min=0.1, output_max=0.9)
loop = pulsim.bind_pi_to_switch(b, pi=pi, measured="out", setpoint=5.0,
                                switch="S", freq=1 / T)
r = pulsim.simulate(b, 2e-3, engine="trbdf2", switch_fn=loop.switch_fn,
                    step_observer=loop.step_observer)
print("closed loop  rising %d/%d  falling %d/%d" % count(r))
