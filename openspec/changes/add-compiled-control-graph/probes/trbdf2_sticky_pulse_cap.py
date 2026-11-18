"""Reproducer: trbdf2's learned narrowest-pulse step cap never relaxes.

core/include/pulsim/solver/trbdf2_transient.hpp, `learn_gate_toggles`:
for a switch_fn with no edge oracle the engine records the narrowest gate
pulse seen so far (`gate_pulse_min`, a running minimum over the whole
run) and caps every later step at 2x it. A closed loop whose PI ramps up
from duty 0 (benchmarks cl_boost_pi: first duty 2.58e-5 of a 10 us
period) shows the engine a 0.28 ns pulse right after the first ticks,
and h stays <= 0.56 ns for the rest of the run: cl_boost_pi reached
max_steps (1e7) at 5.59 ms of 30 ms.

The loop's gate (`CxxChainPwmGate`) declares its edges since G23, which
turns the learning off. Hiding them behind a Python lambda restores the
grind:

  PYTHONPATH=<repo>/python python trbdf2_sticky_pulse_cap.py

Measured (macOS arm64, 2026-11-18), 0.3 ms of cl_boost_pi:
  gate with its edges     accepted 1,590 steps,  max h after 0.1 ms = 3e-7 s
  opaque lambda gate      accepted 519,746 steps,  gate_pulse_min = 2.79e-10 s,
                          max h after 0.1 ms = 5.58e-10 s
"""
import numpy as np

import pulsim


def boost_loop():
    b = pulsim.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 5.0)
    b.add_resistor("R_dcr", "vin", "n_l", 0.1)
    b.add_inductor("L1", "n_l", "sw", 47e-6)
    b.add_switch("SL", "sw", "gnd", 100.0, 1e-6)
    b.add_diode("D", "sw", "out", 100.0, 1e-6)
    b.add_capacitor("C", "out", "gnd", 470e-6, 10.0)
    b.add_resistor("R", "out", "gnd", 40.0)
    pi = pulsim.PIController(Kp=0.005, Ki=50.0, output_min=0.0, output_max=0.9)
    loop = pulsim.bind_pi_to_switch(b, pi=pi, measured="out", setpoint=10.0,
                                    switch="SL", freq=100e3)
    return b, loop


for label, opaque in (("gate with its edges", False), ("opaque lambda gate", True)):
    b, loop = boost_loop()
    gate = loop.switch_fn
    sw = (lambda t: gate(t)) if opaque else gate
    r = pulsim.simulate(b, 3e-4, engine="trbdf2", switch_fn=sw,
                        step_observer=loop.step_observer)
    t = np.asarray(r.times)
    st = r._trbdf2_stats
    print(f"{label:22s} accepted {st['n_accept']:8d}  gate_pulse_min "
          f"{st.get('gate_pulse_min', float('inf')):.3g}  max h after 0.1 ms "
          f"{np.diff(t[t > 1e-4]).max():.3g} s")
