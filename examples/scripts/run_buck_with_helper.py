#!/usr/bin/env python3
"""Closed-loop buck — minimal showcase of v1.5 ergonomics.

Demonstrates `add-python-closed-loop-helper` + `add-python-named-
lookups`. Every line is purposeful:

  * Builder declarative — `c0=` would seed an output IC if needed.
  * `bind_pi_to_switch` collapses the 30-line PI+PWM closure pattern.
  * `result.v("vout")` removes the `n_nodes + branch_idx` arithmetic.

A more featureful closed-loop example (load step, CCM↔DCM
transition) lives at `examples/scripts/run_buck_closed_loop.py`.
"""
from __future__ import annotations

import numpy as np
import pulsim as p

# Plant: V_in=12 V, V_ref=5 V, 220 µH / 220 µF / 8 Ω, 10 kHz PWM.
b = p.CircuitBuilder()
b.add_voltage_source("V1", "vin", "gnd", 12.0)
b.add_mosfet_with_body_diode("Q1", "vin", "sw", R_on=1e-3, R_off=1e9, V_F=0.7)
b.add_diode("D1", "gnd", "sw", 1e3, 1e-9, V_th=0.7)
b.add_inductor("L1", "sw", "vout", 220e-6)
b.add_capacitor("Cout", "vout", "gnd", 220e-6)
b.add_resistor("R_L", "vout", "gnd", 8.0)

# Controller: PI with anti-windup, throttled to one update per PWM cycle.
pi = p.PIController(Kp=0.08, Ki=40.0, output_min=0.05, output_max=0.95)
loop = p.bind_pi_to_switch(
    b, pi=pi,
    measured=lambda x: x[b.node_id_of("vout")],
    setpoint=5.0,
    switch="Q1",
    freq=10e3,
)

# Run.
res = p.simulate(b, t_end=20e-3, dt=2e-6, closed_loops=[loop])

# Inspect via named accessors — no `n_nodes + idx` arithmetic.
v_out_ss = float(np.mean(res.v("vout")[-1000:]))
final_duty = loop.duty_history[-1][1]
print(f"V_out steady-state: {v_out_ss:.4f} V  (target 5.0 V)")
print(f"Final duty:         {final_duty:.4f}  (ideal 5/12 = {5/12:.4f})")
print(f"Error:              {abs(v_out_ss - 5.0) / 5.0 * 100:.2f}%")
