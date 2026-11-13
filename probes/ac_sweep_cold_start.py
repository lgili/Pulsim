"""Probe: run_ac_sweep on a plant that starts away from its steady state.

LC low-pass (100 µH, 47 µF, 5 Ω; f0 = 2.32 kHz, τ = 2RC = 0.47 ms)
started from zero with 24 V applied: the start-up ring is volts while
the response to a 0.1 V perturbation at 11 kHz is millivolts. With 8
settling cycles (0.73 ms) the ring is still in the DFT window.
Analytic: H = 1 / (1 - ω²LC + jωL/R).
"""
from __future__ import annotations

import math
import warnings


import pulsim as p

L, C, R = 100e-6, 47e-6, 5.0


def plant():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    b.add_inductor("L1", "in", "out", L)
    b.add_capacitor("C1", "out", "gnd", C)
    b.add_resistor("RL", "out", "gnd", R)
    return b


def sweep(**kw):
    b = plant()
    n = b.pool.state_size(b.graph)
    k = b.pool.branch_var_id_for_source(b.branch_id_of("Vin"), b.graph)

    def excite(t, eps, f):
        out = [0.0] * n
        out[k] = -eps * math.sin(2 * math.pi * f * t)  # source row: write -v
        return out

    return p.run_ac_sweep(b, freqs=[11e3], excite_fn=excite, output_idx=b.node_id_of("out"),
                          perturbation_amplitude=0.1, settling_cycles=8, cycles_per_point=20,
                          start_from_dc_op=False, verbose=False, **kw)


w = 2 * math.pi * 11e3
exact = 20 * math.log10(abs(1 / (1 - w * w * L * C + 1j * w * L / R)))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    one = sweep(max_extra_windows=0)
print(f"analytic {exact:.3f} dB; one window (old behaviour) {one.mag_dB[0]:.3f} dB; "
      f"settled windows {sweep().mag_dB[0]:.3f} dB")
