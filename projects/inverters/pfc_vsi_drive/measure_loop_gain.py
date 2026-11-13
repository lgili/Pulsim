#!/usr/bin/env python3
"""Measure the PFC cascade's two loop gains by injection, per operating point.

The cascade gains in ``PfcCascadeController`` are designed for a crossover
(``pfc_i_bw_hz``, ``pfc_v_bw_hz``) on an averaged plant. This script checks
that design on the switching model: it adds a small sine to the loop
variable (the duty for the inner loop, K_amp for the outer loop) and reads
L(jω) = −X_ctrl / X_applied from one DFT bin, where X_ctrl is what the
controller computed and X_applied is what reached the plant.

The window spans whole line cycles and the injection frequency is not a
multiple of f_line, so the line-frequency harmonics the duty and K_amp
carry anyway fall into other DFT bins. The plant is time-varying over the
line cycle (V_rect follows |v_ac|); the measured L is its line-cycle
average.

Usage:
    PYTHONPATH=python python3 projects/inverters/pfc_vsi_drive/measure_loop_gain.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pulsim as p  # noqa: E402
from pfc_vsi_drive_pulsim_validation import make_sim_params, simulate_frontend  # noqa: E402
from validation_data import OPERATING_POINTS  # noqa: E402

SETTLE = 0.10          # s before the measurement window
EPS_DUTY = 0.01        # inner-loop injection amplitude (duty)
EPS_K = 0.03           # outer-loop injection amplitude (fraction of K_amp_steady)
# Injection frequencies as multiples of f_line: never an integer multiple.
I_MULT = (5.5, 10.5, 15.5, 20.5, 30.5, 40.5)       # inner loop (~0.3–2.4 kHz)
V_MULT = (0.1, 0.2, 0.3, 0.4, 0.6, 0.8)            # outer loop (~5–48 Hz)


def _one_point(op, f, *, outer):
    sp = make_sim_params(op)
    f_line = float(op.f_line)
    n_line = 2 if not outer else 10                  # window in line cycles
    window = n_line / f_line
    held = {}

    def hook(ctrl):
        amp = EPS_K * ctrl.K_amp_steady if outer else EPS_DUTY

        def inj(t):
            return amp * math.sin(2 * math.pi * f * t)
        if outer:
            ctrl.k_inj = inj
        else:
            ctrl.d_inj = inj
        ctrl.trace = []
        held["ctrl"] = ctrl

    simulate_frontend(sp, t_end=SETTLE + window, on_controller=hook)
    tr = np.asarray(held["ctrl"].trace)
    w = tr[:, 0] >= SETTLE + window - window - 1e-12
    w &= tr[:, 0] < SETTLE + window - 1e-12
    t = tr[w, 0]
    ctrl_col, applied_col = (1, 2) if outer else (3, 4)
    e = np.exp(-2j * np.pi * f * t)
    return -(tr[w, ctrl_col] @ e) / (tr[w, applied_col] @ e)


def _crossover(f, L):
    lf, lm = np.log10(f), np.log10(np.abs(L))
    out = [10 ** (lf[i] - lm[i] * (lf[i + 1] - lf[i]) / (lm[i + 1] - lm[i]))
           for i in np.where((lm[:-1] >= 0) & (lm[1:] < 0))[0]]
    return out[-1] if out else float("nan")


def measure(op, *, outer):
    mult = V_MULT if outer else I_MULT
    f = np.array([m * float(op.f_line) for m in mult])
    L = np.array([_one_point(op, fi, outer=outer) for fi in f])
    return f, L


def main() -> int:
    print("PFC cascade loop gains, measured by injection (line-cycle average)")
    for op in OPERATING_POINTS:
        sp = make_sim_params(op)
        print(f"\n{op.id}: design crossover inner {sp.pfc_i_bw_hz:.0f} Hz, "
              f"outer {sp.pfc_v_bw_hz:.0f} Hz")
        for outer, name in ((False, "inner (I_L002)"), (True, "outer (V_link)")):
            f, L = measure(op, outer=outer)
            for fi, Li in zip(f, L):
                print(f"    {name:15s} f = {fi:7.1f} Hz  |L| = {abs(Li):7.3f}  "
                      f"∠L = {math.degrees(np.angle(Li)):+7.1f}°")
            pm = p.phase_margin_from_loop(f, L)
            gm = p.gain_margin_from_loop(f, L)
            print(f"  {op.id} {name}: crossover {_crossover(f, L):.1f} Hz, "
                  f"phase margin {pm:.1f}°, gain margin "
                  f"{'> sweep' if math.isinf(gm) else f'{gm:.1f} dB'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
