"""Probe: phase of the M3C output current against its reference at 45 Hz.

The controllers sample the currents at the start of each T_s period but
evaluate the angle (PI/dq) or the reference (DBPC) at the period CENTRE
(+ T_s for the DBPC's predict-ahead), half a period late. At 45 Hz and
T_s = 0.5 ms that is 4.05 degrees.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "projects/inverters/m3c_3phase"))
from m3c_3phase_model import (  # noqa: E402
    M3cParams, build_l1_plant, make_sinusoidal_abc_ref, run_l1_dbpc,
    run_l1_dq_full_closed_loop_with_cap_loop,
)

DT, F = 25e-6, 45.0
prm = M3cParams()


def phasor(t, x):
    n = int(round(9 / F / DT))
    t, x = t[-n:], x[-n:]
    w = 2 * np.pi * F
    return 2 * np.mean(x * np.exp(-1j * w * t))  # ref 100*cos(wt) -> 100+0j


r, _, _ = run_l1_dbpc(build_l1_plant(prm), prm, i_out_ref_fn=make_sinusoidal_abc_ref(100.0, F),
                      t_end=0.6, dt=DT)
z = phasor(r.t, r.i_a_out)
print(f"DBPC : |I1| = {abs(z):7.2f} A, phase vs ref = {np.degrees(np.angle(z)):+6.2f} deg")
r, *_ = run_l1_dq_full_closed_loop_with_cap_loop(build_l1_plant(prm), prm, i_d_out_ref=100.0,
                                                 t_end=0.6, dt=DT)
z = phasor(r.t, r.i_a_out)
print(f"PI/dq: |I1| = {abs(z):7.2f} A, phase vs ref = {np.degrees(np.angle(z)):+6.2f} deg "
      f"(i_q = {abs(z) * np.sin(np.angle(z)):+.2f} A)")
