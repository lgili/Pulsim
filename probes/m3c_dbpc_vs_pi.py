"""Probe: why does the M3C DBPC (notebook 06) show more THD than PI/dq?

Runs both controllers at 45 Hz / 100 A on the L1 plant and reports the
fundamental amplitude (Fourier projection over whole periods), the THD
and the per-period quantisation step, for the shipped 6 x 4 kV module
and for the same module voltage split into finer levels.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1] / "projects/inverters/m3c_3phase"
sys.path.insert(0, str(HERE))
from m3c_3phase_model import (  # noqa: E402
    M3cParams, build_l1_plant, make_sinusoidal_abc_ref, run_l1_dbpc,
    run_l1_dq_full_closed_loop_with_cap_loop, thd,
)

DT, F, PK, T_END = 25e-6, 45.0, 100.0, 1.0


def fund(t, i, f=F, periods=9):
    n = int(round(periods / f / DT))
    t, i = t[-n:], i[-n:]
    w = 2 * np.pi * f
    a = 2 * np.mean(i * np.cos(w * t))
    b = 2 * np.mean(i * np.sin(w * t))
    return float(np.hypot(a, b)), thd(i, 1 / DT, f)


for label, kw in [("shipped N=6 x 4 kV", {}),
                  ("N=24 x 1 kV", dict(n_sm_per_module=24, v_cap_nominal=1000.0))]:
    prm = replace(M3cParams(), **kw)
    r, _, _ = run_l1_dbpc(build_l1_plant(prm), prm,
                          i_out_ref_fn=make_sinusoidal_abc_ref(PK, F), t_end=T_END, dt=DT)
    a, h = fund(r.t, r.i_a_out)
    print(f"DBPC  {label:20s}: |I1| = {a:7.2f} A  THD = {h:6.2f} %")
    r, *_ = run_l1_dq_full_closed_loop_with_cap_loop(
        build_l1_plant(prm), prm, i_d_out_ref=PK, t_end=T_END, dt=DT)
    a, h = fund(r.t, r.i_a_out)
    print(f"PI/dq {label:20s}: |I1| = {a:7.2f} A  THD = {h:6.2f} %")
