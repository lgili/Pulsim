#!/usr/bin/env python3
"""PFC-VSI compressor drive — sanity-run on OP 2.3 (220V/1090W).

Runs the two-stage simulator (front-end + inverter, see module
docstring for the rationale) for two line-cycles and prints the
basic KPIs side-by-side with the PSIM reference values — enough
to see at a glance whether the model is roughly right before
running the full validation matrix in ``run_validation.py``.

Usage:
    PYTHONPATH=build/python python3 projects/inverters/pfc_vsi_drive/smoke_test.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from validation_data import KPI_23, OP_23
from pfc_vsi_drive_pulsim_validation import (
    make_sim_params,
    simulate_frontend,
    simulate_inverter,
)


def _rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr * arr)))


def _avg(arr: np.ndarray) -> float:
    return float(np.mean(arr))


def main() -> int:
    print(f"=== PFC-VSI compressor drive — smoke test ===")
    print(f"OP: {OP_23.label}")
    print()

    sp = make_sim_params(OP_23)
    sp.t_end = 0.04          # 2 line cycles
    sp.dt = 2.0e-6
    print(f"  duty_pfc:     {sp.duty_pfc:.3f}")
    print(f"  f_sw_pfc:     {sp.f_sw_pfc/1e3:.1f} kHz")
    print(f"  V_link_tgt:   {sp.V_link_target:.1f} V")
    print(f"  f_motor:      {sp.f_motor:.1f} Hz  (= rpm * pp / 60)")
    print(f"  R_load/L_load: {sp.R_load:.1f} Ω / {sp.L_load*1e3:.1f} mH")
    print(f"  t_end:        {sp.t_end*1e3:.1f} ms,  dt = {sp.dt*1e6:.1f} µs")
    print()

    # ---- Front-end ----
    print("  [1/2] front-end (rectifier + boost + bus → R_eq) ...")
    t0 = time.time()
    try:
        fe = simulate_frontend(sp)
    except Exception as exc:
        print(f"        FAILED: {type(exc).__name__}: {exc}")
        return 1
    dt_fe = time.time() - t0
    print(f"        → {len(fe.times)} samples in {dt_fe:.2f}s")

    # ---- Inverter ----
    print("  [2/2] inverter (IPM + 3φ RL on fixed Vdc) ...")
    t0 = time.time()
    try:
        inv = simulate_inverter(sp)
    except Exception as exc:
        print(f"        FAILED: {type(exc).__name__}: {exc}")
        return 1
    dt_inv = time.time() - t0
    print(f"        → {len(inv.times)} samples in {dt_inv:.2f}s")
    print()

    # Discard first half (transient) for steady-state KPIs
    n_fe = len(fe.times) // 2
    n_inv = len(inv.times) // 2

    v_ac_ss   = fe.v_ac[n_fe:]
    v_link_ss = fe.v_link[n_fe:]
    v_rect_ss = fe.v_rect_p[n_fe:]
    v_a_ss    = inv.v_mid_a[n_inv:]
    v_shunt_ss = inv.v_n_shunt_inv[n_inv:]

    print(f"  KPI                target (PSIM)         Pulsim         delta")
    print(f"  " + "-"*68)
    tgt_v_in = KPI_23.V_in_rms or 0.0
    tgt_v_link = KPI_23.V_link_avg or 0.0
    print(f"  V_ac RMS [V]       {KPI_23.V_in_rms:>16.2f}   {_rms(v_ac_ss):>10.2f}     {_rms(v_ac_ss) - tgt_v_in:+.2f}")
    print(f"  V_link AVG [V]     {KPI_23.V_link_avg:>16.2f}   {_avg(v_link_ss):>10.2f}     {_avg(v_link_ss) - tgt_v_link:+.2f}")
    print(f"  V_rect_p AVG [V]   {'-':>16}   {_avg(v_rect_ss):>10.2f}     (no PSIM ref)")
    print(f"  V_mid_a RMS [V]    {'-':>16}   {_rms(v_a_ss):>10.2f}     (no PSIM ref)")
    # I_F500 ≈ V_R508 / R508 (inverter return current envelope)
    from bom import R508
    i_f500 = v_shunt_ss / float(R508.R)
    print(f"  I_F500 RMS [A]     {KPI_23.I_F500_rms:>16.3f}   {_rms(i_f500):>10.3f}     {_rms(i_f500) - (KPI_23.I_F500_rms or 0):+.3f}")
    print()
    print(f"  (full KPI matrix in run_validation.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
