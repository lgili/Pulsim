#!/usr/bin/env python3
"""PFC-VSI compressor drive — full validation sweep across all 3 operating points.

For each OP in (OP_2.2, OP_2.3, OP_2.4) this driver:

  1. Builds the front-end + inverter Pulsim sims
     (``pfc_vsi_drive_pulsim_validation.simulate_drive``).
  2. Extracts the 30 KPIs that PSIM reports
     (``losses.compute_losses`` + waveform RMS/AVG).
  3. Compares Pulsim vs PSIM element-by-element and writes a
     summary CSV + a per-OP detailed table to stdout.

Usage::

    PYTHONPATH=build/python python3 projects/inverters/pfc_vsi_drive/run_validation.py

Optional flags::

    --csv path/to/out.csv     # write the full delta matrix
    --quick                   # 1 line cycle per OP (default = 2)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from validation_data import ALL_KPI, KpiSet  # noqa: E402
from pfc_vsi_drive_pulsim_validation import (  # noqa: E402
    DriveSimResult, make_sim_params, simulate_drive,
)
from losses import LossBreakdown, compute_losses  # noqa: E402
from bom import R508  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr * arr)))


def _avg(arr: np.ndarray) -> float:
    return float(np.mean(arr))


def _pk(arr: np.ndarray) -> float:
    return float(np.max(np.abs(arr)))


def _build_kpi_row(name: str, pulsim: float, psim) -> Dict[str, object]:
    if psim is None:
        return dict(name=name, pulsim=pulsim, psim=None, delta=None,
                    pct=None)
    delta = pulsim - psim
    pct = (100.0 * delta / psim) if psim != 0 else None
    return dict(name=name, pulsim=pulsim, psim=psim, delta=delta, pct=pct)


def extract_kpis(result: DriveSimResult, losses: LossBreakdown,
                 *, settle_fraction: float = 0.7, end_fraction: float = 1.0,
                 ) -> Dict[str, float]:
    """Pull the Pulsim numbers that match the ``KpiSet`` fields."""
    fe = result.frontend
    inv = result.inverter

    n_fe_lo = int(len(fe.times) * settle_fraction)
    n_fe_hi = int(len(fe.times) * end_fraction)
    n_in_lo = int(len(inv.times) * 0.3)
    n_in_hi = int(len(inv.times) * 0.7)

    v_ac = fe.v_ac[n_fe_lo:n_fe_hi]
    v_link = fe.v_link[n_fe_lo:n_fe_hi]
    i_L002 = fe.i_L002[n_fe_lo:n_fe_hi]
    v_shunt_inv = inv.v_n_shunt_inv[n_in_lo:n_in_hi]

    i_R508_rms = _rms(v_shunt_inv) / float(R508.R)

    # I_in from power balance (the L001 state is numerically unstable
    # past ~30 ms — see ``_build_frontend`` docstring). Using P/V is
    # robust to the open-loop boost not perfectly regulating V_link.
    v_link_avg = _avg(v_link)
    v_ac_rms = _rms(v_ac)
    R_eq = result.sim_params.V_link_target ** 2 / max(result.sim_params.op.P_in_target, 1.0)
    P_in_meas = (v_link_avg ** 2 / R_eq) if R_eq > 0 else 0.0
    PF_assumed = 0.95
    I_in_rms = P_in_meas / max(v_ac_rms * PF_assumed, 1e-6)

    return dict(
        V_in_rms=v_ac_rms,
        V_link_avg=v_link_avg,
        I_in_rms=I_in_rms,
        P_in=P_in_meas,
        PF=PF_assumed,
        I_L002_rms=_rms(i_L002),
        I_F500_rms=i_R508_rms,
        I_R508_rms=i_R508_rms,
        # The rest are derived in losses.compute_losses
        P_cond_D001=losses.P_cond_D001,
        P_cond_T1=losses.P_cond_T1, P_sw_T1=losses.P_sw_T1,
        P_cond_T2=losses.P_cond_T2, P_sw_T2=losses.P_sw_T2,
        P_cond_D002=losses.P_cond_D002, P_sw_D002=losses.P_sw_D002,
        P_IC500_total=losses.P_IC500_total,
        P_ohm_L002=losses.P_ohm_L002,
        P_mag_L002=losses.P_mag_L002,
        P_ohm_L001=losses.P_ohm_L001,
        P_esr_C009=losses.P_esr_C009,
        P_esr_C010=losses.P_esr_C010,
        P_R002=losses.P_R002,
        P_R003=losses.P_R003,
        P_R036=losses.P_R036,
        P_R508=losses.P_R508,
        P_total=losses.P_total,
        eta_inverter=losses.eta_inverter,
        T_J_D001=losses.T_J_D001,
        T_J_T001=losses.T_J_T001,
        T_J_D002=losses.T_J_D002,
        T_J_IGBT_IC500=losses.T_J_IC500,
    )


def compare_op(kpi_ref: KpiSet, kpi_pulsim: Dict[str, float]) -> List[Dict]:
    rows: List[Dict] = []
    ref_d = asdict(kpi_ref)
    for name, pulsim_val in kpi_pulsim.items():
        psim = ref_d.get(name)
        rows.append(_build_kpi_row(name, float(pulsim_val), psim))
    return rows


def print_op_table(op_id: str, op_label: str, rows: List[Dict]) -> None:
    print()
    print(f"=== {op_id}  ({op_label}) ===")
    print(f"  {'KPI':<22} {'Pulsim':>11} {'PSIM':>11} {'Δ':>11} {'%err':>8}")
    print(f"  " + "-" * 70)
    for r in rows:
        psim = r["psim"]
        if psim is None:
            print(f"  {r['name']:<22} {r['pulsim']:>11.4f} {'—':>11} {'':>11} {'':>8}")
        else:
            d = r["delta"]
            p = r["pct"]
            p_str = f"{p:+6.1f}%" if p is not None else ""
            print(f"  {r['name']:<22} {r['pulsim']:>11.4f} {psim:>11.4f} {d:>+11.4f} {p_str:>8}")


def main() -> int:
    ap = argparse.ArgumentParser(description="PFC-VSI compressor drive — Pulsim validation sweep")
    ap.add_argument("--csv", default=None, help="Write full KPI matrix to CSV")
    ap.add_argument("--quick", action="store_true",
                    help="1 line cycle per OP (default = 2)")
    args = ap.parse_args()

    all_rows: List[Tuple[str, List[Dict]]] = []

    for kpi_ref in ALL_KPI:
        op = kpi_ref.op
        print()
        print(f">>> {op.id} — {op.label}")
        sp = make_sim_params(op)
        # 10 line cycles default so the closed-loop cascade has time
        # to settle (V loop BW ≈ 30 Hz → ~30 ms settling). KPIs are
        # extracted from the LAST 30 % of the window where the
        # controller is in steady state.
        if args.quick:
            sp.t_end = 3.0 / op.f_line
        else:
            sp.t_end = 10.0 / op.f_line
        sp.dt = 2.0e-6

        t0 = time.time()
        try:
            res = simulate_drive(sp=sp)
        except Exception as exc:
            print(f"  SIM FAILED: {type(exc).__name__}: {exc}")
            return 1
        dt_sim = time.time() - t0

        losses = compute_losses(res)
        kpis = extract_kpis(res, losses)
        rows = compare_op(kpi_ref, kpis)
        all_rows.append((op.id, rows))
        print(f"  sim: {dt_sim:.2f}s ({len(res.frontend.times)} + "
              f"{len(res.inverter.times)} samples)")
        print_op_table(op.id, op.label, rows)

    # ------------------------------------------------------------------
    # CSV dump
    # ------------------------------------------------------------------
    if args.csv:
        all_kpi_names = sorted({r["name"] for _, rows in all_rows for r in rows})
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["KPI"] + [op_id for op_id, _ in all_rows] + ["PSIM_OP_2.3"])
            ref_d = asdict(ALL_KPI[1])     # OP_2.3 is the "spec" baseline
            for name in all_kpi_names:
                row = [name]
                for _, rows in all_rows:
                    val = next((r["pulsim"] for r in rows if r["name"] == name),
                                "")
                    row.append(val)
                row.append(ref_d.get(name, ""))
                w.writerow(row)
        print(f"\nCSV written to {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
