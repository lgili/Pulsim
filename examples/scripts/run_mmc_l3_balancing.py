#!/usr/bin/env python3
"""L3 MMC arm — sort-and-select balancing vs no balancing.

Starts from a skewed per-SM capacitor distribution (some SMs charged
much higher than the mean) and drives the arm with realistic AC
modulation + arm current. The L3 detailed model is run twice:

  1. **With balancing** (sort-and-select, Tu/Hu/Xu 2011): the
     algorithm chooses *which* SMs to insert based on their
     individual v_C_n values and the sign of i_b. The spread
     between SMs collapses over a few switching periods.

  2. **Without balancing** (``balancing="none"``, fixed
     assignment): no closed-loop control of SM voltages. The
     spread persists — and on a converter with even small
     parameter mismatches it would diverge over time.

Plots:
  * Per-SM trajectories under balancing → visibly converge.
  * Per-SM trajectories without balancing → stay separated.
  * Peak-to-peak spread vs time → balancing crushes it.

This is the standard visualization of why MMC implementations need
balancing control. Matches the spirit of Sousa thesis sec 5.1.1.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pulsim as p


N_SM      = 6
C_SM      = 5e-3            # F
V_C0_SUM  = 600.0           # nominal arm-sum [V]
F_CARRIER = 1000.0          # Hz
F_REF     = 50.0            # AC fundamental
M_DEPTH   = 0.8
I_HAT     = 30.0            # arm-current peak

DT        = 5e-6
T_END     = 80e-3           # 4 fundamental periods


# Modulation reference + arm current — a generic AC-side operating point.
OMEGA_REF = 2.0 * math.pi * F_REF


def m_ref(t: float) -> float:
    return 0.5 + 0.5 * M_DEPTH * math.sin(OMEGA_REF * t)


def i_b_fn(t: float) -> float:
    # Quadrature (90° lag) so cap energy oscillates without net drift.
    return I_HAT * math.cos(OMEGA_REF * t)


def main() -> None:
    # Skewed initial state: equally spaced from 80 V to 120 V (mean = 100 V).
    init_v = np.linspace(80.0, 120.0, N_SM)
    initial_spread = init_v.max() - init_v.min()
    print(
        f"Running L3 balancing comparison "
        f"(N={N_SM}, initial spread = {initial_spread:.1f} V)..."
    )

    p_balanced = p.MmcArmDetailedParams(
        n_sm=N_SM, c_sm=C_SM, f_carrier=F_CARRIER,
        balancing="sort_and_select",
    )
    p_uncontrolled = p.MmcArmDetailedParams(
        n_sm=N_SM, c_sm=C_SM, f_carrier=F_CARRIER,
        balancing="none",
    )

    res_bal = p.simulate_mmc_arm_detailed(
        duration=T_END, dt=DT,
        m_ref=m_ref, i_b=i_b_fn, params=p_balanced,
        initial_v_C_per_sm=init_v,
    )
    res_unc = p.simulate_mmc_arm_detailed(
        duration=T_END, dt=DT,
        m_ref=m_ref, i_b=i_b_fn, params=p_uncontrolled,
        initial_v_C_per_sm=init_v,
    )

    # Report key numbers.
    last_period_idx = res_bal.t >= (T_END - 1.0 / F_REF)
    bal_final_spread = float(np.median(res_bal.v_C_spread[last_period_idx]))
    unc_final_spread = float(np.median(res_unc.v_C_spread[last_period_idx]))
    print(f"  initial spread          = {initial_spread:.2f} V")
    print(f"  final spread (balanced) = {bal_final_spread:.2f} V "
          f"  ({bal_final_spread / initial_spread * 100:.1f} % of initial)")
    print(f"  final spread (no bal.)  = {unc_final_spread:.2f} V "
          f"  ({unc_final_spread / initial_spread * 100:.1f} % of initial)")
    print()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping plot)")
        return

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    ts_ms = res_bal.t * 1e3

    # Panel 1: per-SM v_C with balancing.
    for n in range(N_SM):
        axes[0].plot(ts_ms, res_bal.v_C_per_sm[:, n],
                         linewidth=0.9, alpha=0.85)
    axes[0].axhline(V_C0_SUM / N_SM, color="k", linestyle="--",
                       alpha=0.4, label=f"mean = {V_C0_SUM/N_SM:.0f} V")
    axes[0].set_ylabel("v_C_n [V]")
    axes[0].set_title(
        f"L3 per-SM voltages — sort-and-select balancing ON "
        f"(N={N_SM}, f_switch={N_SM*F_CARRIER:.0f} Hz)"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9, loc="upper right")

    # Panel 2: per-SM v_C without balancing.
    for n in range(N_SM):
        axes[1].plot(ts_ms, res_unc.v_C_per_sm[:, n],
                         linewidth=0.9, alpha=0.85)
    axes[1].axhline(V_C0_SUM / N_SM, color="k", linestyle="--",
                       alpha=0.4, label=f"mean = {V_C0_SUM/N_SM:.0f} V")
    axes[1].set_ylabel("v_C_n [V]")
    axes[1].set_title(
        "L3 per-SM voltages — balancing OFF (fixed assignment)"
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9, loc="upper right")

    # Panel 3: pk-pk spread vs time.
    axes[2].plot(ts_ms, res_bal.v_C_spread,
                     label="sort-and-select", linewidth=1.3, color="C2")
    axes[2].plot(ts_ms, res_unc.v_C_spread,
                     label="none", linewidth=1.3, color="C3")
    axes[2].axhline(initial_spread, color="k", linestyle="--",
                       alpha=0.4, label=f"initial = {initial_spread:.0f} V")
    axes[2].set_xlabel("time [ms]")
    axes[2].set_ylabel("v_C spread (pk-pk) [V]")
    axes[2].set_title("Per-SM voltage spread")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "out" / "mmc_l3_balancing.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Plot saved: {out}")


if __name__ == "__main__":
    main()
