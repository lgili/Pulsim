#!/usr/bin/env python3
"""Compare MMC L0 (continuous m_b) vs L1 (PS-PWM s_b) on a single arm.

Drives both layers with the same modulation reference and arm current,
then plots:

  1. ``m_b(t)`` continuous reference vs the L1 staircase ``s_b/N``.
  2. ``v_b(t)`` — smooth (L0) overlaid on the multilevel staircase (L1).
  3. ``v_C(t)`` — L0 envelope vs L1 (the two should track within one
     PS-PWM bit, since the dynamics are identical eqs 2.9 / 2.14 once
     m_b is replaced by s_b/N).

This is the cleanest visualization of "L1 = L0 + multilevel switching"
that the design doc described. Useful both as a design check and as a
teaching aid for what PS-PWM looks like on a real MMC arm.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pulsim as p


N_SM      = 8                # SMs per arm
C_SM      = 5e-3             # F → C_arm = 625 µF
V_C0      = 800.0
F_CARRIER = 500.0            # Hz per carrier → 4 kHz effective
F_REF     = 50.0             # AC fundamental (modulation)
M_DEPTH   = 0.8
I_HAT     = 40.0             # arm current peak
OMEGA_REF = 2.0 * math.pi * F_REF

DT        = 5e-6             # 5 µs → 50 samples per carrier period
T_END     = 30e-3            # 1.5 modulation periods


def m_ref(t: float) -> float:
    return 0.5 + 0.5 * M_DEPTH * math.sin(OMEGA_REF * t)


def i_b_fn(t: float) -> float:
    # 90° lag so cap doesn't drift on average.
    return I_HAT * math.sin(OMEGA_REF * t - math.pi / 2.0)


def main() -> None:
    print("Running L0 (continuous) and L1 (PS-PWM) on the same arm...")
    params_l0 = p.MmcArmAverageParams(n_sm=N_SM, c_sm=C_SM, v_c0=V_C0)
    params_l1 = p.MmcArmMultilevelParams(
        n_sm=N_SM, c_sm=C_SM, v_c0=V_C0, f_carrier=F_CARRIER,
    )

    res_l0 = p.simulate_mmc_arm_average(
        duration=T_END, dt=DT, m_b=m_ref, i_b=i_b_fn, params=params_l0,
    )
    res_l1 = p.simulate_mmc_arm_multilevel(
        duration=T_END, dt=DT, m_ref=m_ref, i_b=i_b_fn, params=params_l1,
    )

    # Summary numbers.
    print(f"  N_SM       = {N_SM}")
    print(f"  C_SM       = {C_SM*1e3:.2f} mF (C_arm = {C_SM/N_SM*1e6:.0f} µF)")
    print(f"  f_carrier  = {F_CARRIER:.0f} Hz "
          f"(f_switch = {N_SM*F_CARRIER:.0f} Hz)")
    print(f"  f_ref      = {F_REF:.0f} Hz, depth = {M_DEPTH:.2f}")
    print(f"  I_hat      = {I_HAT:.1f} A")
    print()
    print(f"L0 v_C  range: {res_l0.v_C.min():.2f} → {res_l0.v_C.max():.2f} V")
    print(f"L1 v_C  range: {res_l1.v_C.min():.2f} → {res_l1.v_C.max():.2f} V")
    print(f"L1 s_b  range: {res_l1.s_b.min()} → {res_l1.s_b.max()} "
          f"(of 0..{N_SM})")
    diff_v_C = np.max(np.abs(res_l1.v_C - res_l0.v_C))
    print(f"max |L1 v_C - L0 v_C| = {diff_v_C:.4f} V "
          f"(one PS-PWM bit = "
          f"{(res_l0.v_C.max() / N_SM):.3f} V at this v_C scale)")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping plot)")
        return

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    # Panel 1: m_b reference vs staircase s_b/N.
    axes[0].plot(res_l0.t * 1e3, res_l0.m_b,
                     label="m_b (L0 continuous)", linewidth=1.2)
    axes[0].plot(res_l1.t * 1e3, res_l1.s_b / N_SM,
                     drawstyle="steps-post",
                     label=f"s_b / N (L1, N={N_SM})", linewidth=0.8,
                     alpha=0.7)
    axes[0].set_ylabel("modulation index")
    axes[0].set_title(
        f"PS-PWM quantization (f_carrier={F_CARRIER:.0f} Hz, "
        f"f_switch={N_SM*F_CARRIER:.0f} Hz)"
    )
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: v_b smooth vs staircase.
    axes[1].plot(res_l0.t * 1e3, res_l0.v_b,
                     label="v_b (L0 smooth)", linewidth=1.4)
    axes[1].plot(res_l1.t * 1e3, res_l1.v_b,
                     drawstyle="steps-post",
                     label="v_b (L1 multilevel)", linewidth=0.7,
                     alpha=0.7)
    axes[1].set_ylabel("v_b [V]")
    axes[1].set_title("Arm-generated voltage (smooth vs staircase)")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: v_C envelope agreement.
    axes[2].plot(res_l0.t * 1e3, res_l0.v_C,
                     label="v_C (L0)", linewidth=1.4)
    axes[2].plot(res_l1.t * 1e3, res_l1.v_C,
                     label="v_C (L1)", linewidth=0.9, alpha=0.8)
    axes[2].axhline(V_C0, color="k", linestyle="--", alpha=0.3,
                       label=f"v_c0 = {V_C0:.0f} V")
    axes[2].set_xlabel("time [ms]")
    axes[2].set_ylabel("v_C [V]")
    axes[2].set_title("Capacitor-sum voltage (L0 envelope vs L1)")
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()

    out = Path(__file__).resolve().parent / "out" / "mmc_l0_vs_l1.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Plot saved: {out}")


if __name__ == "__main__":
    main()
