#!/usr/bin/env python3
"""Compare MMC L1 (PS-PWM only) vs L2 (PS-PWM + dead-time) on a
single arm under a fast modulation step.

Reproduces the spirit of Sousa thesis fig 3.12: a sharp ``m_ref``
ramp triggers a burst of per-SM toggles, and the L2 model shows
the characteristic ``t_dead``-wide free-wheel "notches" in the arm
voltage that L1 misses entirely.

We hold ``i_b`` positive throughout so free-wheel SMs are *bypassed*
(D2 conducts), and the dead-time notches *subtract* from v_b. Flip
``I_B`` to a negative value to see the symmetric "inserted-during-
dead-time" behaviour where notches *add* to v_b instead.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pulsim as p


N_SM      = 4
C_SM      = 5e-3
V_C0      = 800.0
F_CARRIER = 1000.0
T_DEAD    = 30e-6              # 30 µs IGBT-class dead-time
I_B       = 10.0               # positive — free-wheel SMs are bypassed

DT        = 1e-6               # 1 µs → resolves the dead-time well
T_END     = 4e-3               # 4 carrier periods


def m_ref(t: float) -> float:
    """Ramp m_ref from 0.2 to 0.9 over 0.5 ms starting at t = 1 ms."""
    if t < 1e-3:
        return 0.2
    if t > 1.5e-3:
        return 0.9
    return 0.2 + (0.9 - 0.2) * (t - 1e-3) / 0.5e-3


def main() -> None:
    print("Running L1 (PS-PWM only) and L2 (PS-PWM + dead-time)...")
    params_l1 = p.MmcArmMultilevelParams(
        n_sm=N_SM, c_sm=C_SM, v_c0=V_C0, f_carrier=F_CARRIER,
    )
    params_l2 = p.MmcArmEquivalentParams(
        n_sm=N_SM, c_sm=C_SM, v_c0=V_C0,
        f_carrier=F_CARRIER, t_dead=T_DEAD,
    )

    res_l1 = p.simulate_mmc_arm_multilevel(
        duration=T_END, dt=DT, m_ref=m_ref, i_b=I_B, params=params_l1,
    )
    res_l2 = p.simulate_mmc_arm_equivalent(
        duration=T_END, dt=DT, m_ref=m_ref, i_b=I_B, params=params_l2,
    )

    n_freewheel_samples = int((res_l2.s_u > 0).sum())
    freewheel_duty = n_freewheel_samples / len(res_l2.s_u)
    print(f"  N_SM       = {N_SM}")
    print(f"  C_SM       = {C_SM*1e3:.1f} mF "
          f"(C_arm = {C_SM/N_SM*1e6:.0f} µF)")
    print(f"  f_carrier  = {F_CARRIER:.0f} Hz "
          f"(f_switch = {N_SM*F_CARRIER:.0f} Hz)")
    print(f"  t_dead     = {T_DEAD*1e6:.1f} µs")
    print(f"  I_B        = {I_B:.1f} A")
    print()
    print(f"L2 free-wheel duty = {freewheel_duty*100:.2f} % "
          f"({n_freewheel_samples} / {len(res_l2.s_u)} samples)")
    diff = np.max(np.abs(res_l1.v_b - res_l2.v_b))
    print(f"max |v_b L1 - v_b L2| = {diff:.2f} V  "
          f"(one PS-PWM bit = {V_C0/N_SM:.2f} V)")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping plot)")
        return

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    # Panel 1: m_ref reference.
    axes[0].plot(res_l1.t * 1e3, res_l1.m_ref, color="C0",
                     label="m_ref(t)", linewidth=1.4)
    axes[0].set_ylabel("m_ref")
    axes[0].set_title(
        f"Modulation reference (fast ramp 0.2 → 0.9 over 0.5 ms)"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=9)

    # Panel 2: arm voltage v_b — L1 vs L2.
    axes[1].plot(res_l1.t * 1e3, res_l1.v_b,
                     drawstyle="steps-post",
                     label="v_b (L1: no dead-time)",
                     linewidth=1.0, alpha=0.7)
    axes[1].plot(res_l2.t * 1e3, res_l2.v_b,
                     drawstyle="steps-post",
                     label=f"v_b (L2: t_dead={T_DEAD*1e6:.0f} µs)",
                     linewidth=0.9, alpha=0.85, color="C3")
    axes[1].set_ylabel("v_b [V]")
    axes[1].set_title(
        f"Arm voltage — L2 'notches' during dead-time (i_b > 0 ⇒ "
        f"SMs bypassed)"
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=9)

    # Panel 3: L2 internal s_w / s_u counts.
    axes[2].plot(res_l2.t * 1e3, res_l2.s_w,
                     drawstyle="steps-post", label="s_w (defined inserted)",
                     linewidth=1.0, color="C2")
    axes[2].plot(res_l2.t * 1e3, res_l2.s_u,
                     drawstyle="steps-post",
                     label="s_u (free-wheel)",
                     linewidth=1.0, color="C3")
    axes[2].set_xlabel("time [ms]")
    axes[2].set_ylabel("count")
    axes[2].set_yticks(range(N_SM + 1))
    axes[2].set_title("L2 internal SM-state counts")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right", fontsize=9)

    fig.tight_layout()

    out = Path(__file__).resolve().parent / "out" / "mmc_l1_vs_l2_deadtime.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Plot saved: {out}")


if __name__ == "__main__":
    main()
