#!/usr/bin/env python3
"""Sliding-Mode Observer (SMO) sensorless rotor-angle estimation —
standalone demo using a synthetic PMSM stator-voltage / current
trace.

Why standalone (not coupled to ``pulsim.simulate``)?
---------------------------------------------------
A real PMSM driven open-loop by a 3-phase voltage source has to
*pull into* synchronism from standstill — without a v/f ramp or
a closed-loop controller this either takes longer than the demo
window or fails outright. Demonstrating the SMO requires a known
steady-state operating point, so this script generates the stator
trace analytically from a rotor spinning at a fixed speed and
feeds it to the SMO — the same pattern used by
``test_smo_locks_onto_synthetic_back_emf`` in the pytest suite,
just with progress reporting and a CSV export for plotting.

What the demo shows
-------------------
* The SMO's estimated electrical angle ``θ̂_e`` locks onto the
  true rotor angle ``θ_e_true`` within ~100 ms.
* Steady-state angle error stays within 10°-20° even with the
  saturating-sign chatter suppression.
* Estimated electrical speed ``ω̂_e`` matches the imposed
  ω_e_true within 1 %.

For a fully integrated FOC drive that uses the SMO to feed the
Park transform's angle input, see ``run_pmsm_foc.py`` (sensored)
+ swap ``RotorAngleFromMotor`` with the SMO output. That
end-to-end closed-loop demo is queued for v1.5.1 (waits on the
``BlockChain``-block-form SMO wrapper).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pulsim as p


# ---------- PMSM parameters ------------------------------------------------

R_S      = 0.5
L_S      = 200e-6
PSI_PM   = 0.01
PP       = 4
OMEGA_M  = 200.0            # rotor mech speed [rad/s]
OMEGA_E  = PP * OMEGA_M     # electrical speed [rad/s]
F_E_HZ   = OMEGA_E / (2.0 * math.pi)
E_PEAK   = PP * PSI_PM * OMEGA_M   # back-EMF peak [V]
I_Q_REF  = 1.0              # synthetic q-axis current

DT       = 1.0e-5           # 10 µs — well below the PLL bandwidth
T_END    = 0.2              # 200 ms — PLL settles in ~50 ms


def main():
    print(f"  Synthetic PMSM trace:")
    print(f"    ω_m_true  = {OMEGA_M:.2f} rad/s")
    print(f"    ω_e_true  = {OMEGA_E:.2f} rad/s ({F_E_HZ:.1f} Hz elec)")
    print(f"    E_peak    = {E_PEAK:.4f} V")
    print(f"    i_q_ref   = {I_Q_REF:.2f} A")

    smo = p.SlidingModeObserver(
        Rs=R_S, Ls=L_S,
        K_sl=2.0 * max(E_PEAK, 1.0),
        f_init_hz=F_E_HZ,
        omega_lpf=2.0 * math.pi * 2000.0,
    )

    n_steps = int(T_END / DT)
    log_t = np.empty(n_steps)
    log_theta_true = np.empty(n_steps)
    log_theta_hat = np.empty(n_steps)
    log_omega_hat = np.empty(n_steps)
    log_err = np.empty(n_steps)

    theta_e = 0.0
    for k in range(n_steps):
        # Advance true rotor angle.
        theta_e += OMEGA_E * DT
        if theta_e >= 2.0 * math.pi:
            theta_e -= 2.0 * math.pi
        c, s = math.cos(theta_e), math.sin(theta_e)
        # Stator currents in αβ (Park-inverse from i_d=0, i_q=I_Q_REF).
        i_alpha = -I_Q_REF * s
        i_beta  = +I_Q_REF * c
        # Back-EMF in αβ (PMSM convention).
        e_alpha = -E_PEAK * s
        e_beta  = +E_PEAK * c
        # Voltage that would produce the assumed steady-state currents.
        v_alpha = R_S * i_alpha + e_alpha
        v_beta  = R_S * i_beta  + e_beta
        # Run the observer.
        out = smo.update(
            v_alpha=v_alpha, v_beta=v_beta,
            i_alpha=i_alpha, i_beta=i_beta,
            dt=DT)
        theta_hat, omega_hat = out[0], out[1]
        # Log every step.
        log_t[k] = k * DT
        log_theta_true[k] = theta_e
        log_theta_hat[k] = theta_hat
        log_omega_hat[k] = omega_hat
        # Wrapped angle error in [-π, π].
        err = theta_hat - theta_e
        log_err[k] = math.atan2(math.sin(err), math.cos(err))

    # Steady-state window: drop first 50 ms (PLL transient).
    steady = log_t > 0.05
    err_ss = log_err[steady]
    err_peak = float(np.max(np.abs(err_ss)))
    err_rms = float(np.sqrt(np.mean(err_ss ** 2)))
    omega_hat_ss = float(np.mean(log_omega_hat[steady]))
    omega_err_pct = (omega_hat_ss - OMEGA_E) / OMEGA_E * 100.0

    print(f"\n  SMO performance (after 50 ms lock-in window):")
    print(f"    θ̂ error  : peak {math.degrees(err_peak):.2f}°, "
          f"RMS {math.degrees(err_rms):.2f}°")
    print(f"    ω̂        : {omega_hat_ss:.2f} rad/s "
          f"(vs ω_e_true = {OMEGA_E:.2f}, err = {omega_err_pct:+.2f}%)")
    print(f"    n_samples : {n_steps:,}")

    # Save CSV for the user to plot.
    try:
        import csv
        out_path = Path(__file__).with_name(
            "pmsm_smo_sensorless_trace.csv")
        stride = max(1, n_steps // 4000)
        with out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "theta_true_rad", "theta_hat_rad",
                            "theta_err_rad", "omega_hat_rad_s"])
            for i in range(0, n_steps, stride):
                w.writerow([
                    log_t[i], log_theta_true[i],
                    log_theta_hat[i], log_err[i],
                    log_omega_hat[i]])
        print(f"    trace     → {out_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"    (CSV export skipped: {exc})")


if __name__ == "__main__":
    main()
