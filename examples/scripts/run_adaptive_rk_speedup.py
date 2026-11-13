#!/usr/bin/env python3
"""Adaptive Runge-Kutta speedup demo on the stiff van der Pol
oscillator.

Demonstrates Phase-2.4 adaptive ODE integration: ``DormandPrince5``
(non-stiff embedded RK 5(4)) and ``RadauIIA3`` (L-stable implicit)
both solve the same problem to the same final accuracy but with
**very different step counts** depending on whether the system is
stiff. A fixed-step Euler baseline is included for comparison —
it has to use much smaller ``dt`` than either adaptive method to
stay stable.

Problem
-------
Van der Pol oscillator with stiffness parameter μ:

    dx/dt = y
    dy/dt = μ · (1 − x²) · y − x

At μ = 100 the equation is **very stiff**: rapid transitions between
slow and fast time scales every cycle. Hairer & Wanner Vol. II §IV.4
uses this as the canonical stiff-ODE test.

What you should see
-------------------
* All three integrators agree on the final state to ≥ 3 sig figs.
* **Step count**: Radau IIA(3) accepts ~15-30 steps, DOPRI5
  ~50-80 steps, fixed-step Euler needs ~1000+ at the same
  tolerance.
* **Wall-clock**: on this window Radau takes ~3× fewer steps but
  about as many f evaluations (~400) as DOPRI5, and each Radau step
  pays a Newton solve, so it is NOT faster here (measured: DOPRI5
  ~3 ms, Radau ~9 ms). The window stays on the slow manifold
  (x: 2 → 1.997); the step-count gap grows on windows that cross the
  fast jumps (period ≈ (3 − 2 ln 2)·μ ≈ 160 s at μ = 100).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import pulsim as p


MU = 100.0
X0 = np.array([2.0, 0.0])
T_END = 0.5      # a short stretch of the slow manifold (the cycle is ~160 s)


def vdp(t, x):
    """Van der Pol RHS at stiffness MU."""
    return np.array([x[1], MU * (1.0 - x[0] ** 2) * x[1] - x[0]])


def fixed_step_euler(t_end, dt):
    """Forward-Euler baseline at fixed ``dt``. NOT stable at large
    ``dt`` on stiff problems — included for comparison only."""
    n_steps = int(t_end / dt)
    x = X0.copy()
    t = 0.0
    for _ in range(n_steps):
        x = x + dt * vdp(t, x)
        t += dt
    return x, n_steps


def main():
    print(f"  Van der Pol stiff oscillator (μ={MU}, t_end={T_END} s)")
    print(f"  Initial state: x={X0}")
    print()

    # --- Fixed-step Euler baseline -----------------------------------
    # Need very small dt to stay stable. At μ=100 the fast time
    # constant is ~1/(μ·|y|) — at the start (y=0) the equation is
    # mild, but during transitions y can hit 100+, requiring
    # dt < 1e-5 for stability.
    dt_fixed = 1e-5
    t0 = time.perf_counter()
    x_eul, n_eul = fixed_step_euler(T_END, dt_fixed)
    t_eul = time.perf_counter() - t0
    print(f"  Fixed-step Euler (dt={dt_fixed*1e6:.0f} µs):")
    print(f"    final state  : {x_eul}")
    print(f"    steps        : {n_eul:,}")
    print(f"    wall-clock   : {t_eul*1e3:.1f} ms")

    # --- DOPRI5 adaptive ---------------------------------------------
    dopri = p.DormandPrince5(f=vdp, rtol=1e-4, atol=1e-7)
    t0 = time.perf_counter()
    r_d = dopri.solve(t_span=(0.0, T_END), x0=X0)
    t_d = time.perf_counter() - t0
    print("\n  DormandPrince5 (rtol=1e-4, atol=1e-7):")
    print(f"    final state  : {r_d.x[-1]}")
    print(f"    steps        : {r_d.n_accepted} accepted, "
          f"{r_d.n_rejected} rejected ({r_d.n_f_evals} f evals)")
    print(f"    wall-clock   : {t_d*1e3:.1f} ms")

    # --- Radau IIA(3) adaptive ---------------------------------------
    radau = p.RadauIIA3(f=vdp, rtol=1e-4, atol=1e-7)
    t0 = time.perf_counter()
    r_r = radau.solve(t_span=(0.0, T_END), x0=X0)
    t_r = time.perf_counter() - t0
    print("\n  RadauIIA3 (rtol=1e-4, atol=1e-7):")
    print(f"    final state  : {r_r.x[-1]}")
    print(f"    steps        : {r_r.n_accepted} accepted, "
          f"{r_r.n_rejected} rejected ({r_r.n_f_evals} f evals)")
    print(f"    wall-clock   : {t_r*1e3:.1f} ms")

    # --- Summary -----------------------------------------------------
    print("\n  Summary:")
    print(f"    Euler           : {n_eul:>6} steps, "
          f"{t_eul*1e3:>6.1f} ms wall")
    print(f"    DOPRI5 adaptive : {r_d.n_accepted:>6} steps, "
          f"{t_d*1e3:>6.1f} ms wall  "
          f"({n_eul / r_d.n_accepted:.0f}× fewer steps than Euler)")
    print(f"    Radau adaptive  : {r_r.n_accepted:>6} steps, "
          f"{t_r*1e3:>6.1f} ms wall  "
          f"({n_eul / r_r.n_accepted:.0f}× fewer steps than Euler)")
    final_diff_dopri = np.linalg.norm(r_d.x[-1] - r_r.x[-1])
    print(f"    DOPRI5 vs Radau : final states agree to "
          f"{final_diff_dopri:.2e}")

    # --- Optional CSV trace -----------------------------------------
    try:
        import csv
        out_path = Path(__file__).with_name("adaptive_rk_trace.csv")
        with out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "x_dopri", "y_dopri",
                            "dt_dopri", "step_idx_dopri"])
            # DOPRI5 trace (the ``dt`` chosen for each accepted step
            # exposes the controller's adaptive behaviour — long dt
            # during slow regions, short dt during transitions).
            dt_pad = list(r_d.dt_history) + [float("nan")]
            for i in range(len(r_d.t)):
                w.writerow([r_d.t[i], r_d.x[i, 0], r_d.x[i, 1],
                                dt_pad[i] if i < len(dt_pad) else float("nan"),
                                i])
        print(f"    trace        → {out_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"    (CSV export skipped: {exc})")


if __name__ == "__main__":
    main()
