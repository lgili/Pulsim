#!/usr/bin/env python3
"""Adaptive transient — RL step response, dt grows by 100× toward steady state.

Plant: V_in (1 V step) → R (0.1 Ω) → L (1 mH) → gnd.
Time constant τ = L/R = 10 ms.
Sim horizon: 500 ms (50 τ — way more than needed).

A fixed-dt solver with dt = 1 µs needs 500 000 steps. With variable
dt, the driver picks tiny steps during the early exponential rise
and grows dt by 10–100× once the state has settled — typically 10×
fewer total steps with the same accuracy vs analytic.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pulsim.v2 as p


R = 0.1
L = 1e-3
TAU = L / R          # 10 ms
V_IN = 1.0
T_END = 500e-3


def build_rl() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", V_IN)
    b.add_resistor("R", "vin", "vout", R)
    b.add_inductor("L", "vout", "gnd", L)
    return b


def analytic_i(t):
    """i(t) = (V_in / R) · (1 − e^{−t/τ})"""
    return (V_IN / R) * (1.0 - np.exp(-t / TAU))


def main() -> None:
    b = build_rl()
    # The inductor branch-current state is the one we want to compare
    # against analytic i(t).
    i_idx = b.pool.branch_var_id_for_inductor(
        b.pool.last_branch_id_for_inductor("L"), b.graph) \
            if hasattr(b.pool, "last_branch_id_for_inductor") else \
            b.pool.branch_var_id_for_inductor(2, b.graph)

    print(f"  RL settling: R={R} Ω, L={L*1e3} mH, τ={TAU*1e3} ms")
    print(f"  Horizon: {T_END*1e3} ms ({T_END/TAU:.0f}·τ)")

    res = p.run_transient_adaptive(
        b,
        t_start=0.0, t_end=T_END,
        dt_init=1e-6, dt_min=1e-7, dt_max=200e-6,
        atol=1e-5, rtol=1e-4,
        segment_steps=200,
        verbose=False,
    )

    times = np.asarray(res.times)
    i_num = res.states[:, i_idx]
    i_ref = analytic_i(times)
    err = np.abs(i_num - i_ref)
    rel_err = err / (np.abs(i_ref) + 1e-12)

    n_samples_fixed = int(T_END / 1e-6)  # equivalent fixed-dt count
    print(f"\n  Adaptive transient KPI:")
    print(f"    segments:        {len(res.dt_history)}")
    print(f"    samples kept:    {res.num_steps()} "
          f"(vs ~{n_samples_fixed} for fixed-dt at 1 µs)")
    print(f"    speedup:         "
          f"{n_samples_fixed / res.num_steps():.1f}×")
    print(f"    initial dt:      {res.dt_history[0]*1e6:.2f} µs")
    print(f"    final dt:        {res.dt_history[-1]*1e6:.2f} µs")
    print(f"    growth ratio:    "
          f"{res.dt_history[-1] / res.dt_history[0]:.1f}×")
    print(f"    accepted / rej:  {res.n_accepted} / {res.n_rejected}")
    print(f"    max |i − i_an|:  {err.max():.3e} A")
    print(f"    max rel err:     {rel_err.max():.3e}")

    # Plot
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, (ax_i, ax_dt, ax_err) = plt.subplots(3, 1, figsize=(11, 8),
                                                  sharex=True)
    t_ms = times * 1e3
    ax_i.plot(t_ms, i_num, "C0-", lw=1.2, label="i (adaptive)")
    ax_i.plot(t_ms, i_ref, "k--", lw=0.8, label="i (analytic)")
    ax_i.set_ylabel("i [A]"); ax_i.grid(alpha=0.3)
    ax_i.legend(loc="lower right")
    ax_i.set_title(f"RL adaptive transient — "
                     f"R={R} Ω, L={L*1e3} mH, τ={TAU*1e3} ms")

    # Reconstruct per-sample dt from times.
    dt_per_sample = np.diff(times, prepend=times[0])
    ax_dt.semilogy(t_ms, dt_per_sample * 1e6, "C1-", lw=0.7)
    ax_dt.set_ylabel("dt [µs]"); ax_dt.grid(alpha=0.3, which="both")

    ax_err.semilogy(t_ms, err + 1e-15, "C3-", lw=0.7,
                       label="|i − i_an|")
    ax_err.set_xlabel("time [ms]"); ax_err.set_ylabel("error [A]")
    ax_err.grid(alpha=0.3, which="both")

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "adaptive_rl_settle.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\n  plot → {out}")


if __name__ == "__main__":
    main()
