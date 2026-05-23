#!/usr/bin/env python3
"""Parameter sweep — buck converter output ripple vs (L, C).

Sweeps a regular grid of (L, C) values, runs an open-loop buck
simulation for each, and extracts the peak-to-peak output ripple
in steady state. The result is a 2-D contour that tells you which
(L, C) combinations meet a given ripple target.

The same `p.sweep(...)` API works for any KPI you can compute from
a SimulationResult, so this scales to multi-dimensional design
exploration (load, switching frequency, dead-time, controller gains,
etc.).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pulsim as p


V_IN  = 24.0
F_PWM = 100e3
T_PWM = 1.0 / F_PWM
DUTY  = 0.5
R_LOAD = 5.0
T_END = 1.5e-3
DT    = 1.0e-7


def build(L: float, C: float) -> p.CircuitBuilder:
    """Buck plant parametrized by L and C."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", V_IN)
    b.add_switch("Q1", "vin", "sw", g_on=1e3, g_off=1e-9)
    b.add_diode("D_FW", "gnd", "sw", g_on=1e3, g_off=1e-9, V_th=0.7)
    b.add_inductor("L1", "sw", "vout", L)
    b.add_capacitor("Cout", "vout", "gnd", C)
    b.add_resistor("R_load", "vout", "gnd", R_LOAD)
    return b


def make_switch_fn(num_switches: int):
    """Fixed 50 % duty PWM on switch 0."""
    def sw_fn(t: float):
        phase = math.fmod(t, T_PWM) / T_PWM
        m = p.SwitchStateMask(num_switches)
        if phase < DUTY:
            m.set(0, True)
        return m
    return sw_fn


def ripple_kpi(res, params) -> dict:
    """Extract peak-to-peak V_out ripple in the last 0.3 ms (≈30 PWM
    cycles at 100 kHz) AND the mean output voltage."""
    times = np.asarray(res.times)
    states = np.asarray(res.states)
    # vout node is added third — use builder's node_id_of via a
    # closure if you want exact tracking. Here it's index 2 (after
    # vin → sw → vout for this topology).
    # We pull vout dynamically by constructing a dummy builder.
    b_dummy = build(params["L"], params["C"])
    vout_idx = b_dummy.node_id_of("vout")
    v_out = states[:, vout_idx]
    tail_mask = times >= (times[-1] - 0.3e-3)
    v_tail = v_out[tail_mask]
    return {
        "v_ripple_pp": float(v_tail.max() - v_tail.min()),
        "v_mean":      float(v_tail.mean()),
    }


def main() -> None:
    # 5 × 5 grid → 25 simulations. Each sim is ~10 ms wall → ~250 ms
    # total. Demonstrates sweep without needing parallel.
    L_grid = np.geomspace(50e-6, 500e-6, 5)
    C_grid = np.geomspace(10e-6, 200e-6, 5)
    print(f"  L: {[f'{l*1e6:.0f}µH' for l in L_grid]}")
    print(f"  C: {[f'{c*1e6:.0f}µF' for c in C_grid]}")

    out = p.sweep(
        build,
        params={"L": L_grid, "C": C_grid},
        kpi_fn=ripple_kpi,
        t_end=T_END, dt=DT,
        switch_fn=make_switch_fn(2),
        max_event_iterations=8,
    )

    print(f"\n  Sweep summary:")
    print(f"    n_simulations: {out.n_simulations}")
    print(f"    wall_time:     {out.wall_time_s:.2f} s")
    print(f"    failed runs:   {len(out.failed)}")
    print(f"    KPIs:          {list(out.kpis.keys())}")

    # Reshape to 2D grids for contouring.
    ripple = out.kpis["v_ripple_pp"].reshape(out.shape)
    v_mean = out.kpis["v_mean"].reshape(out.shape)

    print(f"\n  Ripple [mV peak-to-peak] grid (rows=L, cols=C):")
    print(f"    L\\C    {'  '.join(f'{c*1e6:>5.0f}µF' for c in C_grid)}")
    for i, l in enumerate(L_grid):
        row = "  ".join(f"{ripple[i, j]*1e3:>6.1f}"
                          for j in range(len(C_grid)))
        print(f"    {l*1e6:>3.0f}µH  {row}")

    print(f"\n  V_mean [V] grid (target {V_IN*DUTY:.1f} V):")
    print(f"    L\\C    {'  '.join(f'{c*1e6:>5.0f}µF' for c in C_grid)}")
    for i, l in enumerate(L_grid):
        row = "  ".join(f"{v_mean[i, j]:>6.2f}"
                          for j in range(len(C_grid)))
        print(f"    {l*1e6:>3.0f}µH  {row}")

    # Plot.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, (ax_r, ax_v) = plt.subplots(1, 2, figsize=(13, 5))

    L_mesh, C_mesh = np.meshgrid(L_grid*1e6, C_grid*1e6, indexing="ij")
    cs_r = ax_r.contourf(C_mesh, L_mesh, ripple*1e3, levels=15,
                            cmap="viridis")
    plt.colorbar(cs_r, ax=ax_r, label="V_out ripple p-p [mV]")
    ax_r.set_xscale("log"); ax_r.set_yscale("log")
    ax_r.set_xlabel("C [µF]"); ax_r.set_ylabel("L [µH]")
    ax_r.set_title("Buck output ripple vs (L, C)")

    cs_v = ax_v.contourf(C_mesh, L_mesh, v_mean, levels=15,
                            cmap="plasma")
    plt.colorbar(cs_v, ax=ax_v, label="V_out mean [V]")
    ax_v.set_xscale("log"); ax_v.set_yscale("log")
    ax_v.set_xlabel("C [µF]"); ax_v.set_ylabel("L [µH]")
    ax_v.set_title(f"Mean V_out (target {V_IN*DUTY:.1f} V)")

    plt.tight_layout()
    out_path = Path(__file__).resolve().parent / "output" / "sweep_buck_lc.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"\n  plot → {out_path}")


if __name__ == "__main__":
    main()
