"""Figure 4.2 — Cache amortisation curve.

Total wall-clock time vs simulated step count for two policies:

  * SPICE-style: cost ~= N_steps * T_SPICE  (linear, no upfront)
  * Pulsim cache:   cost ~= |visited|*T_cold + N_steps * T_hot
                          (constant offset, much smaller slope)

The crossover point is at ~10 steps; from there Pulsim widens
its lead linearly. By 10^4 steps Pulsim is ~10x ahead.

Parameters reflect the representative per-converter numbers
documented in chapter 4 §4.3 (T_cold = 100 us, T_hot = 5 us,
T_SPICE = 50 us, |visited| = 4). The shaded band shows ±50% on
T_cold to reflect cross-converter variability.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render(output_dir: Path) -> None:
    # Per-step costs (microseconds)
    T_SPICE_us = 50.0
    T_hot_us   = 5.0
    visited    = 4
    T_cold_us  = 100.0

    n_steps = np.logspace(0, 6, 200)  # 1 to 1e6

    spice_ms  = n_steps * T_SPICE_us * 1e-3
    pulsim_ms = (visited * T_cold_us + n_steps * T_hot_us) * 1e-3
    pulsim_ms_lo = (visited * (T_cold_us * 0.5)
                     + n_steps * T_hot_us) * 1e-3
    pulsim_ms_hi = (visited * (T_cold_us * 1.5)
                     + n_steps * T_hot_us) * 1e-3

    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    ax.loglog(n_steps, spice_ms, color="#d62728", lw=1.8,
                label="SPICE-style (re-factorise every step)")
    ax.loglog(n_steps, pulsim_ms, color="#1f77b4", lw=1.8,
                label="Pulsim PWL cache")
    ax.fill_between(n_steps, pulsim_ms_lo, pulsim_ms_hi,
                     color="#1f77b4", alpha=0.18,
                     label=r"Pulsim band (±50% on $T_{\mathrm{cold}}$)")

    # Crossover marker + on-plot annotation
    crossover = (visited * T_cold_us) / (T_SPICE_us - T_hot_us)
    cross_ms = crossover * T_SPICE_us * 1e-3
    ax.scatter([crossover], [cross_ms], s=70, marker="o",
                color="#222", zorder=6, edgecolor="white",
                linewidth=1.2)
    ax.annotate(
        rf"crossover  $\approx$ {crossover:.0f} steps",
        xy=(crossover, cross_ms),
        xytext=(crossover * 4, cross_ms / 8),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#555", lw=0.7),
    )

    # Speedup annotation at N = 1e5 — anchored to the actual
    # SPICE/Pulsim curves so it's never off-plot
    n_asymp_idx = np.argmin(np.abs(n_steps - 1e5))
    n_asymp_val = n_steps[n_asymp_idx]
    spice_at_asymp  = spice_ms[n_asymp_idx]
    pulsim_at_asymp = pulsim_ms[n_asymp_idx]
    ratio = spice_at_asymp / pulsim_at_asymp

    ax.axvline(n_asymp_val, color="#999", ls=":", lw=0.7)
    # Vertical span connecting the two points at N=1e5
    ax.plot([n_asymp_val, n_asymp_val],
              [pulsim_at_asymp, spice_at_asymp],
              color="#888", lw=1.0, alpha=0.6)
    # Centered between them on log-y
    midpoint_y = np.sqrt(pulsim_at_asymp * spice_at_asymp)
    ax.text(n_asymp_val * 1.2, midpoint_y,
              rf"$\bf{{{ratio:.1f}\times}}$"
              "\n"
              rf"@ $N=10^5$",
              fontsize=10, color="#1f77b4",
              ha="left", va="center", weight="normal",
              linespacing=1.4)

    ax.set_xlabel(r"Simulated step count $N_{\mathrm{steps}}$")
    ax.set_ylabel("Cumulative wall-clock time (ms)")
    ax.set_title("PWL cache amortisation: cumulative cost vs simulation length\n"
                 r"(per-step parameters: $T_{\mathrm{cold}} = 100\,\mu$s,"
                 r" $T_{\mathrm{hot}} = 5\,\mu$s,"
                 r" $T_{\mathrm{SPICE}} = 50\,\mu$s,"
                 r" $|\mathrm{visited}| = 4$)",
                 pad=8)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "fig41_amortisation_curve.png")
    fig.savefig(output_dir / "fig41_amortisation_curve.pdf")
