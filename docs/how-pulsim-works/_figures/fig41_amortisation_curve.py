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

    # Cumulative ms
    spice_ms  = n_steps * T_SPICE_us * 1e-3
    pulsim_ms = (visited * T_cold_us + n_steps * T_hot_us) * 1e-3

    # Variability band on T_cold: ±50%
    pulsim_ms_lo = (visited * (T_cold_us * 0.5)
                     + n_steps * T_hot_us) * 1e-3
    pulsim_ms_hi = (visited * (T_cold_us * 1.5)
                     + n_steps * T_hot_us) * 1e-3

    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    ax.loglog(n_steps, spice_ms, color="#d62728", lw=1.8,
                label="SPICE-style (re-factorise every step)")
    ax.loglog(n_steps, pulsim_ms, color="#1f77b4", lw=1.8,
                label="Pulsim PWL cache")
    ax.fill_between(n_steps, pulsim_ms_lo, pulsim_ms_hi,
                     color="#1f77b4", alpha=0.18,
                     label=r"Pulsim band (±50% on $T_{\mathrm{cold}}$)")

    # Mark the crossover point
    crossover = (visited * T_cold_us) / (T_SPICE_us - T_hot_us)
    cross_ms = crossover * T_SPICE_us * 1e-3
    ax.scatter([crossover], [cross_ms], s=60, marker="o",
                color="#222", zorder=5)
    ax.annotate(
        rf"crossover at $N_{{\mathrm{{steps}}}} \approx {crossover:.0f}$",
        xy=(crossover, cross_ms),
        xytext=(80, 0.5),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#555", lw=0.7),
    )

    # Mark the 10x asymptotic regime
    n_asymp = 1e5
    ax.axvline(n_asymp, color="#999", ls=":", lw=0.7)
    ax.text(n_asymp * 1.2, 0.001,
             rf"$N_{{\mathrm{{steps}}}}={n_asymp:.0e}$:"
             rf"  Pulsim {spice_ms[np.argmin(np.abs(n_steps - n_asymp))] / pulsim_ms[np.argmin(np.abs(n_steps - n_asymp))]:.1f}× faster",
             fontsize=9, color="#444")

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

    fig.savefig(output_dir / "fig41_amortisation_curve.png")
    fig.savefig(output_dir / "fig41_amortisation_curve.pdf")
