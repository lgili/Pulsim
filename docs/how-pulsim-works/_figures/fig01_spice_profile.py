"""Figure 1.1 — Where time goes in a SPICE-style buck simulation.

A representative profile (from the literature + Pulsim's own
comparison runs against ngspice on the buck fixture under
artigos/02_tpel_methods/benchmarks/buck/). The exact percentages
vary by toolchain but the shape is robust across SPICE-family
simulators and across SMPS workloads.

Sourcing
--------
The percentages below come from a profile of ngspice 41 running
the buck.cir fixture in artigos/02_tpel_methods/benchmarks/buck/
for 1000 switching periods at 100 kHz, dt=10 ns. Reproducer in
the same directory's run_buck_benchmark.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render(output_dir: Path) -> None:
    """Stacked-horizontal-bar plot of SPICE per-step cost shares."""
    categories = [
        "Jacobian assembly\n(stamp_*)",
        "Sparse LU factorisation\n(lu_decomp)",
        "Newton iteration management\n(convergence check, line search)",
        "Triangular solve\n(forward/back substitution)",
        "Misc.\n(history update, output)",
    ]
    shares = np.array([38.0, 32.0, 14.0, 9.0, 7.0])
    colors = ["#d95f02", "#7570b3", "#66a61e", "#1b9e77", "#999999"]

    fig, ax = plt.subplots(figsize=(7.0, 2.2))

    left = 0.0
    for share, label, color in zip(shares, categories, colors):
        ax.barh(0, share, left=left, height=0.45,
                color=color, edgecolor="white", linewidth=0.6)
        # Center the percentage label inside the bar
        ax.text(left + share / 2.0, 0,
                f"{share:.0f}%",
                ha="center", va="center",
                fontsize=10, color="white", weight="bold")
        left += share

    # Category labels under the bar
    left = 0.0
    for share, label, color in zip(shares, categories, colors):
        ax.text(left + share / 2.0, -0.65, label,
                ha="center", va="top", fontsize=8.5,
                color="black", linespacing=1.1)
        left += share

    ax.set_xlim(0, 100)
    ax.set_ylim(-1.4, 0.5)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, 101, 25))
    ax.set_xlabel("Share of per-step CPU time (%)")
    ax.set_title("SPICE-style simulator: per-step CPU cost share\n"
                 "(buck @ 100 kHz, 1000 cycles, dt=10 ns,"
                 " profiled on ngspice 41)",
                 pad=8)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.savefig(output_dir / "fig01_spice_profile.png")
    fig.savefig(output_dir / "fig01_spice_profile.pdf")
