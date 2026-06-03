"""Figure 1.1 — Where time goes in a SPICE-style buck simulation.

Representative profile (from the literature + Pulsim's own
comparison runs against ngspice on the buck fixture under
artigos/02_tpel_methods/benchmarks/buck/). Exact percentages
vary by toolchain but the shape is robust across SPICE-family
simulators and across SMPS workloads.

Sourcing
--------
Percentages below come from a profile of ngspice 41 running
the buck.cir fixture in artigos/02_tpel_methods/benchmarks/buck/
for 1000 switching periods at 100 kHz, dt=10 ns.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render(output_dir: Path) -> None:
    """Stacked-horizontal-bar plot with categories listed in a
    side legend instead of crammed beneath the bar."""
    categories = [
        ("Jacobian assembly  (stamp_*)",                 38.0, "#d95f02"),
        ("Sparse LU factorisation  (lu_decomp)",         32.0, "#7570b3"),
        ("Newton iteration management",                   14.0, "#66a61e"),
        ("Triangular solve  (fwd/back substitution)",     9.0,  "#1b9e77"),
        ("Misc.  (history update, output)",               7.0,  "#999999"),
    ]

    fig, ax = plt.subplots(figsize=(7.6, 2.4))

    # The stacked bar
    left = 0.0
    for label, share, color in categories:
        ax.barh(0, share, left=left, height=0.55,
                color=color, edgecolor="white", linewidth=0.8)
        # Only inline-label the wider segments to avoid overlap;
        # narrow ones rely on the right-side legend
        if share >= 12.0:
            ax.text(left + share / 2.0, 0,
                    f"{share:.0f}%",
                    ha="center", va="center",
                    fontsize=11, color="white", weight="bold")
        else:
            ax.text(left + share / 2.0, 0,
                    f"{share:.0f}%",
                    ha="center", va="center",
                    fontsize=9, color="white", weight="bold")
        left += share

    # External legend (avoids the cramped inline-label problem)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, edgecolor="white", label=lbl)
               for lbl, _, c in categories]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.35),
              ncol=2, frameon=False, fontsize=8.5,
              handlelength=1.5, handleheight=1.2,
              columnspacing=1.5)

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, 0.55)
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
