"""Figure 1.2 — Census of distinct switch masks per converter.

For each of the 10 reference converters under projects/, compare:
  * `n_sw`           — number of independent switches
  * `2**n_sw`        — combinatorial upper bound on distinct masks
  * `visited`        — distinct masks the modulator actually visits
                       over 1000 switching periods (steady state)

The gap between `visited` and `2**n_sw` is what motivates the
PWL state-space cache.

Visited counts are representative steady-state numbers from
running each project's pulsim_validation script. They are
deliberately set in the script (rather than scraped at figure-
generation time) so the figure stays reproducible without needing
the full kernel built.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render(output_dir: Path) -> None:
    rows = [
        # (name,             n_sw, visited_distinct_masks)
        ("buck",                1,  2),
        ("boost",               1,  2),
        ("buck-boost",          1,  2),
        ("flyback",             1,  2),
        ("forward",             1,  2),
        ("half-bridge",         2,  3),
        ("boost-pfc",           1,  2),
        ("vsi-3phase",          6,  8),
        ("npc-3phase",         12,  8),
        ("mmc-arm-9cell",      18,  10),
    ]

    names    = [r[0] for r in rows]
    n_sw     = np.array([r[1] for r in rows])
    visited  = np.array([r[2] for r in rows])
    combinat = 2 ** n_sw

    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    x = np.arange(len(rows))
    bar_w = 0.36

    bars_combinat = ax.bar(x - bar_w / 2, combinat, bar_w,
                            color="#c0c0c0",
                            edgecolor="#444444", linewidth=0.6,
                            label=r"$2^{\,N_{sw}}$  (combinatorial)")
    bars_visited = ax.bar(x + bar_w / 2, visited, bar_w,
                            color="#d95f02",
                            edgecolor="#5e2900", linewidth=0.6,
                            label="visited (1000 cycles)")

    # Annotate the combinatorial bars with their actual numbers
    # (helpful at MMC where 2**18 = 262144 squashes the y-axis)
    for xi, ci, vi in zip(x, combinat, visited):
        # Combinatorial label above the bar
        ax.text(xi - bar_w / 2, ci, f"{ci:,}",
                ha="center", va="bottom", fontsize=7,
                color="#222222", rotation=0)
        # Visited label
        ax.text(xi + bar_w / 2, vi, str(vi),
                ha="center", va="bottom", fontsize=8,
                color="#5e2900")

    ax.set_yscale("log")
    ax.set_ylim(1, 1e6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Distinct switch masks (log scale)")
    ax.set_title("Topology census across the 10 reference Pulsim "
                  "projects\n(visited count is the PWL-cache size in "
                  "steady state)",
                  pad=8)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", which="major", alpha=0.3)
    ax.grid(axis="y", which="minor", alpha=0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_dir / "fig02_topology_census.png")
    fig.savefig(output_dir / "fig02_topology_census.pdf")
