"""Figure 16.1 — DSED vs PWL speedup across 6 converter topologies.

Bar chart of wall-clock speedup (DSED Bridge.12 over PWL Bridge.13)
on the bench-sweep harness output (scripts/bench_dsed_vs_pwl.py).
Numbers are the measured geo-mean = 14.5× run captured during the
v1.6.0 release validation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render(output_dir: Path) -> None:
    # Data: from scripts/bench_dsed_vs_pwl.py output, captured for v1.6.0
    topologies = [
        "Buck CCM\n24V→12V",
        "Boost\n12V→24V",
        "Buck-boost\n24V→−24V",
        "Half-bridge\n+ sine V_in",
        "Floating-cap\nRLC decay",
        "NPC split-bus\n2 caps",
    ]
    speedups = [19.5, 18.8, 19.6, 12.1, 12.7, 8.3]
    n_steps_dsed = [1007, 2007, 1007, 1007, 435, 507]
    n_steps_pwl = [50001, 100001, 50001, 50001, 10001, 5001]

    geomean = np.exp(np.mean(np.log(speedups)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                      gridspec_kw={"width_ratios": [3, 2]})

    # ----- Left panel: speedup bar chart -----
    x = np.arange(len(topologies))
    bars = ax1.bar(x, speedups,
                   color=["#2E86AB" if s >= geomean else "#A23B72"
                          for s in speedups],
                   edgecolor="black", linewidth=0.6)
    ax1.axhline(geomean, color="black", linestyle="--", linewidth=1.0,
                label=f"Geo-mean = {geomean:.1f}×")
    ax1.set_xticks(x)
    ax1.set_xticklabels(topologies, rotation=0, fontsize=8.5)
    ax1.set_ylabel("Wall-clock speedup\nDSED vs PWL (×)", fontsize=10)
    ax1.set_title("DSED ÷ PWL across 6 converter topologies",
                  fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3, linestyle=":")
    ax1.set_ylim(0, 22)

    for i, (bar, s) in enumerate(zip(bars, speedups)):
        ax1.text(bar.get_x() + bar.get_width() / 2.0,
                  bar.get_height() + 0.3,
                  f"{s:.1f}×", ha="center", va="bottom",
                  fontsize=8.5, fontweight="bold")

    # ----- Right panel: step-count ratio -----
    ratios = [pwl / dsed for pwl, dsed in zip(n_steps_pwl, n_steps_dsed)]
    short_names = ["Buck", "Boost", "B-Boost", "HB+sine", "RLC", "NPC"]
    bars2 = ax2.barh(np.arange(len(short_names)), ratios,
                     color="#3A6B35", edgecolor="black", linewidth=0.6)
    ax2.set_yticks(np.arange(len(short_names)))
    ax2.set_yticklabels(short_names, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Steps PWL ÷ steps DSED", fontsize=10)
    ax2.set_title("Where the speedup comes from:\nfewer steps",
                  fontsize=11, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3, linestyle=":")
    for i, (bar, r) in enumerate(zip(bars2, ratios)):
        ax2.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2.0,
                  f"{r:.0f}×", ha="left", va="center",
                  fontsize=8.5, fontweight="bold")
    ax2.set_xlim(0, max(ratios) * 1.18)

    fig.suptitle(
        "DSED (engine='dsed', native) vs PWL (engine='pwl', fixed-step trap) — "
        "buck CCM headline 24×",
        fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = output_dir / f"fig161_dsed_topology_sweep.{ext}"
        fig.savefig(out, dpi=180 if ext == "png" else None,
                    bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    out = Path(__file__).parent / "output"
    render(out)
