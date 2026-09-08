"""Figure 8.1 — Three speedup ratios vs n_state from the captured
microbench (artigos/02_tpel_methods/benchmarks/results/rank1_microbench.csv).

Plots:
  * B/A — sliding-solver amortisation (Eigen vs baseline)
  * C/B — path-based win on top
  * C/A — headline (multiplicative)

Highlights the small-n crossover zone (n_state <= 12) where the
per-mask cache wins.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CSV  = REPO / "artigos" / "02_tpel_methods" / "benchmarks" / "results" \
            / "rank1_microbench.csv"


def load_csv():
    rows = []
    with open(CSV) as f:
        for row in csv.DictReader(f):
            rows.append({
                "N":               int(row["N"]),
                "n_state":         int(row["n_state"]),
                "us_solve":        float(row["us_per_solve"]),
                "us_eigen":        float(row["us_per_eigen"]),
                "us_pulsim":       float(row["us_per_pulsim"]),
                "speedup_B_A":     float(row["speedup_eigen_vs_solve"]),
                "speedup_C_A":     float(row["speedup_pulsim_vs_solve"]),
                "speedup_C_B":     float(row["speedup_pulsim_vs_eigen"]),
                "pulsim_hits":     int(row["pulsim_rank1_hits"]),
                "pulsim_fallbacks": int(row["pulsim_fallbacks"]),
            })
    return rows


def render(output_dir: Path) -> None:
    rows = load_csv()
    n_state = np.array([r["n_state"] for r in rows])
    b_over_a = np.array([r["speedup_B_A"] for r in rows])
    c_over_a = np.array([r["speedup_C_A"] for r in rows])
    c_over_b = np.array([r["speedup_C_B"] for r in rows])

    # Wider canvas + dedicated bottom legend region keeps the
    # in-line C/A annotations from colliding with the legend.
    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    # Crossover zone shading
    ax.axhspan(0.5, 1.0, color="#f5f5f5", alpha=0.6, lw=0, zorder=0)
    ax.axvspan(5.5, 12.5, color="#fff6e8", alpha=0.6, lw=0, zorder=0)
    ax.text(8.5, 0.55, "small-n crossover\n(per-mask cache wins here)",
              ha="center", va="bottom", fontsize=8, color="#aa6600",
              style="italic")

    # Reference line at 1.0×
    ax.axhline(1.0, color="#888888", lw=0.7, ls=":")

    # Three speedup curves
    ax.plot(n_state, c_over_a, color="#d95f02", lw=2.0, marker="o",
              markersize=6,
              label=r"$C/A$ — headline (Pulsim path-based vs baseline)")
    ax.plot(n_state, b_over_a, color="#1f77b4", lw=1.6, marker="s",
              markersize=5,
              label=r"$B/A$ — amortised-symbolic (Eigen sliding vs baseline)")
    ax.plot(n_state, c_over_b, color="#2ca02c", lw=1.6, marker="^",
              markersize=5,
              label=r"$C/B$ — path-based on top (Pulsim vs Eigen sliding)")

    # Annotate the headline at n=14, 18, 22, 26 — placed BELOW the
    # data point so they don't bump into the C/B curve above
    for n_val in (14, 18, 22, 26):
        idx = np.argmin(np.abs(n_state - n_val))
        ax.annotate(f"{c_over_a[idx]:.2f}×",
                     xy=(n_state[idx], c_over_a[idx]),
                     xytext=(0, 12),
                     textcoords="offset points",
                     ha="center", va="bottom",
                     fontsize=9, color="#d95f02", weight="bold")

    ax.set_xlabel(r"$n_{\mathrm{state}}$")
    ax.set_ylabel("Speedup ratio (×)")
    ax.set_title("3-backend microbench: speedup vs $n_{\\mathrm{state}}$\n"
                 "(macOS / Apple Silicon / 2000 single-bit Gray-code flips per N)",
                  pad=8)
    ax.set_xticks(n_state)
    ax.set_ylim(0.4, 3.4)
    # Bottom legend in a 3-column row underneath the plot — keeps
    # the plot area clear for annotations
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                ncol=1, frameon=False, fontsize=8.5,
                handlelength=2.5)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "fig81_speedup_vs_n.png")
    fig.savefig(output_dir / "fig81_speedup_vs_n.pdf")
