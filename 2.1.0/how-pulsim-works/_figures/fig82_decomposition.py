"""Figure 8.2 — Multiplicative decomposition of the headline speedup.

For the saturated-win range (n_state in {14, 18, 22, 26}), show
the headline C/A as the multiplicative stack:
  C/A = (B/A) * (C/B)

Visualised as stacked bars: lower slice = B/A (blue), upper
slice = the addition needed to reach C/A (green, labelled with
the multiplicative factor C/B).

Each slice has its numeric label centered IN the slice; the
total C/A appears above the bar in orange. No labels overlap.

Reads from the captured CSV.
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


def render(output_dir: Path) -> None:
    targets = {14, 18, 22, 26}
    rows = []
    with open(CSV) as f:
        for row in csv.DictReader(f):
            n = int(row["n_state"])
            if n in targets:
                rows.append({
                    "n_state":    n,
                    "B_over_A":   float(row["speedup_eigen_vs_solve"]),
                    "C_over_A":   float(row["speedup_pulsim_vs_solve"]),
                    "C_over_B":   float(row["speedup_pulsim_vs_eigen"]),
                })
    rows.sort(key=lambda r: r["n_state"])

    n_state  = np.array([r["n_state"]   for r in rows])
    b_over_a = np.array([r["B_over_A"]  for r in rows])
    c_over_b = np.array([r["C_over_B"]  for r in rows])
    c_over_a = np.array([r["C_over_A"]  for r in rows])

    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    x = np.arange(len(rows))
    bar_w = 0.55

    # Bottom slice: B/A
    ax.bar(x, b_over_a, bar_w,
            color="#1f77b4", edgecolor="white", linewidth=0.7,
            label=r"$B/A$ — amortised-symbolic (sliding solver)")
    # Top slice: extra needed to reach C/A
    extra = c_over_a - b_over_a
    ax.bar(x, extra, bar_w, bottom=b_over_a,
            color="#2ca02c", edgecolor="white", linewidth=0.7,
            label=r"× $C/B$ — path-based partial refactor on top")

    # C/A total label above each bar
    for xi, ca in zip(x, c_over_a):
        ax.text(float(xi), ca + 0.08, f"C/A = {ca:.2f}×",
                ha="center", va="bottom",
                fontsize=9.5, weight="bold", color="#d95f02")

    # Per-slice labels — centered in the slice
    for xi, ba, cb in zip(x, b_over_a, c_over_b):
        # Lower slice midpoint
        ax.text(float(xi), float(ba) / 2.0, f"{ba:.2f}×",
                ha="center", va="center",
                color="white", fontsize=9.5, weight="bold")
        # Upper slice midpoint (correct calculation)
        upper_mid = float(ba) + (float(c_over_a[int(xi)]) - float(ba)) / 2.0
        ax.text(float(xi), upper_mid, f"× {cb:.2f}",
                ha="center", va="center",
                color="white", fontsize=9.5, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([rf"$n_{{\mathrm{{state}}}} = {n}$" for n in n_state])
    ax.set_ylabel("Cumulative speedup vs baseline (×)")
    ax.set_ylim(0, 3.5)
    ax.set_title("Headline speedup decomposed as $C/A = (B/A) \\cdot (C/B)$\n"
                 "(saturated-win range, $n_{\\mathrm{state}} \\geq 14$)",
                 pad=8)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "fig82_decomposition.png")
    fig.savefig(output_dir / "fig82_decomposition.pdf")
