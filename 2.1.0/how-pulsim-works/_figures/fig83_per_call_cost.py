"""Figure 8.3 — Per-call cost (microseconds) vs n_state for all
three backends.

The "flat vs linear" figure — Pulsim path-based (C) stays
nearly flat as n grows; the baseline solve (A) climbs linearly.
That's the asymptotic argument that the speedup widens at scale.

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
    rows = []
    with open(CSV) as f:
        for row in csv.DictReader(f):
            rows.append({
                "n_state":   int(row["n_state"]),
                "us_solve":  float(row["us_per_solve"]),
                "us_eigen":  float(row["us_per_eigen"]),
                "us_pulsim": float(row["us_per_pulsim"]),
            })
    rows.sort(key=lambda r: r["n_state"])
    n_state = np.array([r["n_state"]   for r in rows])
    us_A    = np.array([r["us_solve"]  for r in rows])
    us_B    = np.array([r["us_eigen"]  for r in rows])
    us_C    = np.array([r["us_pulsim"] for r in rows])

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    # Linear-fit visualisation for the saturated-win range
    # (skip the small-n noise where the timer resolution dominates)
    sat = n_state >= 14
    if sat.any():
        for label, y, color in [("baseline solve (A)", us_A, "#d62728"),
                                  ("Pulsim path-based (C)", us_C, "#2ca02c")]:
            coef = np.polyfit(n_state[sat], y[sat], 1)
            xs = np.linspace(13, 28, 50)
            ys = coef[0] * xs + coef[1]
            ax.plot(xs, ys, color=color, lw=0.7, ls="--", alpha=0.5)
            ax.text(28.3, ys[-1], rf"slope ≈ {coef[0]:.2f} µs/n",
                      ha="left", va="center", fontsize=7.5, color=color)

    # Curves
    ax.plot(n_state, us_A, color="#d62728", lw=2.0, marker="o",
              markersize=6, label="(A) baseline solve — per-mask cache")
    ax.plot(n_state, us_B, color="#1f77b4", lw=1.6, marker="s",
              markersize=5, label="(B) sliding solver + Eigen LU")
    ax.plot(n_state, us_C, color="#2ca02c", lw=2.0, marker="^",
              markersize=6,
              label="(C) Pulsim path-based partial refactor")

    # Annotate slope: per-call cost growth from n=14 to n=26
    if sat.any():
        n14_idx = np.argmin(np.abs(n_state - 14))
        n26_idx = np.argmin(np.abs(n_state - 26))
        ax.annotate(
            rf"$\Delta$µs/$\Delta n$"
            rf" = {(us_A[n26_idx] - us_A[n14_idx]) / (n_state[n26_idx] - n_state[n14_idx]):.2f}"
            rf"  vs"
            rf" {(us_C[n26_idx] - us_C[n14_idx]) / (n_state[n26_idx] - n_state[n14_idx]):.2f}",
            xy=(20, 9.0), xytext=(15, 13.5),
            fontsize=8.5, color="#333",
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#aaa", lw=0.5),
            arrowprops=dict(arrowstyle="->", color="#666", lw=0.5),
        )

    ax.set_xlabel(r"$n_{\mathrm{state}}$")
    ax.set_ylabel(r"Per-call cost (µs)")
    ax.set_title("Per-call cost vs $n_{\\mathrm{state}}$: flat vs linear scaling\n"
                 "(2000 single-bit Gray-code flips per N, Apple Silicon)",
                 pad=8)
    ax.set_xticks(n_state)
    ax.set_xlim(5, 29)
    ax.set_ylim(0, 18)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_dir / "fig83_per_call_cost.png")
    fig.savefig(output_dir / "fig83_per_call_cost.pdf")
