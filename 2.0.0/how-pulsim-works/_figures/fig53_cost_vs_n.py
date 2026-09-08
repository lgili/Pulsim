"""Figure 5.3 — Asymptotic per-call cost vs n for four algorithms.

Log-log plot. Curves:
  * Dense Gaussian elimination  ~ n**3
  * Dense LU with pivoting       ~ n**3 / 3
  * Sparse LU natural ordering  ~ n**2 (representative; depends
                                  on fill)
  * Sparse LU + RCM ordering    ~ n * log(n)  (banded fixture)

A dashed grey rectangle highlights the SMPS-relevant range
n in [5, 200].

Coefficients calibrated so the curves are roughly correct
at n = 100 (where dense LU = ~30k ops, sparse+RCM = ~700 ops).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render(output_dir: Path) -> None:
    n = np.logspace(1, 3.5, 200)  # 10 to ~3200

    # Calibrate at n = 100
    n_ref = 100.0
    dense_ge_ref  = (n_ref ** 3) * 1.0          # ~1e6 ops
    dense_lu_ref  = (n_ref ** 3) / 3.0          # ~3.3e5 ops
    sparse_natural_ref = (n_ref ** 2) * 5.0     # ~5e4 ops, fill-blown
    sparse_rcm_ref     = n_ref * np.log(n_ref) * 10.0  # ~4.6e3 ops

    dense_ge      = (n ** 3) * 1.0
    dense_lu      = (n ** 3) / 3.0
    sparse_nat    = (n ** 2) * 5.0
    sparse_rcm    = n * np.log(n) * 10.0

    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    ax.loglog(n, dense_ge,   color="#888888",
                lw=1.4, ls="--", label=r"Dense Gaussian elim. $\sim n^3$")
    ax.loglog(n, dense_lu,   color="#9467bd",
                lw=1.6,           label=r"Dense LU $\sim n^3/3$")
    ax.loglog(n, sparse_nat, color="#d62728",
                lw=1.6,           label=r"Sparse LU, natural order $\sim n^2$")
    ax.loglog(n, sparse_rcm, color="#1f77b4",
                lw=2.0,           label=r"Sparse LU + RCM $\sim n \log n$  (Pulsim)")

    # SMPS range highlight
    ax.axvspan(5, 200, color="#fff7e0", alpha=0.5, lw=0)
    ax.text(31, dense_ge[np.argmin(np.abs(n - 31))] * 3, "SMPS range",
             fontsize=9, color="#aa6600", weight="bold")
    ax.axvline(5,   color="#cc9900", ls=":", lw=0.7)
    ax.axvline(200, color="#cc9900", ls=":", lw=0.7)

    # Speedup annotation at n = 200
    n_pt = 200.0
    rcm_pt  = n_pt * np.log(n_pt) * 10.0
    lu_pt   = (n_pt ** 3) / 3.0
    ratio   = lu_pt / rcm_pt
    ax.annotate(
        rf"{ratio:.0f}× speedup at $n = 200$",
        xy=(n_pt, rcm_pt), xytext=(900, 500),
        fontsize=9, color="#1f77b4",
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.8),
    )

    ax.set_xlabel(r"Matrix size $n$ (log scale)")
    ax.set_ylabel("Per-factorisation cost (flops, log scale)")
    ax.set_title("Asymptotic cost vs matrix size for four LU algorithms\n"
                 "(SMPS range highlighted; coefficients calibrated at $n=100$)",
                 pad=8)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_dir / "fig53_cost_vs_n.png")
    fig.savefig(output_dir / "fig53_cost_vs_n.pdf")
