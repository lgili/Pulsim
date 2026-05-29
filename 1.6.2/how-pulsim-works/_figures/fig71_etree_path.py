"""Figure 7.1 — Etree path walk on the buck-like 8x8 fixture.

Reuses fig52's etree layout but highlights:
  * The changed column (c = 2) in red
  * The path columns (the chain up to root) in orange
  * The off-path columns in blue (unchanged after partial update)

The path is computed by walking parent[k] from c upward.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee

# Reuse fig52's helpers by importing them
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fig52_etree import build_a, compute_etree, layout_tree


def compute_path(parent, start_col):
    """Walk parent chain from start_col up to root."""
    path = [start_col]
    cur = start_col
    while parent[cur] != -1:
        cur = parent[cur]
        path.append(cur)
    return path


def render(output_dir: Path) -> None:
    A = build_a()
    n = A.shape[0]
    perm = np.array(reverse_cuthill_mckee(A + A.T))
    P = sp.eye(n).tocsc()[perm, :]
    A_perm = P @ A @ P.T

    parent = compute_etree(A_perm + A_perm.T)
    xs, ys = layout_tree(parent)

    # Changed column = 2 (the column that toggles on S_2 flip)
    changed = 2
    path = compute_path(parent, changed)
    on_path = set(path)
    off_path = set(range(n)) - on_path

    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    # Edges
    for k in range(n):
        if parent[k] == -1:
            continue
        on_p = (k in on_path and parent[k] in on_path)
        lw = 2.0 if on_p else 1.0
        color = "#d95f02" if on_p else "#aaaaaa"
        ax.annotate(
            "", xy=(xs[parent[k]], ys[parent[k]]),
            xytext=(xs[k], ys[k]),
            arrowprops=dict(arrowstyle="->", lw=lw, color=color,
                             connectionstyle="arc3,rad=0.05"),
            zorder=2,
        )

    # Nodes
    for k in range(n):
        if k == changed:
            color = "#d62728"      # red: the column that changed
            label = f"{k} ← changed"
        elif k in on_path:
            color = "#d95f02"      # orange: on-path
            label = str(k)
        else:
            color = "#1f77b4"      # blue: off-path
            label = str(k)
        ax.scatter([xs[k]], [ys[k]], s=620, color=color,
                    edgecolor="black", linewidth=0.8, zorder=3)
        ax.text(xs[k], ys[k], str(k),
                ha="center", va="center",
                color="white", fontsize=10, weight="bold",
                zorder=4)

    # Legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
                markerfacecolor="#d62728", markeredgecolor="black",
                markersize=12, label="Changed column"),
        Line2D([0], [0], marker="o", color="w",
                markerfacecolor="#d95f02", markeredgecolor="black",
                markersize=12, label="Path columns (need re-elim.)"),
        Line2D([0], [0], marker="o", color="w",
                markerfacecolor="#1f77b4", markeredgecolor="black",
                markersize=12, label="Off-path (unchanged)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
                frameon=True, fontsize=8)

    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(-0.6, max(ys) + 0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(
        f"Path walk from changed column $c={changed}$ to etree root\n"
        f"Path length: {len(path)}/{n} columns "
        f"({100 * len(path) / n:.0f}% of the matrix)",
        fontsize=10, pad=10,
    )

    fig.tight_layout()
    fig.savefig(output_dir / "fig71_etree_path.png")
    fig.savefig(output_dir / "fig71_etree_path.pdf")
