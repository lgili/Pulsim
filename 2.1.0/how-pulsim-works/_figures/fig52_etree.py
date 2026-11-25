"""Figure 5.2 — Elimination tree for an 8x8 MNA matrix.

Computes the etree via Liu 1986 / Davis 2006 §4.10's disjoint-
set ancestor-compression algorithm (matching what
PulsimSparseLuSolver::compute_etree_ does), then renders it as
a tree.

Compact layout — narrower aspect ratio fits the tree without
the wide empty whitespace the v1 layout had.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee


def build_a():
    """Same buck-like 8x8 as fig51."""
    n = 8
    rows, cols, vals = [], [], []
    def add(i, j, v):
        rows.append(i); cols.append(j); vals.append(v)
    for k in range(n):
        add(k, k, 1.0)
    couples = [(0, 1), (1, 2), (2, 3),
               (1, 4), (2, 5), (3, 6),
               (4, 5), (5, 6), (6, 7)]
    for i, j in couples:
        add(i, j, -0.5); add(j, i, -0.5)
        rows.append(i); cols.append(i); vals.append(0.5)
        rows.append(j); cols.append(j); vals.append(0.5)
    add(7, 0, 1.0); add(0, 7, 1.0)
    return sp.csc_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()


def compute_etree(A_sym: sp.csc_matrix):
    n = A_sym.shape[0]
    parent = [-1] * n
    ancestor = [-1] * n
    A_sym = A_sym.tocsc()
    for k in range(n):
        col = A_sym.getcol(k).indices
        for i in col:
            if i >= k:
                continue
            j = i
            while ancestor[j] != -1 and ancestor[j] != k:
                tmp = ancestor[j]
                ancestor[j] = k
                j = tmp
            if ancestor[j] == -1:
                ancestor[j] = k
                if parent[j] == -1:
                    parent[j] = k
    return parent


def layout_tree(parent):
    n = len(parent)
    depth = [0] * n
    for k in range(n):
        d = 0
        cur = k
        while parent[cur] != -1:
            cur = parent[cur]; d += 1
        depth[k] = d
    max_depth = max(depth)
    xs = np.array([k for k in range(n)], dtype=float)
    ys = np.array([max_depth - depth[k] for k in range(n)],
                   dtype=float)
    return xs, ys


def render(output_dir: Path) -> None:
    A = build_a()
    n = A.shape[0]
    perm = np.array(reverse_cuthill_mckee(A + A.T))
    P = sp.eye(n).tocsc()[perm, :]
    A_perm = P @ A @ P.T

    parent = compute_etree(A_perm + A_perm.T)
    xs, ys = layout_tree(parent)

    # Compact 6x3 figure — no wasted whitespace
    fig, ax = plt.subplots(figsize=(6.4, 3.0))

    # Edges first
    for k in range(n):
        if parent[k] == -1:
            continue
        ax.annotate(
            "", xy=(xs[parent[k]], ys[parent[k]]),
            xytext=(xs[k], ys[k]),
            arrowprops=dict(arrowstyle="->", lw=1.2,
                             color="#555555",
                             connectionstyle="arc3,rad=0.05"),
        )

    # Nodes
    for k in range(n):
        is_root = (parent[k] == -1)
        color = "#d95f02" if is_root else "#1f77b4"
        ax.scatter([xs[k]], [ys[k]], s=560, color=color,
                    edgecolor="black", linewidth=0.8, zorder=3)
        ax.text(xs[k], ys[k], str(k),
                ha="center", va="center",
                color="white", fontsize=10, weight="bold",
                zorder=4)

    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(-0.4, max(ys) + 0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Two-line title — orange root explanation on a separate line
    ax.set_title(
        "Elimination tree of an 8×8 MNA matrix (RCM-ordered)\n"
        "orange = roots; arrows: column $k$ → parent$[k]$",
        fontsize=10, pad=8,
    )

    fig.tight_layout()
    fig.savefig(output_dir / "fig52_etree.png")
    fig.savefig(output_dir / "fig52_etree.pdf")
