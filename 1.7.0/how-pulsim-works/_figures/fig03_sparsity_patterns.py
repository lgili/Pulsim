"""Figure 2.2 — Sparsity patterns for representative SMPS MNA matrices.

Three side-by-side spy plots:

  * Buck            (n =  5, dense relative to size)
  * NPC 3-phase     (n = 22, ~12% dense, irregular)
  * MMC arm (9-SM)  (n = 30, ~7% dense, banded)

Each panel has its own title region with PROPER vertical spacing
from the next so subtitles don't run into each other horizontally.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp


def build_buck_matrix() -> sp.csr_matrix:
    """Buck companion matrix at S0=on, S1=off (n=5)."""
    n = 5
    rows = []
    cols = []
    vals = []

    def add(i, j, v):
        rows.append(i); cols.append(j); vals.append(v)

    G_S0, G_S1, G_R, G_C = 1e3, 1e-9, 0.2, 2 * 100e-6 / 100e-9
    add(0, 0,  G_S0);  add(0, 1, -G_S0)
    add(1, 0, -G_S0);  add(1, 1, G_S0 + G_S1);  add(1, 2, -G_S1)
    add(2, 1, -G_S1);  add(2, 2, G_S1 + G_R + G_C)
    add(0, 3, +1.0)
    add(3, 0, +1.0)
    add(4, 4, 1.0)

    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def build_npc_matrix() -> sp.csr_matrix:
    n = 22
    rng = np.random.default_rng(seed=2)
    rows, cols, vals = [], [], []

    def add(i, j, v):
        rows.append(i); cols.append(j); vals.append(v)

    for k in range(n):
        add(k, k, rng.uniform(0.5, 5.0))

    for leg in range(3):
        leg_top = 2 + 3 * leg
        leg_mid = 3 + 3 * leg
        leg_out = 4 + 3 * leg
        for from_, to_ in [(0, leg_top), (0, leg_mid),
                            (1, leg_mid), (1, leg_out),
                            (leg_top, leg_mid), (leg_mid, leg_out)]:
            v = rng.uniform(0.1, 2.0)
            add(from_, to_, -v); add(to_, from_, -v)
            add(from_, from_, +v); add(to_, to_, +v)
        ind_row = 11 + leg
        add(ind_row, leg_out, +1.0)
        add(leg_out, ind_row, +1.0)
        add(ind_row, ind_row, -100e-6)
    src_row = 14
    add(src_row, 0, +1.0); add(0, src_row, +1.0)
    for k in range(15, n):
        add(k, k, 1.0)

    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def build_mmc_matrix(n_sm: int = 9) -> sp.csr_matrix:
    n = 2 * n_sm + 12
    rng = np.random.default_rng(seed=3)
    rows, cols, vals = [], [], []

    def add(i, j, v):
        rows.append(i); cols.append(j); vals.append(v)

    for k in range(n):
        add(k, k, rng.uniform(0.5, 5.0))

    for k in range(n_sm - 1):
        i, j = 2 * k + 1, 2 * (k + 1) + 1
        v = rng.uniform(0.1, 1.5)
        add(i, j, -v); add(j, i, -v)

    add(0, 2 * n_sm, +1.0); add(2 * n_sm, 0, +1.0)
    add(2 * n_sm - 1, 2 * n_sm + 2, +1.0)
    add(2 * n_sm + 2, 2 * n_sm - 1, +1.0)
    for src in range(2 * n_sm + 4, n):
        target = rng.integers(0, 2 * n_sm)
        add(src, target, +1.0); add(target, src, +1.0)

    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def _spy(ax, M, name, n_sw_count):
    ax.spy(M, markersize=3.5, color="#1b1b1b")
    n = M.shape[0]
    density = M.nnz / (n * n) * 100.0
    # Two-line title with clear spacing
    ax.set_title(f"{name}\n({n_sw_count} switches)",
                  fontsize=10, pad=4)
    # Stats annotation as a text box BELOW the panel, anchored to
    # the axis bbox so it doesn't run into sibling panels.
    ax.text(0.5, -0.07,
              f"$n = {n}$    nnz = {M.nnz}    density = {density:.1f}%",
              ha="center", va="top",
              transform=ax.transAxes,
              fontsize=8.5, color="#444")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)


def render(output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.0))

    _spy(axes[0], build_buck_matrix(),  "Buck",         1)
    _spy(axes[1], build_npc_matrix(),   "NPC 3-phase",  12)
    _spy(axes[2], build_mmc_matrix(9),  "MMC arm, 9 SMs", 18)

    fig.suptitle("MNA sparsity patterns across representative SMPS",
                 fontsize=11, y=1.02)
    # Reserve room at the bottom for the stat captions
    fig.subplots_adjust(left=0.04, right=0.98, top=0.82, bottom=0.16,
                        wspace=0.22)
    fig.savefig(output_dir / "fig03_sparsity_patterns.png")
    fig.savefig(output_dir / "fig03_sparsity_patterns.pdf")
