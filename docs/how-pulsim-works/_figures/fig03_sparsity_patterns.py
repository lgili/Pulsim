"""Figure 2.2 — Sparsity patterns for representative SMPS MNA matrices.

Three side-by-side spy plots:

  * Buck            (n =  5, dense relative to size)
  * NPC 3-phase     (n = 22, ~12% dense, irregular)
  * MMC arm (9-SM)  (n = 30, ~7% dense, banded)

The matrices are *constructed* (not loaded) so the figure
regenerates without the kernel built. The structure faithfully
mirrors what Pulsim's `assemble_segment(...)` produces — block-
diagonal for independent stages, plus the +/-1 spike pattern for
voltage-source augmentation rows.

If you change a converter or want to swap to real matrices read
from the kernel, replace the build_* helpers below with calls
into pybind11's `cache.matrix_for(mask)`.
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

    # Node-conductance block (n0..n2 = Vin, Sw, Vout)
    G_S0, G_S1, G_R, G_C = 1e3, 1e-9, 0.2, 2 * 100e-6 / 100e-9
    add(0, 0,  G_S0);  add(0, 1, -G_S0)
    add(1, 0, -G_S0);  add(1, 1, G_S0 + G_S1);  add(1, 2, -G_S1)
    add(2, 1, -G_S1);  add(2, 2, G_S1 + G_R + G_C)
    # Voltage-source augmentation: branch_current_id = 3
    add(0, 3, +1.0)
    add(3, 0, +1.0)
    # Dummy aux node row to keep dim = 5
    add(4, 4, 1.0)

    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def build_npc_matrix() -> sp.csr_matrix:
    """NPC 3-phase MNA. 6 internal nodes + 3 phase-leg outputs +
    4 DC-link nodes + 3 source-augmentation rows + 6 inductor-
    current rows = n = 22 (rough but representative; pattern is
    what matters)."""
    n = 22
    rng = np.random.default_rng(seed=2)

    # 3 phase legs ⇒ 3 ~4×4 blocks; each leg shares the 2 DC-link
    # midpoint nodes (rows 0 and 1).
    rows, cols, vals = [], [], []

    def add(i, j, v):
        rows.append(i); cols.append(j); vals.append(v)

    # Backbone diagonal
    for k in range(n):
        add(k, k, rng.uniform(0.5, 5.0))

    # DC-link mid (rows 0, 1) talk to each phase leg's inputs
    for leg in range(3):
        leg_top = 2 + 3 * leg
        leg_mid = 3 + 3 * leg
        leg_out = 4 + 3 * leg
        # Each switch + clamping-diode contributes a 2-node coupling
        for from_, to_ in [(0, leg_top), (0, leg_mid),
                            (1, leg_mid), (1, leg_out),
                            (leg_top, leg_mid), (leg_mid, leg_out)]:
            v = rng.uniform(0.1, 2.0)
            add(from_, to_, -v); add(to_, from_, -v)
            add(from_, from_, +v); add(to_, to_, +v)
        # Inductor branch-current row (rows 11+leg)
        ind_row = 11 + leg
        add(ind_row, leg_out, +1.0)
        add(leg_out, ind_row, +1.0)
        add(ind_row, ind_row, -100e-6)
    # Source-augmentation rows for the DC bus
    src_row = 14
    add(src_row, 0, +1.0); add(0, src_row, +1.0)
    # Anchored aux nodes (rows 15..21) — small diag entries
    for k in range(15, n):
        add(k, k, 1.0)

    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def build_mmc_matrix(n_sm: int = 9) -> sp.csr_matrix:
    """MMC arm with n_sm submodules → ~banded matrix.

    Each SM has (cap voltage state + branch-current row) and
    couples to its neighbors via the arm inductor.

    Result: n = 30 for n_sm = 9.
    """
    n = 2 * n_sm + 12  # 2 per SM + DC-link + arm + AC-side
    rng = np.random.default_rng(seed=3)
    rows, cols, vals = [], [], []

    def add(i, j, v):
        rows.append(i); cols.append(j); vals.append(v)

    for k in range(n):
        add(k, k, rng.uniform(0.5, 5.0))

    # SM-to-SM coupling through the arm inductor (banded)
    for k in range(n_sm - 1):
        i, j = 2 * k + 1, 2 * (k + 1) + 1
        v = rng.uniform(0.1, 1.5)
        add(i, j, -v); add(j, i, -v)

    # AC + DC link coupling at the ends
    # First-SM ↔ AC-mid, last-SM ↔ DC-rail
    add(0, 2 * n_sm, +1.0); add(2 * n_sm, 0, +1.0)
    add(2 * n_sm - 1, 2 * n_sm + 2, +1.0)
    add(2 * n_sm + 2, 2 * n_sm - 1, +1.0)
    # A spread of constraint rows
    for src in range(2 * n_sm + 4, n):
        target = rng.integers(0, 2 * n_sm)
        add(src, target, +1.0); add(target, src, +1.0)

    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def _spy(ax, M, title):
    ax.spy(M, markersize=2.5, color="#1b1b1b")
    n = M.shape[0]
    density = M.nnz / (n * n) * 100.0
    ax.set_title(f"{title}\n$n = {n}$,  nnz = {M.nnz},  "
                  f"density = {density:.1f}%",
                  fontsize=10, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)


def render(output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))

    _spy(axes[0], build_buck_matrix(),  "Buck (1 switch)")
    _spy(axes[1], build_npc_matrix(),   "NPC 3-phase (12 switches)")
    _spy(axes[2], build_mmc_matrix(9),  "MMC arm, 9 SMs (18 switches)")

    fig.suptitle("MNA sparsity patterns across representative SMPS",
                 fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(output_dir / "fig03_sparsity_patterns.png")
    fig.savefig(output_dir / "fig03_sparsity_patterns.pdf")
