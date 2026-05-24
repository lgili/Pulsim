"""Figure 6.2 — Dynamic pattern discovery vs symbolic prediction.

Three side-by-side spy plots of L+U on the buck-like 8x8:
  * "Symbolic pre-pivot" — what analyze() predicted, assuming
                            Prow = Pcol
  * "Actual (post-pivot)" — what factorize() actually produced,
                            with missed entries highlighted red
  * "Dynamic discovery" — what dynamic-pattern recording stores

The "missed" entries are what bug 2 corrupted before the
dynamic-pattern fix.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee


def build_a():
    """Buck-like 8x8 with M[7,7] = 0 (voltage-source augmentation)."""
    n = 8
    rows, cols, vals = [], [], []
    def add(i, j, v):
        rows.append(i); cols.append(j); vals.append(v)
    # Most diagonals are 1; row 7 has zero diagonal (forces a pivot)
    for k in range(n - 1):
        add(k, k, 1.0)
    # Coupling
    couples = [(0, 1), (1, 2), (2, 3),
               (1, 4), (2, 5), (3, 6),
               (4, 5), (5, 6), (6, 7)]
    for i, j in couples:
        add(i, j, -0.5); add(j, i, -0.5)
        rows.append(i); cols.append(i); vals.append(0.5)
        rows.append(j); cols.append(j); vals.append(0.5)
    # Voltage-source spike on row 7 (zero diagonal)
    add(7, 0, 1.0); add(0, 7, 1.0)
    return sp.csc_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()


def render(output_dir: Path) -> None:
    A = build_a()
    n = A.shape[0]
    perm = np.array(reverse_cuthill_mckee(A + A.T))
    P = sp.eye(n).tocsc()[perm, :]
    A_perm = P @ A @ P.T

    # Pretend we did the analyze symbolically assuming Prow = Pcol:
    # the pattern is just A_perm's + symbolic fill via etree walk.
    # We'll approximate it by taking |A_perm| and its symbolic
    # fill bounded by chol_pat(A_perm + I) — which slightly
    # underestimates the post-pivot pattern.
    pre_pivot = (np.abs(A_perm.toarray()) > 1e-12).astype(int)
    # Add ad-hoc symbolic predicted fill (etree walk approximation):
    for k in range(n):
        for i in range(k + 1, n):
            if pre_pivot[i, k] and pre_pivot[k, k]:
                # Predicts fill at (i, j) for j > k where A_perm[k, j] != 0
                for j in range(k + 1, n):
                    if pre_pivot[k, j]:
                        pre_pivot[i, j] = 1

    # Compute the actual L+U via scipy's splu (no symmetric perm)
    from scipy.sparse.linalg import splu
    lu = splu(A_perm.tocsc(), permc_spec="NATURAL",
              options={"DiagPivotThresh": 1e-3})
    L = lu.L.toarray(); U = lu.U.toarray()
    actual = ((np.abs(L) > 1e-12) | (np.abs(U) > 1e-12)).astype(int)

    # Highlight the entries the symbolic phase missed: in actual,
    # not in pre_pivot
    missed = ((actual == 1) & (pre_pivot == 0)).astype(int)

    # Dynamic discovery = actual (by definition, dynamic recording
    # captures every nonzero in x at factorisation time)
    dynamic = actual.copy()

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7))

    # Panel 1: pre-pivot prediction
    axes[0].spy(sp.csr_matrix(pre_pivot), markersize=7, color="#1b1b1b")
    axes[0].set_title(f"Symbolic pre-pivot\nprediction (nnz = {int(pre_pivot.sum())})",
                       fontsize=10, pad=4)

    # Panel 2: actual factorisation, with missed entries in red
    axes[1].spy(sp.csr_matrix(actual & ~missed), markersize=7,
                 color="#1b1b1b")
    axes[1].spy(sp.csr_matrix(missed), markersize=10, color="#d62728",
                 marker="x")
    axes[1].set_title(
        f"Actual $L+U$ (nnz = {int(actual.sum())})\n"
        rf"$\times$ = {int(missed.sum())} missed by symbolic",
        fontsize=10, pad=4)

    # Panel 3: dynamic discovery (= actual, but framed differently)
    axes[2].spy(sp.csr_matrix(dynamic), markersize=7, color="#2ca02c")
    axes[2].set_title(
        f"Dynamic discovery\n(captures all, nnz = {int(dynamic.sum())})",
        fontsize=10, pad=4)

    for ax in axes:
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_linewidth(0.5)

    fig.suptitle("Why analyze's symbolic phase isn't enough under partial pivoting",
                  fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(output_dir / "fig62_dynamic_pattern.png")
    fig.savefig(output_dir / "fig62_dynamic_pattern.pdf")
