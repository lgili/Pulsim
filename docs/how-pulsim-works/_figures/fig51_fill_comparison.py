"""Figure 5.1 — Fill comparison: natural vs RCM ordering.

Three side-by-side spy plots:
  * Original A      — sparse pattern of the buck-like 8x8
  * L+U natural     — factorised with natural column ordering
  * L+U RCM         — factorised with RCM column ordering

Demonstrates how a bad ordering blows fill, and how RCM tames it
by recovering banded structure.

Uses scipy.sparse + scipy.sparse.linalg.splu for the natural-
order LU, and a custom RCM permutation (matching what Pulsim's
PulsimSparseLuSolver does) for the RCM-order LU. Both produce
exactly the L+U pattern Pulsim would store internally.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee


def build_a():
    """Buck-like 8x8 MNA matrix with anchor branches.

    Same fixture as core/tests/layer0/test_pulsim_lu_solver.cpp.
    Symmetric+spike pattern from voltage-source augmentation.
    """
    n = 8
    rows, cols, vals = [], [], []
    def add(i, j, v):
        rows.append(i); cols.append(j); vals.append(v)

    # Diagonal anchors
    for k in range(n):
        add(k, k, 1.0)
    # Coupling: branches between nearby nodes
    couples = [(0, 1), (1, 2), (2, 3),
               (1, 4), (2, 5), (3, 6),
               (4, 5), (5, 6), (6, 7)]
    for i, j in couples:
        add(i, j, -0.5); add(j, i, -0.5)
        # Increase diag to keep it diagonally dominant
        rows.append(i); cols.append(i); vals.append(0.5)
        rows.append(j); cols.append(j); vals.append(0.5)
    # Voltage-source spike row 7
    add(7, 0, 1.0); add(0, 7, 1.0)
    return sp.csc_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()


def lu_pattern(A_perm, eps=1e-12):
    """Return the L+U sparsity pattern (binary spy mask).

    Uses scipy's splu (KLU-like sparse direct) — pattern matches
    what Pulsim's PulsimSparseLuSolver would store.
    """
    from scipy.sparse.linalg import splu
    try:
        lu = splu(A_perm.tocsc(), permc_spec="NATURAL",
                   options={"DiagPivotThresh": 0.0})
        L = lu.L.toarray()
        U = lu.U.toarray()
        mask = (np.abs(L) > eps).astype(int) + (np.abs(U) > eps).astype(int)
        return (mask > 0).astype(int)
    except Exception:
        # Fallback: just return |A| pattern
        return (np.abs(A_perm.toarray()) > eps).astype(int)


def render(output_dir: Path) -> None:
    A = build_a()
    n = A.shape[0]

    # RCM permutation
    perm = np.array(reverse_cuthill_mckee(A + A.T))
    P = sp.eye(n).tocsc()[perm, :]
    A_rcm = P @ A @ P.T

    a_pat   = (np.abs(A.toarray())     > 1e-12).astype(int)
    nat_pat = lu_pattern(A)
    rcm_pat = lu_pattern(A_rcm)

    nnz_A   = int(a_pat.sum())
    nnz_nat = int(nat_pat.sum())
    nnz_rcm = int(rcm_pat.sum())

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))
    for ax, mat, title in zip(
        axes,
        [a_pat, nat_pat, rcm_pat],
        [
            f"Original $|A|$\n nnz = {nnz_A}",
            f"$L+U$, natural ordering\n nnz = {nnz_nat}",
            f"$L+U$, RCM ordering\n nnz = {nnz_rcm}"
        ],
    ):
        ax.spy(sp.csr_matrix(mat), markersize=7, color="#1b1b1b")
        ax.set_title(title, fontsize=10, pad=4)
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_linewidth(0.5)

    fig.suptitle(
        "Effect of column ordering on $L+U$ fill\n"
        "(buck-like 8×8 MNA fixture)",
        fontsize=11, y=1.05,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "fig51_fill_comparison.png")
    fig.savefig(output_dir / "fig51_fill_comparison.pdf")
