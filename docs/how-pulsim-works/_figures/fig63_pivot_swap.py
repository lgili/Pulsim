"""Figure 6.3 — Pivot-row swap visualisation at column k=2.

Three-panel figure showing:
  * Left   — L being built, columns 0 and 1 stored, x loaded for col 2.
              x[7] is the argmax (red highlight).
  * Centre — the swap: rows 2 and 7 exchange in x AND in the
              already-stored columns 0/1 of L (yellow highlights).
  * Right  — after-swap state, ready for column 2's pivot normalisation.

Schematic — not a real factorisation trace. Designed to convey
the row-swap mechanics + Prow_ permutation update visually.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def grid_lu(ax, L_vals, x_vals, n=8, title="", swap_rows=None,
            argmax_row=None, swap_cols=None):
    """Render an n x n grid showing L (lower triangle) + workspace
    column x to the right.

    L_vals[i][j] is the L value (or None / NaN). Same for x_vals.
    """
    # L grid
    for i in range(n):
        for j in range(n):
            if j > i:
                ax.add_patch(Rectangle((j, n - 1 - i), 1, 1,
                                          facecolor="#f0f0f0",
                                          edgecolor="#cccccc",
                                          linewidth=0.5))
            elif j == i:
                ax.add_patch(Rectangle((j, n - 1 - i), 1, 1,
                                          facecolor="#dde6f3",
                                          edgecolor="#555555",
                                          linewidth=0.6))
                ax.text(j + 0.5, n - 1 - i + 0.5, "1",
                          ha="center", va="center", fontsize=8,
                          color="#1f3a6b")
            else:
                v = L_vals[i][j]
                hl = (swap_rows is not None
                       and i in swap_rows
                       and j in (swap_cols or []))
                facecolor = "#fff7cc" if hl else "#ffffff"
                ax.add_patch(Rectangle((j, n - 1 - i), 1, 1,
                                          facecolor=facecolor,
                                          edgecolor="#888888",
                                          linewidth=0.5))
                if v is not None:
                    ax.text(j + 0.5, n - 1 - i + 0.5,
                              f"{v:.1f}", ha="center", va="center",
                              fontsize=7, color="#111")

    # Workspace x column to the right of the grid
    x0 = n + 0.4
    for i in range(n):
        is_argmax = (argmax_row is not None and i == argmax_row)
        is_swap   = (swap_rows is not None and i in swap_rows)
        facecolor = "#ffd5d5" if is_argmax else ("#fff7cc" if is_swap else "#e9f5e9")
        ax.add_patch(Rectangle((x0, n - 1 - i), 1, 1,
                                  facecolor=facecolor,
                                  edgecolor="#666666",
                                  linewidth=0.6))
        v = x_vals[i]
        if v is not None:
            ax.text(x0 + 0.5, n - 1 - i + 0.5, f"{v:.1f}",
                      ha="center", va="center",
                      fontsize=7,
                      color=("#cc0000" if is_argmax else "#222"))

    # Header
    ax.text(n / 2, n + 0.45, "$L$ (stored)", ha="center",
              va="bottom", fontsize=9, weight="bold")
    ax.text(x0 + 0.5, n + 0.45, "$x$", ha="center",
              va="bottom", fontsize=9, weight="bold")

    # Row labels on the left
    for i in range(n):
        ax.text(-0.3, n - 1 - i + 0.5, str(i),
                  ha="right", va="center", fontsize=7, color="#555")

    ax.set_xlim(-0.7, x0 + 1.2)
    ax.set_ylim(-0.2, n + 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10, pad=2)


def render(output_dir: Path) -> None:
    n = 8

    # State BEFORE swap: columns 0, 1 stored (made-up small values)
    L_before = [[None] * n for _ in range(n)]
    L_before[1][0] = -0.3
    L_before[2][0] = 0.1; L_before[2][1] = -0.4
    L_before[3][1] =  0.2
    L_before[4][1] = -0.1
    L_before[7][0] =  0.5            # <-- will move to row 2 after swap
    L_before[7][1] =  0.3            # <-- ditto
    x_before = [None] * n
    x_before[2] = 0.05    # tiny — bad pivot
    x_before[3] = -0.1
    x_before[5] =  0.2
    x_before[7] =  1.4    # argmax

    # State AFTER swap: same matrix but rows 2 and 7 swapped in L cols 0/1
    L_after = [row[:] for row in L_before]
    L_after[2][0], L_after[7][0] = L_before[7][0], L_before[2][0]
    L_after[2][1], L_after[7][1] = L_before[7][1], L_before[2][1]
    x_after = list(x_before)
    x_after[2], x_after[7] = x_before[7], x_before[2]

    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.6))

    # Panel 1
    grid_lu(axes[0], L_before, x_before, n=n,
             title="Before swap (col k=2 workspace loaded)\n"
                    "$|x[2]| = 0.05$ too small — pivot search "
                    "finds $|x[7]| = 1.4$",
             argmax_row=7)

    # Panel 2: arrow indicating the swap, with both rows highlighted
    grid_lu(axes[1], L_before, x_before, n=n,
             title="Swap rows 2 ↔ 7 in $x$ AND in stored L cols 0,1\n"
                    "(highlighted yellow)",
             swap_rows=(2, 7), swap_cols=(0, 1))

    # Panel 3
    grid_lu(axes[2], L_after, x_after, n=n,
             title="After swap: column 2 ready for pivot normalisation\n"
                    "$P_{row}$ also updated: $P_{row}[2] \\leftrightarrow P_{row}[7]$")

    fig.suptitle("Pivot-row swap at column $k = 2$  (buck-like 8×8 fixture)",
                  fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig63_pivot_swap.png")
    fig.savefig(output_dir / "fig63_pivot_swap.pdf")
