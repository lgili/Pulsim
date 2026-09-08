"""Figure 9.1 — Pulsim 10-layer architecture stack.

A vertical stack of 10 boxes (Layer 0 at bottom, Layer 9 at
top), each labeled with its responsibility + line-count
contribution. The right margin annotates which chapter
discusses each layer's algorithmic content.

Stylised; line counts are approximate (representative of the
v1.3.0 snapshot).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


LAYERS = [
    # (level, title, summary, approx_lines, chapter_ref)
    (9, "Python `pp.simulate()` facade",
        "control / plot / sweep / thermal / mmc / spice helpers",
        10000, "ergonomic API"),
    (8, "YAML loader",
        "declarative-circuit format → graph + pool",
        500,  ""),
    (7, "pybind11 Python bindings",
        "thin C++ → Python wrapping",
        1000, ""),
    (6, "CircuitBuilder (C++)",
        "ergonomic builder over Layer 1-3",
        1500, ""),
    (5, "run_transient + event detection",
        "the simulation driver loop",
        2000, ""),
    (4, "PwlStateSpaceCache + trapezoidal companion",
        "per-mask J·x = b cache, rank1 fast-path",
        3000, "chapters 3, 4, 7"),
    (3, "Stamping pipeline",
        "per-device KCL contributions to J + b",
        800,  "chapter 2"),
    (2, "Devices + AD",
        "Resistor, Capacitor, MOSFET, Diode, …",
        2000, ""),
    (1, "Graph + SwitchStateMask",
        "topology + bit-mask state enumeration",
        600,  ""),
    (0, "Numeric primitives + PulsimSparseLuSolver",
        "Real, Index, Matrix, DirectSolver, in-house sparse LU",
        2000, "chapters 5, 6, 7"),
]


def render(output_dir: Path) -> None:
    # Wider figure + dedicated columns avoid the overlap problem
    # the v1 layout had between the line-count / chapter-ref column
    # and the highlight sidebar.
    fig, ax = plt.subplots(figsize=(9.0, 6.0))

    # Column x-coordinates (carved out explicitly so nothing overlaps)
    BOX_LEFT  = 0.0
    BOX_RIGHT = 5.6
    LINES_X   = 5.95
    CHAP_X    = 7.05
    SIDEBAR_X = 8.6

    max_lines = max(l[3] for l in LAYERS)

    for layer in sorted(LAYERS, key=lambda x: -x[0]):
        level, title, summary, n_lines, chapter_ref = layer
        y = level
        shade = 0.25 + 0.55 * (n_lines / max_lines)
        is_highlight = level in (0, 4)
        face = (1.0 - shade * 0.4, 1.0 - shade * 0.2, 1.0)
        edge = "#d95f02" if is_highlight else "#1f3a6b"
        lw   = 2.0 if is_highlight else 0.8

        rect = FancyBboxPatch(
            (BOX_LEFT, y), BOX_RIGHT - BOX_LEFT, 0.85,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=face, edgecolor=edge, linewidth=lw,
        )
        ax.add_patch(rect)

        # Layer number left of the box
        ax.text(-0.35, y + 0.45, f"L{level}",
                ha="right", va="center",
                fontsize=11, weight="bold", color="#222")

        # Title + summary inside the box
        ax.text(BOX_LEFT + 0.15, y + 0.6, title,
                ha="left", va="center",
                fontsize=10, weight="bold", color="#111")
        ax.text(BOX_LEFT + 0.15, y + 0.22, summary,
                ha="left", va="center",
                fontsize=8.5, color="#444")

        # Line-count in its dedicated column
        ax.text(LINES_X, y + 0.45, f"~{n_lines:,} lines",
                ha="left", va="center",
                fontsize=8.5, color="#666", style="italic")

        # Chapter-ref in its dedicated column
        if chapter_ref:
            color = "#d95f02" if is_highlight else "#666"
            weight = "bold" if is_highlight else "normal"
            ax.text(CHAP_X, y + 0.45, chapter_ref,
                    ha="left", va="center",
                    fontsize=8.5, color=color, weight=weight)

    # Sidebar — vertical bar spanning ONLY L0 and L4 with arrows
    # pointing to each. Now in its own column so it doesn't collide.
    ax.add_patch(Rectangle((SIDEBAR_X, 0.0), 0.18, 0.85,
                              facecolor="#d95f02", alpha=0.7, lw=0))
    ax.add_patch(Rectangle((SIDEBAR_X, 4.0), 0.18, 0.85,
                              facecolor="#d95f02", alpha=0.7, lw=0))

    # Connect L0 ↔ L4 with a vertical line on the sidebar so the
    # reader sees they're related
    ax.plot([SIDEBAR_X + 0.09, SIDEBAR_X + 0.09], [0.85, 4.0],
              color="#d95f02", lw=1.0, ls=":", alpha=0.5)

    ax.text(SIDEBAR_X + 0.32, 2.43,
              "v1.3.0\nalgorithmic\ncontributions\n(this doc set)",
              ha="left", va="center", fontsize=8.5,
              color="#d95f02", weight="bold",
              linespacing=1.3)

    # Axis cosmetics
    ax.set_xlim(-1.4, 11.2)
    ax.set_ylim(-0.5, 10.4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("Pulsim 10-layer architecture (v1.3.0)\n"
                  "strict acyclic dependency: each layer uses only "
                  "the layers below",
                  fontsize=11, pad=10)

    fig.tight_layout()
    fig.savefig(output_dir / "fig91_layer_stack.png")
    fig.savefig(output_dir / "fig91_layer_stack.pdf")
