"""Figure 15.1 — Per-step cost ladder across the 4 DSED bridges.

Stacked-bar visualisation showing where each bridge cut Python
overhead. Numbers from Bridge.11 + Bridge.12 profiles measured
during the v1.6.0 release validation on Apple M1.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render(output_dir: Path) -> None:
    bridges = ["PWL\n(baseline)", "Bridge.5\nPython sched",
                "Bridge.10\nC++ sched\n+ Py adapter",
                "Bridge.11\nC++ sched\n+ C++ adapter",
                "Bridge.12\nC++ sched\n+ C++ adapter\n+ native PWM"]

    # Categories of per-step cost (µs/step)
    # Source: profiler captures during v1.6.0 validation.
    #   PWL row is the trap engine for reference.
    #   DSED rows are decomposed by where the time goes.
    cpp_inner   = [1.07, 0.5,  0.5,   1.7,  1.7]    # native math (gemv, LU, PI)
    py_callback = [0.0, 30.0,  17.0,  1.1,  0.05]   # GIL + cast + Python rhs
    py_sched    = [0.0, 20.0,  1.0,   0.1,  0.1]    # Python scheduler logic
    py_misc     = [0.0, 10.8,  3.7,   0.9,  0.34]   # numpy round-trip, conversion

    total = np.array(cpp_inner) + np.array(py_callback) + np.array(py_sched) + np.array(py_misc)
    speedups_vs_pwl = total[0] / total

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(bridges))
    width = 0.65

    p1 = ax.bar(x, cpp_inner,   width, label="C++ inner loop (gemv, LU, PI ctrl, Hermite)",
                color="#3A6B35", edgecolor="black", linewidth=0.5)
    p2 = ax.bar(x, py_misc,     width, bottom=cpp_inner,
                label="Python misc (numpy round-trip, dict ops, conversions)",
                color="#F4A261", edgecolor="black", linewidth=0.5)
    p3 = ax.bar(x, py_callback, width, bottom=np.array(cpp_inner) + np.array(py_misc),
                label="Python callbacks (rhs/A_matrix per step ≫ GIL cost)",
                color="#E76F51", edgecolor="black", linewidth=0.5)
    p4 = ax.bar(x, py_sched,    width,
                bottom=np.array(cpp_inner) + np.array(py_misc) + np.array(py_callback),
                label="Python scheduler logic (while loops, dict updates)",
                color="#B23A48", edgecolor="black", linewidth=0.5)

    # Total labels
    for i, t in enumerate(total):
        ax.text(i, t + 1.5, f"{t:.2f} µs",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(i, t + 6, f"{speedups_vs_pwl[i]:.1f}× vs PWL",
                ha="center", va="bottom", fontsize=9,
                color="#222244",
                fontweight="bold" if speedups_vs_pwl[i] >= 1.0 else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels(bridges, fontsize=9.5)
    ax.set_ylabel("Per-step cost (µs)\nbuck CCM, 1007 DSED / 50001 PWL steps",
                  fontsize=10)
    ax.set_title("DSED per-step cost across 4 native bridges — buck CCM, "
                  "Apple M1, release build",
                  fontsize=11, fontweight="bold")
    ax.set_ylim(0, 80)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    # Annotation arrows
    ax.annotate("",
                xy=(2, total[2] + 0.5), xytext=(1, total[1] + 0.5),
                arrowprops=dict(arrowstyle="->", color="#444444", lw=1.2))
    ax.text(1.5, (total[1] + total[2]) / 2 + 5,
             "Bridge.10:\nC++ sched\neliminates\nPy interpreter",
             ha="center", fontsize=7.5, color="#444444",
             bbox=dict(facecolor="white", edgecolor="#bbbbbb", pad=2))

    ax.annotate("",
                xy=(3, total[3] + 0.5), xytext=(2, total[2] + 0.5),
                arrowprops=dict(arrowstyle="->", color="#444444", lw=1.2))
    ax.text(2.5, (total[2] + total[3]) / 2 + 5,
             "Bridge.11:\nnative adapter\nkills RHS GIL",
             ha="center", fontsize=7.5, color="#444444",
             bbox=dict(facecolor="white", edgecolor="#bbbbbb", pad=2))

    ax.annotate("",
                xy=(4, total[4] + 0.5), xytext=(3, total[3] + 0.5),
                arrowprops=dict(arrowstyle="->", color="#444444", lw=1.2))
    ax.text(3.5, (total[3] + total[4]) / 2 + 5,
             "Bridge.12:\nnative PWM kills\nswitch_fn GIL",
             ha="center", fontsize=7.5, color="#444444",
             bbox=dict(facecolor="white", edgecolor="#bbbbbb", pad=2))

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = output_dir / f"fig151_bridge_ladder.{ext}"
        fig.savefig(out, dpi=180 if ext == "png" else None,
                    bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    out = Path(__file__).parent / "output"
    render(out)
