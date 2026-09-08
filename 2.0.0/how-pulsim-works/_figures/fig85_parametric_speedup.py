"""Fig 8.5 — Parametric refactor speedup (Pulsim vs legacy rebuild).

Source CSV: artigos/02_tpel_methods/benchmarks/results/parametric_microbench.csv
(captured by core/tests/benchmarks/test_bench_parametric_sweep.cpp).

2-panel figure:
  * Panel A (left): Pulsim ÷ legacy rebuild speedup vs sweep-point
                    count, one line per n_switches.
  * Panel B (right): Per-point µs (legacy vs Pulsim) at n_switches=8,
                     showing the absolute wall-clock improvement.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CSV_PATH = (
    HERE.parents[2]
    / "artigos" / "02_tpel_methods" / "benchmarks" / "results"
    / "parametric_microbench.csv"
)


def _load_rows():
    rows = []
    with open(CSV_PATH, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append({
                "n_switches": int(r["n_switches"]),
                "n_state":    int(r["n_state"]),
                "n_pts":      int(r["n_sweep_points"]),
                "us_legacy":  float(r["us_per_point_legacy"]),
                "us_pulsim":  float(r["us_per_point_pulsim"]),
                "spd":        float(r["speedup_pulsim_vs_legacy"]),
            })
    return rows


def render(output_dir: Path) -> None:
    rows = _load_rows()
    if not rows:
        print("  [fig85] no data — skip")
        return

    n_sw_values = sorted({r["n_switches"] for r in rows})
    n_pts_values = sorted({r["n_pts"] for r in rows})

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(8.0, 3.4),
        gridspec_kw={"wspace": 0.32},
    )

    # ---- Panel A: Pulsim ÷ legacy speedup vs n_sweep_points -------------
    cmap = plt.get_cmap("plasma")
    for k, n_sw in enumerate(n_sw_values):
        color = cmap(0.15 + 0.7 * k / max(1, len(n_sw_values) - 1))
        ys = []
        for n_pts in n_pts_values:
            row = next((r for r in rows
                        if r["n_switches"] == n_sw and r["n_pts"] == n_pts),
                        None)
            ys.append(row["spd"] if row else np.nan)
        n_state = next(r["n_state"] for r in rows
                        if r["n_switches"] == n_sw)
        axL.plot(n_pts_values, ys, "o-", color=color,
                  label=f"n_state={n_state} ({n_sw} switches)",
                  linewidth=1.6, markersize=5)
    axL.axhline(1.0, color="grey", linestyle="--", linewidth=0.7,
                  label="parity")
    axL.set_xscale("log")
    axL.set_xlabel("Sweep points (log scale)")
    axL.set_ylabel("Pulsim path-based ÷ legacy rebuild")
    axL.set_title("Parametric sweep speedup\n"
                  "(captured 2026-05-24, ARM64, -O3)",
                  fontsize=10)
    axL.set_ylim(bottom=0.0)
    axL.legend(fontsize=8, loc="lower right")
    axL.grid(True, alpha=0.3, which="both")

    # ---- Panel B: per-point µs at n_switches=8 ---------------------------
    # Pick the largest fixture to show the absolute timing story.
    target_n_sw = n_sw_values[-1]
    target_rows = sorted(
        [r for r in rows if r["n_switches"] == target_n_sw],
        key=lambda r: r["n_pts"])
    n_pts_x = [r["n_pts"] for r in target_rows]
    us_legacy = [r["us_legacy"] for r in target_rows]
    us_pulsim = [r["us_pulsim"] for r in target_rows]
    n_state = target_rows[0]["n_state"]

    bar_w = 0.4
    idx = np.arange(len(n_pts_x))
    axR.bar(idx - bar_w/2, us_legacy, bar_w,
              label="legacy (rebuild/point)", color="#d95f02")
    axR.bar(idx + bar_w/2, us_pulsim, bar_w,
              label="Pulsim refactor_parametric", color="#1b9e77")
    axR.set_xticks(idx)
    axR.set_xticklabels([str(n) for n in n_pts_x])
    axR.set_xlabel("Sweep points")
    axR.set_ylabel("Per-point cost (µs)")
    axR.set_title(f"Absolute wall-time (n_state={n_state})",
                    fontsize=10)
    axR.legend(fontsize=8, loc="upper right")
    axR.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Fig 8.5 — Parametric refactor for sweeps / Monte Carlo (v1.4.0)",
                  fontsize=11, y=1.02)

    out_png = output_dir / "fig85_parametric_speedup.png"
    out_pdf = output_dir / "fig85_parametric_speedup.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    print(f"  [fig85] wrote {out_png.name} + .pdf")


if __name__ == "__main__":
    from generate_all import apply_paper_style
    apply_paper_style()
    render(HERE / "output")
