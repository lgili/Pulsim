"""Fig 8.6 — AC sweep per-frequency cost: Pulsim vs Eigen baseline.

Source CSV: artigos/02_tpel_methods/benchmarks/results/ac_sweep_microbench.csv
(captured by core/tests/benchmarks/test_bench_ac_sweep.cpp).

2-panel figure:
  * Panel A (left): µs per frequency vs matrix size n — both backends.
                    Shows the asymptotic crossover at n ≥ 64 where
                    Eigen's reachability-based sparse triangular solve
                    pulls ahead.
  * Panel B (right): Pulsim ÷ Eigen ratio vs n. The v1.4.0 "rough
                     parity at SMPS sizes" headline.

This figure supports the chapter-9 architecture story: v1.4.0 ships
the in-house complex LU primarily for the software-supply-chain
benefit (no third-party LU on the production path), not for a
per-frequency speedup.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CSV_PATH = (
    HERE.parents[2]
    / "artigos" / "02_tpel_methods" / "benchmarks" / "results"
    / "ac_sweep_microbench.csv"
)


def _load_rows():
    rows = []
    with open(CSV_PATH, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append({
                "n":           int(r["n"]),
                "nnz":         int(r["matrix_nnz"]),
                "us_eigen":    float(r["us_per_freq_eigen"]),
                "us_pulsim":   float(r["us_per_freq_pulsim"]),
                "spd":         float(r["speedup_pulsim_vs_eigen"]),
                "parity_eps":  float(r["max_solve_diff"]),
            })
    return sorted(rows, key=lambda r: r["n"])


def render(output_dir: Path) -> None:
    rows = _load_rows()
    if not rows:
        print("  [fig86] no data — skip")
        return

    ns = [r["n"] for r in rows]
    eigen_us = [r["us_eigen"] for r in rows]
    pulsim_us = [r["us_pulsim"] for r in rows]
    spd = [r["spd"] for r in rows]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(8.0, 3.4),
        gridspec_kw={"wspace": 0.32},
    )

    # ---- Panel A: per-frequency µs vs n ---------------------------------
    axL.plot(ns, eigen_us, "o-", color="#d95f02",
              label="Eigen::SparseLU (baseline)",
              linewidth=1.6, markersize=5)
    axL.plot(ns, pulsim_us, "s-", color="#1b9e77",
              label="PulsimComplexSparseLuSolver (v1.4.0)",
              linewidth=1.6, markersize=5)
    axL.set_xscale("log", base=2)
    axL.set_yscale("log")
    axL.set_xlabel("Matrix size n")
    axL.set_ylabel("Per-frequency cost (µs)")
    axL.set_xticks(ns)
    axL.set_xticklabels([str(n) for n in ns])
    axL.set_title("AC sweep per-frequency cost\n"
                    "(100 log-spaced freqs, 1 Hz–1 MHz)",
                    fontsize=10)
    axL.legend(fontsize=8, loc="upper left")
    axL.grid(True, alpha=0.3, which="both")

    # ---- Panel B: Pulsim ÷ Eigen ratio + parity Δ -----------------------
    bars = axR.bar(range(len(ns)), spd, color="#1b9e77", alpha=0.8,
                    label="Pulsim ÷ Eigen")
    axR.axhline(1.0, color="grey", linestyle="--", linewidth=0.7,
                  label="parity")
    # Annotate each bar with the ratio
    for i, (bar, ratio) in enumerate(zip(bars, spd)):
        axR.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.05,
                  f"{ratio:.2f}×", ha="center", fontsize=8)
    axR.set_xticks(range(len(ns)))
    axR.set_xticklabels([f"n={n}" for n in ns])
    axR.set_ylabel("Pulsim ÷ Eigen ratio")
    axR.set_ylim(0, max(2.5, max(spd) * 1.2))
    axR.set_title("Per-frequency speedup ratio\n"
                    "(parity Δ ≤ 4×10⁻²¹ at every n)",
                    fontsize=10)
    axR.legend(fontsize=8, loc="upper left")
    axR.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Fig 8.6 — AC sweep complex LU: in-house vs Eigen baseline (v1.4.0)",
                  fontsize=11, y=1.02)

    out_png = output_dir / "fig86_ac_sweep_parity.png"
    out_pdf = output_dir / "fig86_ac_sweep_parity.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    print(f"  [fig86] wrote {out_png.name} + .pdf")


if __name__ == "__main__":
    from generate_all import apply_paper_style
    apply_paper_style()
    render(HERE / "output")
