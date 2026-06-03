"""Fig 8.4 — Multi-bit rank-1 speedup vs Hamming distance + n_state.

Source CSV: artigos/02_tpel_methods/benchmarks/results/multi_bit_microbench.csv
(captured by core/tests/benchmarks/test_bench_multi_bit_rank1.cpp).

Renders a 2-panel paper-style figure:
  * Panel A (left): Pulsim ÷ Eigen speedup vs delta_bits, one line per N.
                    Isolates the path-based win (Eigen sliding solver =
                    full factorize per flip = v1.3.0 emulation for
                    multi-bit transitions).
  * Panel B (right): Multi-bit hit rate (fraction of calls that took
                     the path-union path successfully) vs delta_bits.
                     Shows the graceful decay as the union path widens.

Activated by `generate_all.py`; mirrors fig81/fig82/fig83 conventions.
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
    / "multi_bit_microbench.csv"
)


def _load_rows():
    rows = []
    with open(CSV_PATH, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append({
                "N":          int(r["N"]),
                "n_state":    int(r["n_state"]),
                "delta":      int(r["delta_bits"]),
                "n_calls":    int(r["n_calls"]),
                "spd_pulsim_vs_solve":
                    float(r["speedup_pulsim_vs_solve"]),
                "spd_pulsim_vs_eigen":
                    float(r["speedup_pulsim_vs_eigen"]),
                "single_bit":  int(r["single_bit_hits"]),
                "multi_bit":   int(r["multi_bit_hits"]),
                "full":        int(r["full_refactor_hits"]),
                "fall":        int(r["fallbacks"]),
            })
    return rows


def render(output_dir: Path) -> None:
    rows = _load_rows()
    if not rows:
        print("  [fig84] no data — skip")
        return

    Ns = sorted({r["N"] for r in rows})
    deltas = sorted({r["delta"] for r in rows})

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(8.0, 3.4),
        gridspec_kw={"wspace": 0.32},
    )

    # ---- Panel A: speedup vs delta_bits ---------------------------------
    cmap = plt.get_cmap("viridis")
    for k, N in enumerate(Ns):
        color = cmap(0.15 + 0.7 * k / max(1, len(Ns) - 1))
        ys = []
        for d in deltas:
            row = next((r for r in rows if r["N"] == N and r["delta"] == d), None)
            ys.append(row["spd_pulsim_vs_eigen"] if row else np.nan)
        axL.plot(deltas, ys, "o-", color=color,
                 label=f"N={N} (n={N+2})", linewidth=1.6, markersize=5)
    axL.axhline(1.0, color="grey", linestyle="--", linewidth=0.7,
                  label="parity")
    axL.set_xlabel("Hamming distance δ between consecutive masks")
    axL.set_ylabel("Pulsim ÷ Eigen (sliding solver baseline)")
    axL.set_title("Path-union speedup\n(captured 2026-05-24, ARM64, -O3)",
                    fontsize=10)
    axL.set_xticks(deltas)
    axL.set_ylim(bottom=0.0)
    axL.legend(fontsize=8, loc="lower left", ncol=2)
    axL.grid(True, alpha=0.3)

    # ---- Panel B: multi-bit hit rate vs delta_bits -----------------------
    for k, N in enumerate(Ns):
        color = cmap(0.15 + 0.7 * k / max(1, len(Ns) - 1))
        ys = []
        for d in deltas:
            row = next((r for r in rows if r["N"] == N and r["delta"] == d), None)
            if row is None:
                ys.append(np.nan)
            else:
                hit_rate = (row["single_bit"] + row["multi_bit"]) / row["n_calls"]
                ys.append(100.0 * hit_rate)
        axR.plot(deltas, ys, "s-", color=color,
                 label=f"N={N}", linewidth=1.6, markersize=5)
    axR.set_xlabel("Hamming distance δ")
    axR.set_ylabel("Path-based hit rate (%)")
    axR.set_title("Fraction of calls taking path-union\n"
                  "(remainder falls back to full factorize)",
                  fontsize=10)
    axR.set_xticks(deltas)
    axR.set_ylim(-5, 105)
    axR.legend(fontsize=8, loc="upper right")
    axR.grid(True, alpha=0.3)

    fig.suptitle("Fig 8.4 — Multi-bit rank-1 path-union performance (v1.4.0)",
                  fontsize=11, y=1.02)

    out_png = output_dir / "fig84_multi_bit_speedup.png"
    out_pdf = output_dir / "fig84_multi_bit_speedup.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    print(f"  [fig84] wrote {out_png.name} + .pdf")


if __name__ == "__main__":
    from generate_all import apply_paper_style
    apply_paper_style()
    render(HERE / "output")
