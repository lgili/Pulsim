"""Extract publication-grade figures from the MMC validation pipeline.

Re-runs `mmc_pulsim_validation.simulate_mmc(...)` in both
balanced and open-loop modes and saves three publication-grade
figures as **both** 300 dpi PNG and vector PDF:

    figures/
      fig1_mmc_arm_and_ac.{png,pdf}    — arm voltages + AC output
      fig2_mmc_caps_balanced.{png,pdf} — 2N cap voltages, sort-and-select
      fig3_mmc_caps_drift.{png,pdf}    — open-loop drift + corrupted AC

These three figures are the seed material for both:

    * IEEE TPEL methods paper §VI (complex topology case study)
    * IEEE JESTPE application paper §IV (results — N = 3 baseline)

Usage:

    cd artigos/_shared/scripts
    python extract_figures.py                # default → ../figures/mmc/
    python extract_figures.py --out custom/  # custom output dir

The script depends only on numpy + matplotlib + pulsim. It re-runs
the simulations from scratch (the notebook outputs are not parsed)
so figure quality is independent of any prior notebook execution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


# Resolve repo root from this file's known location:
#   artigos/_shared/scripts/extract_figures.py  →  ../../..
REPO_ROOT = Path(__file__).resolve().parents[3]
MMC_DIR = REPO_ROOT / "projects" / "inverters" / "mmc"


def _import_mmc():
    """Import the MMC simulation module with the project on sys.path."""
    if str(MMC_DIR) not in sys.path:
        sys.path.insert(0, str(MMC_DIR))
    # The modules live at runtime under ``projects/inverters/mmc/`` —
    # they are not on the static import path, so type-checkers can't
    # resolve them. The ``type: ignore`` is intentional.
    import mmc_model  # type: ignore[import-not-found]  # noqa: E402,F401
    import mmc_pulsim_validation as mmc_val  # type: ignore[import-not-found]  # noqa: E402
    return mmc_model, mmc_val


def _publication_rcparams() -> None:
    """Apply matplotlib defaults appropriate for IEEE Transactions figures.

    Targets:
      * font-family that renders with the LaTeX `times` package fallback
      * line widths visible at column-width reduction
      * tick density that doesn't clutter at 3" wide reduction
    """
    plt.rcParams.update({
        "font.family":      "serif",
        "font.size":        10,
        "axes.titlesize":   11,
        "axes.labelsize":   10,
        "legend.fontsize":  8,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "lines.linewidth":  1.0,
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.05,
    })


def _save_dual(fig: Figure, out_dir: Path, basename: str) -> None:
    """Save the figure as both PNG (300dpi) and PDF (vector)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{basename}.png"
    pdf_path = out_dir / f"{basename}.pdf"
    fig.savefig(png_path, format="png")
    fig.savefig(pdf_path, format="pdf")
    print(f"  wrote {png_path.relative_to(REPO_ROOT)}")
    print(f"  wrote {pdf_path.relative_to(REPO_ROOT)}")


def make_fig1_arm_and_ac(p, res_bal, out_dir: Path) -> None:
    """Figure 1 — arm voltages + AC output (one fundamental period)."""
    t = np.asarray(res_bal["times"])
    mask = t < 1.0 / p.f_o
    t_ms = t[mask] * 1e3

    fig, (ax_arm, ax_ac) = plt.subplots(
        2, 1, figsize=(6.5, 4.5), sharex=True
    )

    ax_arm.plot(t_ms, res_bal["v_arm_up"][mask], color="C0",
                lw=0.7, label=r"$v_{\mathrm{arm,up}}(t)$")
    ax_arm.plot(t_ms, res_bal["v_arm_lo"][mask], color="C1",
                lw=0.7, label=r"$v_{\mathrm{arm,lo}}(t)$")
    for k in range(p.N_sm + 1):
        ax_arm.axhline(k * p.V_C_nominal, color="k", ls=":", alpha=0.25)
    ax_arm.set_ylabel("Arm voltage [V]")
    ax_arm.set_title(
        f"Arm voltages — {p.N_sm + 1} discrete levels at "
        rf"$k \cdot V_C$, $k \in [0, {p.N_sm}]$"
    )
    ax_arm.legend(loc="upper right")

    ax_ac.plot(t_ms, res_bal["v_ac"][mask], color="C2", lw=0.7,
               label=r"Pulsim $v_{\mathrm{ac}}(t)$")
    omega = 2 * np.pi * p.f_o
    v_ac_fund = p.V_o_pk * np.sin(omega * t[mask])
    ax_ac.plot(t_ms, v_ac_fund, color="C3", lw=1.4,
               label=rf"Analytical fundamental ({p.V_o_pk:.0f} V pk)")
    for k in range(-p.N_sm, p.N_sm + 1, 2):
        ax_ac.axhline(k * 0.5 * p.V_C_nominal, color="k",
                      ls=":", alpha=0.2)
    ax_ac.set_xlabel("Time [ms]")
    ax_ac.set_ylabel(r"$v_{\mathrm{ac}}$ [V]")
    ax_ac.legend(loc="lower right")

    fig.tight_layout()
    _save_dual(fig, out_dir, "fig1_mmc_arm_and_ac")
    plt.close(fig)


def make_fig2_caps_balanced(p, res_bal, out_dir: Path) -> None:
    """Figure 2 — all 2N sub-module cap voltages, sort-and-select balanced."""
    t = np.asarray(res_bal["times"])

    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    for i in range(p.N_sm):
        ax.plot(t * 1e3, res_bal["v_caps_up"][i],
                lw=0.9, label=fr"SM$_{{u{i}}}$ (upper)")
    for i in range(p.N_sm):
        ax.plot(t * 1e3, res_bal["v_caps_lo"][i],
                lw=0.9, ls="--", label=fr"SM$_{{l{i}}}$ (lower)")
    ax.axhline(p.V_C_nominal, color="k", ls=":", alpha=0.5,
               label=fr"Nominal $V_C = V_{{dc}}/N$ = {p.V_C_nominal:.1f} V")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("SM capacitor voltage [V]")
    ax.set_title(
        f"All {2 * p.N_sm} sub-module cap voltages — sort-and-select"
    )
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    _save_dual(fig, out_dir, "fig2_mmc_caps_balanced")
    plt.close(fig)


def make_fig3_caps_drift(p, res_open, out_dir: Path) -> None:
    """Figure 3 — open-loop drift + corrupted AC output (no balancing)."""
    t = np.asarray(res_open["times"])

    fig, (ax_c, ax_ac) = plt.subplots(
        2, 1, figsize=(6.5, 5.0), sharex=True
    )

    for i in range(p.N_sm):
        ax_c.plot(t * 1e3, res_open["v_caps_up"][i],
                  lw=1.0, label=fr"SM$_{{u{i}}}$")
    for i in range(p.N_sm):
        ax_c.plot(t * 1e3, res_open["v_caps_lo"][i],
                  lw=1.0, ls="--", label=fr"SM$_{{l{i}}}$")
    ax_c.axhline(p.V_C_nominal, color="k", ls=":", alpha=0.5,
                 label=fr"Target $V_C = V_{{dc}}/N$ = {p.V_C_nominal:.1f} V")
    ax_c.set_ylabel("SM cap voltage [V]")
    ax_c.set_title("Open-loop — caps diverge in a few cycles")
    ax_c.legend(loc="upper right", fontsize=7, ncol=2)

    ax_ac.plot(t * 1e3, res_open["v_ac"], color="C3", lw=0.5,
               label=r"Open-loop $v_{\mathrm{ac}}(t)$ — irregular")
    omega = 2 * np.pi * p.f_o
    v_ac_fund = p.V_o_pk * np.sin(omega * t)
    ax_ac.plot(t * 1e3, v_ac_fund, color="k", lw=1.0, alpha=0.5,
               label="Analytical fundamental (target)")
    ax_ac.set_xlabel("Time [ms]")
    ax_ac.set_ylabel(r"$v_{\mathrm{ac}}$ [V]")
    ax_ac.legend(loc="lower right")

    fig.tight_layout()
    _save_dual(fig, out_dir, "fig3_mmc_caps_drift")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "Extract MMC figures").splitlines()[0]
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent.parent / "figures" / "mmc",
        help="Output directory (default: ../figures/mmc/)",
    )
    parser.add_argument(
        "--t-end", type=float, default=0.05,
        help="Simulation horizon in seconds (default 0.05 = 3 cycles @ 60 Hz)",
    )
    parser.add_argument(
        "--dt", type=float, default=5e-6,
        help="Simulation timestep in seconds (default 5e-6)",
    )
    args = parser.parse_args()

    _publication_rcparams()

    print("Importing MMC simulation module ...")
    mmc_model, mmc_val = _import_mmc()
    p = mmc_model.MMCParams()

    print(f"Simulating MMC ({p.N_sm} SMs/arm, "
          f"t_end={args.t_end} s, dt={args.dt} s) — balanced ...")
    res_bal = mmc_val.simulate_mmc(
        p, t_end=args.t_end, dt=args.dt, balance_caps=True,
    )

    print(f"Simulating MMC — open-loop (no balancing) ...")
    res_open = mmc_val.simulate_mmc(
        p, t_end=args.t_end, dt=args.dt, balance_caps=False,
    )

    out_dir: Path = args.out
    print(f"Writing figures to {out_dir} ...")
    make_fig1_arm_and_ac(p, res_bal, out_dir)
    make_fig2_caps_balanced(p, res_bal, out_dir)
    make_fig3_caps_drift(p, res_open, out_dir)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
