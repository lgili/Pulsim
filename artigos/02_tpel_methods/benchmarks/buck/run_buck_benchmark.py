"""Buck converter — Pulsim vs ngspice head-to-head benchmark.

Runs the same buck topology (24 V → 12 V, 100 kHz, 5 ms window) in
both simulators, then reports:

    * Wall-clock simulation time
    * Peak resident-memory delta (best-effort, via resource.getrusage)
    * Output-voltage RMSE between the two simulators (at coincident
      sample instants, linear interpolation onto a common 1 µs grid)

This is the first data point in the TPEL methods paper §VI benchmark
table. The same orchestrator pattern will be reused for the other 9
converters in `projects/converters/` and `projects/inverters/` — each
just needs its own `<topology>.cir` netlist and a thin Pulsim runner.

Usage:

    cd artigos/02_tpel_methods/benchmarks/buck
    python run_buck_benchmark.py

    # Optional: bypass ngspice and just dump the Pulsim numbers
    python run_buck_benchmark.py --skip-ngspice

Outputs:

    ../results/buck_summary.csv              (one-row CSV)
    ../results/buck_waveform_overlay.png     (optional comparison plot)
    ./buck_ngspice_out.txt                   (ngspice ASCII trace)
"""

from __future__ import annotations

import argparse
import csv
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
# HERE = <repo>/artigos/02_tpel_methods/benchmarks/buck  →  4 levels up
REPO_ROOT = HERE.parents[3]
BUCK_DIR = REPO_ROOT / "projects" / "converters" / "buck"
RESULTS_DIR = HERE.parent / "results"

# Common comparison grid: 1 µs samples over 5 ms = 5000 samples.
# Coarser than either simulator's internal step; the comparison is
# of low-frequency content (output voltage ripple at 100 kHz max),
# not of switching edges per se.
COMPARE_T_END = 5e-3
COMPARE_DT = 1e-6


# ---------------------------------------------------------------------------
# Pulsim runner

def run_pulsim() -> dict:
    """Run the Pulsim buck simulation and report wall-time + memory."""
    if str(BUCK_DIR) not in sys.path:
        sys.path.insert(0, str(BUCK_DIR))

    import buck_model  # type: ignore[import-not-found]
    import buck_pulsim_validation as bv  # type: ignore[import-not-found]

    bp = buck_model.BuckParams()
    duty = bp.V_o / bp.V_g  # = D = 0.5 in defaults

    rusage_before = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    t, v_out = bv.simulate_buck(bp, duty=duty,
                                 t_end=COMPARE_T_END, dt=1e-7,
                                 warm_start=True)
    wall_s = time.perf_counter() - t0
    rusage_after = resource.getrusage(resource.RUSAGE_SELF)

    # ru_maxrss is in KiB on Linux, bytes on macOS — normalise to MiB.
    raw_delta = rusage_after.ru_maxrss - rusage_before.ru_maxrss
    if sys.platform == "darwin":
        peak_mem_mib = raw_delta / (1024 ** 2)
    else:
        peak_mem_mib = raw_delta / 1024

    return {
        "name":         "pulsim",
        "wall_s":       wall_s,
        "peak_mem_mib": peak_mem_mib,
        "n_samples":    int(len(t)),
        "times":        np.asarray(t),
        "v_out":        np.asarray(v_out),
    }


# ---------------------------------------------------------------------------
# ngspice runner

def _parse_ngspice_wrdata(txt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse an ASCII `wrdata` output file.

    The `.cir` deck uses `wrdata buck_ngspice_out.txt v(out)` which
    writes a two-column whitespace-separated ASCII file:

        <time_s>  <v_out_V>
        <time_s>  <v_out_V>
        ...

    No header rows. We rely on `numpy.loadtxt` for robust parsing.

    Returns (time, V(out)).
    """
    arr = np.loadtxt(txt_path)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise RuntimeError(
            f"wrdata file shape mismatch: got {arr.shape}, "
            f"expected (N, 2). Is the `.control` block in buck.cir "
            f"writing exactly one extra vector?"
        )
    return arr[:, 0], arr[:, 1]


def run_ngspice() -> dict:
    """Invoke ngspice on `buck.cir` and parse the output."""
    if shutil.which("ngspice") is None:
        raise RuntimeError(
            "ngspice not on PATH. Install via "
            "`brew install ngspice` (macOS) or `apt install ngspice` "
            "(Debian/Ubuntu)."
        )

    cir_path = HERE / "buck.cir"
    out_path = HERE / "buck_ngspice_out.txt"
    if out_path.exists():
        out_path.unlink()

    rusage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    proc = subprocess.run(
        ["ngspice", "-b", str(cir_path)],
        capture_output=True, text=True, cwd=HERE,
    )
    wall_s = time.perf_counter() - t0
    rusage_after = resource.getrusage(resource.RUSAGE_CHILDREN)

    if proc.returncode != 0:
        raise RuntimeError(
            "ngspice failed:\n--- stdout ---\n"
            + proc.stdout
            + "\n--- stderr ---\n"
            + proc.stderr
        )
    if not out_path.exists():
        raise RuntimeError(
            "ngspice ran but did not write buck_ngspice_out.txt\n"
            "Check the .control block in buck.cir.\n"
            "stdout:\n" + proc.stdout
        )

    raw_delta = rusage_after.ru_maxrss - rusage_before.ru_maxrss
    if sys.platform == "darwin":
        peak_mem_mib = raw_delta / (1024 ** 2)
    else:
        peak_mem_mib = raw_delta / 1024

    t, v = _parse_ngspice_wrdata(out_path)

    return {
        "name":         "ngspice",
        "wall_s":       wall_s,
        "peak_mem_mib": peak_mem_mib,
        "n_samples":    int(len(t)),
        "times":        t,
        "v_out":        v,
    }


# ---------------------------------------------------------------------------
# Comparison

def compare_outputs(a: dict, b: dict) -> dict:
    """Interpolate both onto a common grid, compute RMSE + max error.

    Both simulators report the LC-filter output `v(out)`. Comparison
    is over the second half of the 5 ms window (i.e. 2.5–5 ms) so the
    inrush transient is excluded and only the steady-state ripple is
    compared.
    """
    grid_t = np.arange(0, COMPARE_T_END, COMPARE_DT)
    settle_mask = grid_t >= 0.5 * COMPARE_T_END

    a_interp = np.interp(grid_t, a["times"], a["v_out"])
    b_interp = np.interp(grid_t, b["times"], b["v_out"])

    diff = a_interp[settle_mask] - b_interp[settle_mask]
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_abs = float(np.max(np.abs(diff)))
    mean_a = float(np.mean(a_interp[settle_mask]))
    mean_b = float(np.mean(b_interp[settle_mask]))

    return {
        "rmse_V":               rmse,
        "max_abs_err_V":        max_abs,
        "mean_v_out_pulsim":    mean_a if a["name"] == "pulsim" else mean_b,
        "mean_v_out_ngspice":   mean_b if b["name"] == "ngspice" else mean_a,
        "compare_grid_dt_s":    COMPARE_DT,
        "compare_window_s":     (COMPARE_T_END * 0.5, COMPARE_T_END),
    }


def write_summary(rows: list[dict], cmp: dict | None,
                   out_path: Path) -> None:
    """Write a one-table CSV summary of the run."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "simulator", "wall_s", "peak_mem_mib", "n_samples",
        "mean_v_out_V", "rmse_vs_other_V", "max_abs_err_vs_other_V",
        "platform", "python", "compare_window_s",
    ]

    plat_str = f"{platform.system()} {platform.release()} {platform.machine()}"
    py_str = f"{sys.implementation.name} {platform.python_version()}"

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            sim_name = r["name"]
            if cmp is not None:
                mean_key = "mean_v_out_" + sim_name
                mean_v_str = f"{cmp[mean_key]:.4f}"
                rmse_str = f"{cmp['rmse_V']:.6f}"
                max_err_str = f"{cmp['max_abs_err_V']:.6f}"
                window_str = (
                    f"{cmp['compare_window_s'][0]:.4g}-"
                    f"{cmp['compare_window_s'][1]:.4g}"
                )
            else:
                mean_v_str = ""
                rmse_str = ""
                max_err_str = ""
                window_str = ""
            w.writerow({
                "simulator":                sim_name,
                "wall_s":                   f"{r['wall_s']:.4f}",
                "peak_mem_mib":             f"{r['peak_mem_mib']:.2f}",
                "n_samples":                r["n_samples"],
                "mean_v_out_V":             mean_v_str,
                "rmse_vs_other_V":          rmse_str,
                "max_abs_err_vs_other_V":   max_err_str,
                "platform":                 plat_str,
                "python":                   py_str,
                "compare_window_s":         window_str,
            })
    print(f"Summary written to {out_path.relative_to(REPO_ROOT)}")


def maybe_plot_overlay(pulsim_run: dict, ngspice_run: dict,
                        out_path: Path) -> None:
    """Generate a side-by-side waveform overlay (best-effort)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping waveform overlay")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Zoom to a short window so the ripple is visible
    zoom_start = 4.0e-3
    zoom_end = 4.1e-3

    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(8.5, 5.5))

    ax_full.plot(pulsim_run["times"] * 1e3, pulsim_run["v_out"],
                  color="C0", lw=0.7, label="Pulsim")
    ax_full.plot(ngspice_run["times"] * 1e3, ngspice_run["v_out"],
                  color="C3", lw=0.7, ls="--", label="ngspice")
    ax_full.set_xlabel("Time [ms]")
    ax_full.set_ylabel(r"$v_\mathrm{out}$ [V]")
    ax_full.set_title("Buck: full 5 ms window")
    ax_full.legend(loc="lower right")
    ax_full.grid(True, alpha=0.3)

    m_p = (pulsim_run["times"] >= zoom_start) & (pulsim_run["times"] <= zoom_end)
    m_n = (ngspice_run["times"] >= zoom_start) & (ngspice_run["times"] <= zoom_end)
    ax_zoom.plot(pulsim_run["times"][m_p] * 1e3, pulsim_run["v_out"][m_p],
                  color="C0", lw=0.9, label="Pulsim")
    ax_zoom.plot(ngspice_run["times"][m_n] * 1e3, ngspice_run["v_out"][m_n],
                  color="C3", lw=0.9, ls="--", label="ngspice")
    ax_zoom.set_xlabel("Time [ms]")
    ax_zoom.set_ylabel(r"$v_\mathrm{out}$ [V]")
    ax_zoom.set_title(f"Buck: zoom to {zoom_start*1e3:.1f}–"
                       f"{zoom_end*1e3:.2f} ms (steady-state ripple)")
    ax_zoom.legend(loc="lower right")
    ax_zoom.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Main

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ngspice", action="store_true",
                         help="Run Pulsim only — useful if ngspice missing")
    parser.add_argument("--no-plot", action="store_true",
                         help="Skip waveform overlay generation")
    args = parser.parse_args()

    print(f"Platform: {platform.platform()}  Python {platform.python_version()}")
    print(f"Pulsim project: {BUCK_DIR.relative_to(REPO_ROOT)}")
    print()

    print("[1/2] Running Pulsim ...")
    pulsim_run = run_pulsim()
    print(f"      wall = {pulsim_run['wall_s']:.3f} s "
          f"({pulsim_run['n_samples']:,} samples, "
          f"Δmem ≈ {pulsim_run['peak_mem_mib']:.1f} MiB)")

    runs = [pulsim_run]
    cmp_result: dict | None = None

    if not args.skip_ngspice:
        print("[2/2] Running ngspice ...")
        ngspice_run = run_ngspice()
        print(f"      wall = {ngspice_run['wall_s']:.3f} s "
              f"({ngspice_run['n_samples']:,} samples, "
              f"Δmem ≈ {ngspice_run['peak_mem_mib']:.1f} MiB)")
        runs.append(ngspice_run)

        cmp_result = compare_outputs(pulsim_run, ngspice_run)
        print()
        print(f"Steady-state comparison ({cmp_result['compare_window_s'][0]*1e3:.1f}–"
              f"{cmp_result['compare_window_s'][1]*1e3:.1f} ms):")
        print(f"  mean V_out:  Pulsim {cmp_result['mean_v_out_pulsim']:.3f} V  "
              f"|  ngspice {cmp_result['mean_v_out_ngspice']:.3f} V")
        print(f"  RMSE:        {cmp_result['rmse_V']*1e3:.2f} mV")
        print(f"  max |err|:   {cmp_result['max_abs_err_V']*1e3:.2f} mV")

        if not args.no_plot:
            maybe_plot_overlay(
                pulsim_run, ngspice_run,
                RESULTS_DIR / "buck_waveform_overlay.png",
            )

    write_summary(runs, cmp_result, RESULTS_DIR / "buck_summary.csv")

    if cmp_result is not None:
        ratio = runs[1]["wall_s"] / runs[0]["wall_s"]
        print()
        print(f"Wall-time ratio (ngspice / Pulsim): {ratio:.2f}×")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
