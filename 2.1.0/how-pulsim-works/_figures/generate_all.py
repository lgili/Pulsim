#!/usr/bin/env python3
"""Regenerate every figure used in docs/how-pulsim-works/.

Each fig0N_*.py module defines a `render(output_dir: Path)`
function that writes `figXX_name.png` and `figXX_name.pdf`
into the supplied directory.

Usage
-----
From repo root:

    python docs/how-pulsim-works/_figures/generate_all.py

Or from this directory:

    python generate_all.py

Outputs land in `./output/` next to this script.

Style
-----
All figures use a uniform matplotlib stylesheet (defined inline
below) so the doc set has visual coherence. The same stylesheet
is reused by the TPEL paper figure regeneration scripts.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"


def apply_paper_style() -> None:
    """Uniform style: serif body, sans titles, 300 dpi, ~6×4 in.

    Calibrated for IEEE column width (3.5 in single, 7.16 in double).
    Targets ~10 pt body text after rendering at 300 dpi.
    """
    mpl.rcParams.update({
        # Backend + rendering
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
        # Typography (IEEE-ish)
        "font.family":         "serif",
        "font.serif":          ["Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset":    "cm",
        "font.size":           10,
        "axes.titlesize":      11,
        "axes.labelsize":      10,
        "xtick.labelsize":     9,
        "ytick.labelsize":     9,
        "legend.fontsize":     9,
        # Axes / grid
        "axes.linewidth":      0.6,
        "axes.grid":           True,
        "grid.alpha":          0.3,
        "grid.linewidth":      0.4,
        # Lines / markers
        "lines.linewidth":     1.4,
        "lines.markersize":    5,
        # Legends
        "legend.frameon":      False,
        "legend.borderaxespad": 0.3,
    })


def discover_figure_modules() -> list[str]:
    """Return ordered list of `fig0N_*` module names in this dir."""
    return sorted(
        p.stem for p in HERE.glob("fig*_*.py")
        if p.is_file() and p.stem != "generate_all"
    )


def main() -> int:
    apply_paper_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(HERE))
    modules = discover_figure_modules()
    if not modules:
        print("No fig*_*.py modules found in", HERE)
        return 1

    print(f"Regenerating {len(modules)} figure(s) into {OUTPUT_DIR}")
    for name in modules:
        try:
            mod = importlib.import_module(name)
        except Exception as exc:
            print(f"  [SKIP] {name}: import failed ({exc})")
            continue
        if not hasattr(mod, "render"):
            print(f"  [SKIP] {name}: no render() function")
            continue
        try:
            mod.render(OUTPUT_DIR)
            print(f"  [ok]   {name}")
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            plt.close("all")

    return 0


if __name__ == "__main__":
    sys.exit(main())
