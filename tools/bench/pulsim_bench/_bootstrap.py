"""Locate the Pulsim repo's `benchmarks/` directory and inject it into
`sys.path` so the existing runners (already patched in Onda 1+1.5)
can be imported.

Resolution order:
    1. `PULSIM_REPO_ROOT` env var (explicit override — `<root>/benchmarks`).
    2. Walk upward from this file looking for a sibling `benchmarks/`
       dir alongside a `pyproject.toml` whose `[project].name` is
       "pulsim" — matches a typical `tools/bench/` checkout layout.
    3. CWD search: if invoked from inside a Pulsim checkout, look for
       `./benchmarks/benchmark_runner.py` first, then climb parents.

Raises `RuntimeError` only when *all* strategies fail. Most callers
should let that bubble up as a clear startup error.

This module also exposes `repo_root()` and `benchmarks_dir()` for
subcommands that need filesystem anchors (e.g. default --benchmarks
path resolution).
"""

from __future__ import annotations

import os
import sys
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Optional


_MARKER_DIR = "circuits"  # stable: benchmarks/circuits/ predates the reorg


def _is_pulsim_root(p: Path) -> bool:
    """Heuristic: `p` looks like the Pulsim repo root if it has a
    `benchmarks/circuits/` directory AND a `pyproject.toml` whose
    `[project].name == "pulsim"`. The double-check avoids picking up
    a sibling project that also has a `benchmarks/` dir.

    Marker chosen post-Phase B reorg: `benchmarks/circuits/` is the
    one structural element that does NOT move (per REORG_PLAN.md
    Phase E sub-categorises files inside it, but the dir stays).
    """
    bench = p / "benchmarks" / _MARKER_DIR
    pyproject = p / "pyproject.toml"
    if not bench.is_dir() or not pyproject.is_file():
        return False
    try:
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        return str(data.get("project", {}).get("name", "")).strip() == "pulsim"
    except Exception:
        # If we can't parse pyproject, fall back to the marker dir
        # being present.
        return True


def _walk_upward(start: Path) -> Optional[Path]:
    p = start.resolve()
    for candidate in [p] + list(p.parents):
        if _is_pulsim_root(candidate):
            return candidate
    return None


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Return the Pulsim repo root, raising RuntimeError on failure."""
    # Strategy 1: explicit env var override.
    env = os.environ.get("PULSIM_REPO_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
        if _is_pulsim_root(root):
            return root
        # If user set it explicitly but it's wrong, fail loudly.
        raise RuntimeError(
            f"PULSIM_REPO_ROOT='{env}' is set but does not look like a "
            f"Pulsim checkout (missing benchmarks/{_MARKER_DIR}/ or "
            f"pyproject.toml [project].name != 'pulsim')"
        )

    # Strategy 2: walk up from this file. With `pip install -e
    # tools/bench`, __file__ is in the repo under tools/bench/.
    here = Path(__file__).resolve()
    found = _walk_upward(here.parent)
    if found is not None:
        return found

    # Strategy 3: walk up from cwd.
    found = _walk_upward(Path.cwd())
    if found is not None:
        return found

    raise RuntimeError(
        "Could not locate the Pulsim repo root. Tried walking up from "
        f"{here} and {Path.cwd()}. Set PULSIM_REPO_ROOT=/path/to/Pulsim "
        "to override."
    )


@lru_cache(maxsize=1)
def benchmarks_dir() -> Path:
    """Return `<repo_root>/benchmarks`."""
    return repo_root() / "benchmarks"


@lru_cache(maxsize=1)
def scripts_dir() -> Path:
    """Return `<repo_root>/scripts`."""
    return repo_root() / "scripts"


def ensure_benchmarks_on_path() -> Path:
    """Add `benchmarks/tools/` to `sys.path[0]` (idempotent) and return it.

    The bench runners (`benchmark_runner.py`, `benchmark_ngspice.py`,
    `stress_suite.py`, `local_limit_suite.py`, `_console.py`,
    `freeze_kpi_baseline.py`, `kpi_gate.py`) all live under
    `benchmarks/tools/` after the Phase B structural reorg, and they
    import each other as siblings (`from benchmark_runner import …`).
    Adding that dir to sys.path is the minimally-invasive way to keep
    them importable from this package.
    """
    tdir = benchmarks_dir() / "tools"
    tstr = str(tdir)
    if tstr not in sys.path:
        sys.path.insert(0, tstr)
    return tdir
