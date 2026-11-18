"""Strict reference grading of trbdf2 with and without mode_selection='lcp'.

    PYTHONPATH=<worktree>/python:<worktree>/benchmarks python3 trbdf2_lcp_grade.py > trbdf2_lcp_grade.out

Every circuit of reference_status.py, graded exactly as it grades (same
references, tolerances, interp measure), in two configurations: trbdf2 and
trbdf2 + 'lcp'. REFERENCE_STATUS.md is not written. Printed: the outcome
counts per configuration and every cell whose outcome or error differs.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4] / "benchmarks"
sys.path[:0] = [str(ROOT / "tools"), str(ROOT)]
import reference_status as rs  # noqa: E402


def main() -> int:
    warnings.simplefilter("ignore")
    rs.CONFIGS = {"trbdf2": ("trbdf2", {}), "trbdf2+lcp": ("trbdf2", {"mode_selection": "lcp"})}
    cells = rs.differential(["circuits/*"])
    by = {}
    for r in cells:
        by.setdefault(r["circuit"], {})[r["config"]] = r
    for cfg in rs.CONFIGS:
        n = {k: sum(1 for r in cells if r["config"] == cfg and r["outcome"] == k) for k in rs.KINDS}
        print(cfg, n)
    for c, d in sorted(by.items()):
        a, b = d.get("trbdf2", {}), d.get("trbdf2+lcp", {})
        ea, eb = a.get("err_rms"), b.get("err_rms")
        if a.get("outcome") != b.get("outcome") or ea != eb:
            print(f"{c:60s} {a.get('outcome')} {ea} -> {b.get('outcome')} {eb} {(b.get('message') or '')[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
