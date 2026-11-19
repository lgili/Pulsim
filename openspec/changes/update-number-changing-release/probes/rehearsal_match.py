#!/usr/bin/env python3
"""Proof B(ii): REFERENCE_STATUS cells after the flip against the rehearsal.

    rehearsal_match.py REF_JSON      # the --json of benchmarks/tools/reference_status.py

Each released column must reproduce the rehearsal column that previewed it
(pwl <- pwl+next, trbdf2 <- trbdf2+next, dsed <- dsed+next, auto <- auto+next),
and each @legacy column the pre-release column (pwl@legacy <- pwl, ...): same
outcome per cell, and the same printed number where the rehearsal printed one.
Prints the counts and every differing cell; exit 1 when any differs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REH = ROOT / "benchmarks" / "NEXT_DEFAULTS_REHEARSAL.md"
MAP = {"pwl": "pwl+next", "trbdf2": "trbdf2+next", "dsed": "dsed+next", "auto": "auto+next",
       "pwl@legacy": "pwl", "pwl+restart@legacy": "pwl+restart", "pwl+split@legacy": "pwl+split",
       "trbdf2@legacy": "trbdf2", "dsed@legacy": "dsed"}


def rehearsal_cells():
    lines = REH.read_text(encoding="utf-8").splitlines()
    i = lines.index("## Every cell (err_rms / exact, or outcome)")
    head = [c.strip() for c in lines[i + 2].strip("|").split("|")]
    out = {}
    for ln in lines[i + 4:]:
        if not ln.startswith("|"):
            break
        row = [c.strip() for c in ln.strip("|").split("|")]
        out[row[0]] = dict(zip(head[1:], row[1:]))
    return out


def printed(rec):
    """The rehearsal's text of a cell, without its ``/ ex`` part (REFERENCE_STATUS has none)."""
    if rec["outcome"] in ("refused", "error", "timeout", "n/a"):
        return rec["outcome"]
    return f"{rec['err_rms']:.3g}" if "err_rms" in rec else rec["outcome"]


def main(argv) -> int:
    reh = rehearsal_cells()
    d = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    recs = [(r["circuit"].removeprefix("circuits/"), r) for r in d["differential"]]
    recs += [(f"analytic:{r['case']}", r) for r in d["analytic"]]
    bad, n, outcome_bad = [], 0, 0
    for circ, r in recs:
        old = reh.get(circ, {}).get(MAP.get(r["config"], ""))
        if old is None:
            continue
        n += 1
        old = old.split(" / ex ")[0]
        new = printed(r)
        if new != old:
            same = (new == old) or (old[:1].isdigit() and new[:1].isdigit())
            outcome_bad += not same
            bad.append(f"{circ} {r['config']} ({r['outcome']}): rehearsal {MAP[r['config']]} = {old}; now {new}")
    print(f"{n} cells compared, {len(bad)} differ in the printed value, {outcome_bad} in kind")
    print("\n".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
