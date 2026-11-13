#!/usr/bin/env python3
"""Measure the analytic reference set and (re)write its ceilings.

    freeze_analytic.py            # print the table
    freeze_analytic.py --write    # rewrite benchmarks/manifests/analytic.yaml

Ceiling = 2 x measured err_rms, floored at 1e-9 (pwl, trbdf2) and 1e-6 (dsed,
whose ``integrator='auto'`` selection is being changed separately). A cell
listed under ``known:`` in the existing file is a recorded defect and keeps
its entry; a refusing engine is recorded as ``refused``. Review the diff:
a looser ceiling needs a ``reason``.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import metrics  # noqa: E402
from analytic import CASES  # noqa: E402

MANIFEST = ROOT / "manifests" / "analytic.yaml"
ENGINES = ("pwl", "trbdf2", "dsed")
FLOOR = {"pwl": 1e-9, "trbdf2": 1e-9, "dsed": 1e-6}


def measure(case, engine, k=1.0):
    r = case.run(engine, k)
    return metrics.err_rms(np.asarray(r.times), case.observe(r), case.exact,
                           abs_scale=case.abs_scale, exclude_t0=case.exclude_t0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    old = yaml.safe_load(MANIFEST.read_text()) if MANIFEST.exists() else {"cases": {}}
    out = {}
    for c in CASES:
        ent = {}
        for e in ENGINES:
            prev = old["cases"].get(c.id, {}).get(e, {})
            if e not in c.engines:
                try:
                    c.run(e)
                    ent[e] = {"error": "ran although the case lists it as refused"}
                except Exception as ex:  # noqa: BLE001
                    ent[e] = {"refused": type(ex).__name__}
                continue
            if "known" in prev:
                try:
                    ent[e] = dict(prev, measured=float(f"{measure(c, e):.3e}"))
                except Exception as ex:  # noqa: BLE001 - a recorded defect
                    ent[e] = dict(prev, measured=type(ex).__name__)
            else:
                m = measure(c, e)
                ent[e] = {"ceiling": float(f"{max(2 * m, FLOOR[e]):.2e}"), "measured": float(f"{m:.3e}")}
                if "reason" in prev:
                    ent[e]["reason"] = prev["reason"]
            print(f"{c.id:18s} {e:6s} {ent[e]}")
        out[c.id] = ent
    if args.write:
        head = [ln for ln in MANIFEST.read_text().splitlines() if ln.startswith("#")] if MANIFEST.exists() else []
        MANIFEST.write_text("\n".join(head) + "\n" + yaml.safe_dump({"cases": out}, sort_keys=False, width=100))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
