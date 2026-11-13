#!/usr/bin/env python3
"""Differential runner: benchmark circuit × engine, time-weighted error.

Each cell is ``agree`` / ``disagree`` (``err_rms`` against the reference
within / outside the tolerance), ``refused`` (the engine's typed refusal:
``pulsim.UnsupportedError`` when it exists, else ``ValueError`` /
``NotImplementedError``) or ``error`` (anything else). Reference, first
available: ``trbdf2`` at
``rtol = 1e-8``, ``pwl`` at ``dt/4`` when ``dt/2`` and ``dt/4`` agree within
a quarter of the tolerance. The first stored sample is excluded from every
comparison (on pwl / trbdf2 it is the initial state, not t = 0+). The
ngspice comparison is ``benchmark_ngspice.py`` (nightly job).

Usage::

    differential_runner.py --circuits 'linear/*' --engines pwl,trbdf2,dsed \
        --out results.json

Exit status 1 when a cell is ``disagree`` or ``error`` and the triage
manifest does not list that outcome under ``known:`` for the circuit.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402
import v1_reader  # noqa: E402

import pulsim  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRIAGE = ROOT / "manifests" / "v2_triage.yaml"
ENGINES = ("pwl", "trbdf2", "dsed")
REFUSAL = tuple(x for x in (getattr(pulsim, "UnsupportedError", None),
                            ValueError, NotImplementedError) if x is not None)


def load_triage(path: Path = TRIAGE) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["circuits"]


def default_tol(engine: str, switched: bool) -> float:
    if not switched:
        return 1e-3
    return 2e-2 if engine == "pwl" else 1e-3


def observables_of(case) -> list:
    cols = list(case.observables) or ([case.validation["observable"]]
                                      if case.validation.get("observable") else [])
    if not cols:
        b, _, _ = case.build()
        cols = [s for s in b.state_var_names() if s.startswith(("V(", "I("))]
    return cols


def traces(res, cols):
    t = np.asarray(res.times, dtype=float)
    return t, {c: np.asarray(v1_reader.observable(res, c), dtype=float) for c in cols}


def compare(tr, ref, cols, scale):
    """Worst (err_rms, err_max) over the observables."""
    t, ys = tr
    tr_, yr = ref
    worst = (0.0, 0.0)
    for c in cols:
        a = np.abs(yr[c]).max()
        kw = dict(abs_scale=max(scale * a, 1e-9), exclude_t0=True)
        r = (tr_, yr[c])
        e = (metrics.err_rms(t, ys[c], r, **kw), metrics.err_max(t, ys[c], r, **kw))
        worst = max(worst, e)
    return worst


def reference(case, cols, tol):
    try:
        res = case.run("trbdf2", rtol=1e-8, atol=1e-12)
        return "trbdf2 rtol=1e-8", traces(res, cols)
    except Exception:
        pass
    a = traces(case.run("pwl", dt=case.dt / 2), cols)
    b = traces(case.run("pwl", dt=case.dt / 4), cols)
    e, _ = compare(a, b, cols, 1e-3)
    if e <= tol / 4:
        return "pwl dt/4", b
    return None, None


def run_cell(case, engine):
    t0 = time.perf_counter()
    try:
        res = case.run(engine)
        return "ran", res, time.perf_counter() - t0, None
    except REFUSAL as ex:
        return "refused", None, time.perf_counter() - t0, f"{type(ex).__name__}: {ex}"[:300]
    except Exception as ex:  # noqa: BLE001 - classified, never swallowed
        return "error", None, time.perf_counter() - t0, f"{type(ex).__name__}: {ex}"[:300]


def run(patterns, engines, triage):
    out = []
    for rel, entry in sorted(triage.items()):
        if entry["status"] != "runs" or not any(fnmatch.fnmatch(rel, p) for p in patterns):
            continue
        case = v1_reader.load(ROOT / rel)
        b, sw, _ = case.build()
        blob = json.dumps(case._components)
        switched = sw is not None or any(k in blob for k in ('"diode"', '"pwm"', '"pulse"'))
        cols = observables_of(case)
        tols = {e: entry.get("tol", {}).get(e, default_tol(e, switched)) for e in engines}
        ref_name, ref = reference(case, cols, min(tols.values()))
        for e in engines:
            kind, res, wall, msg = run_cell(case, e)
            rec = dict(circuit=rel, engine=e, reference=ref_name, tol=tols[e],
                       wall_s=round(wall, 4), message=msg)
            if kind != "ran":
                rec["outcome"] = kind
            elif ref is None:
                rec["outcome"] = "error"
                rec["message"] = "no converged reference"
            else:
                er, em = compare(traces(res, cols), ref, cols, 1e-3)
                rec.update(err_rms=er, err_max=em,
                           outcome="agree" if er <= tols[e] else "disagree")
            known = entry.get("known", {}).get(e)
            rec["expected"] = rec["outcome"] in ("agree", "refused") or rec["outcome"] == known
            out.append(rec)
            print(f"{rec['outcome']:9s} {e:6s} {rel}  {rec.get('err_rms', '')}"
                  f"{'' if rec['expected'] else '   <-- NEW'}", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--circuits", default="*", help="comma-separated globs on circuits/<family>/<file>")
    ap.add_argument("--engines", default=",".join(ENGINES))
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    pats = [p if p.startswith("circuits/") else f"circuits/{p}" for p in args.circuits.split(",")]
    recs = run(pats, args.engines.split(","), load_triage())
    Path(args.out).write_text(json.dumps(recs, indent=1), encoding="utf-8")
    bad = [r for r in recs if not r["expected"]]
    print(f"{len(recs)} cells, {len(bad)} unexpected -> {args.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
