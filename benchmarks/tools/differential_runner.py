#!/usr/bin/env python3
"""Differential runner: benchmark circuit × engine, time-weighted error.

Each cell is ``agree`` / ``disagree`` (``err_rms`` against the reference
within / outside the tolerance), ``refused`` (the engine's typed refusal:
``pulsim.UnsupportedError`` when it exists, else ``ValueError`` /
``NotImplementedError``) or ``error`` (anything else). Reference, first
available: ``trbdf2`` at
``rtol = 1e-8``, ``pwl`` at ``dt/4`` when ``dt/2`` and ``dt/4`` agree within
a quarter of the tolerance. The first stored sample is excluded from every
comparison (on pwl / trbdf2 it is the initial state, not t = 0+).

The GATE (agree / disagree, and every ``known:`` freeze) is the strict
``interp`` measure: every observable, every interval, linearly
interpolated on the union of both grids. Next to it each cell records the
``edge`` DIAGNOSTIC, which explains a disagreement and never decides one:
the error with the steps that hold a recorded switch-state change (either
trace) and the nodes a voltage source ties to ground left out, the
excluded fraction of the compared span, and the edge-timing mismatch
between matched switching instants. Above ``EDGE_CAP`` excluded, or when
one trace records ``CHATTER`` times the other's switch-state changes, the
diagnostic is the word ``undersampled`` / ``chatter`` (``missed edges`` when
one trace never switches), not a number. The
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
import math
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


def _alias(engine: str) -> str:
    """'exact' is the canonical name of 'dsed': same manifest entries."""
    return "dsed" if engine == "exact" else engine


def default_tol(engine: str, switched: bool) -> float:
    if not switched:
        return 1e-3
    return 2e-2 if engine == "pwl" else 1e-3


def source_forced(case) -> set:
    """``V(node)`` of every node an independent voltage source ties to ground:
    the source imposes it, nothing is computed there."""
    out = set()
    for c in case._components:
        if c["type"] == "voltage_source":
            a, k = (v1_reader._node(n) for n in c["nodes"][:2])
            if "gnd" in (a, k) and a != k:
                out.add(f"V({k if a == 'gnd' else a})")
    return out


def observables_of(case) -> list:
    cols = list(case.observables) or ([case.validation["observable"]]
                                      if case.validation.get("observable") else [])
    if not cols:
        b, _, _ = case.build()
        cols = [s for s in b.state_var_names() if s.startswith(("V(", "I("))]
    return cols


def switch_record(res, t):
    """The recorded switch states, ``(n_samples, n_switches)``, or None."""
    try:
        s = np.asarray(res.switch_states)
    except Exception:  # noqa: BLE001 - no record
        return None
    return s if s.ndim == 2 and s.shape[1] > 0 and s.shape[0] == t.size else None


def switching_steps(s, t):
    """``(t_k, t_k+1)`` of every step across which the record ``s`` changes."""
    if s is None:
        return []
    k = np.flatnonzero(np.any(s[1:] != s[:-1], axis=1))
    return [(t[i], t[i + 1]) for i in k if t[i + 1] > t[i]]


def traces(res, cols):
    """``(t, {col: y}, switch record or None)``."""
    t = np.asarray(res.times, dtype=float)
    return (t, {c: np.asarray(v1_reader.observable(res, c), dtype=float) for c in cols},
            switch_record(res, t))


EDGE_CAP = 0.10   # excluded fraction above which the edge diagnostic is not a number
CHATTER = 2.0     # a record with this many times the other's changes chatters


def _errs(tr, ref, c, scale, exclude=()):
    t, ys, _ = tr
    tr_, yr, _ = ref
    kw = dict(abs_scale=max(scale * np.abs(yr[c]).max(), 1e-9), exclude_t0=True, exclude=exclude)
    r = (tr_, yr[c])
    return metrics.err_rms(t, ys[c], r, **kw), metrics.err_max(t, ys[c], r, **kw)


def compare(tr, ref, cols, scale):
    """The gate: worst strict (err_rms, err_max) over the observables."""
    return max(_errs(tr, ref, c, scale) for c in cols)


def _covered(iv, lo, hi):
    """Length of the union of the ``(a, b)`` intervals inside ``[lo, hi]``."""
    tot, end = 0.0, lo
    for a, b in sorted(iv):
        a, b = max(a, end), min(b, hi)
        if b > a:
            tot, end = tot + b - a, b
    return tot


def _instants(s, t, j, v):
    """Sorted instants (first sample in the new state) at which switch ``j``
    enters state ``v``."""
    k = np.flatnonzero((s[1:, j] != s[:-1, j]) & (s[1:, j] == v))
    return t[k + 1]


def _nearest(x, y):
    """|x_i - nearest y| for each x_i (``inf`` when ``y`` is empty)."""
    if not y.size:
        return np.full(x.shape, math.inf)
    k = np.clip(np.searchsorted(y, x), 1, y.size - 1) if y.size > 1 else np.zeros(x.shape, int)
    return np.minimum(np.abs(x - y[k]), np.abs(x - y[np.maximum(k - 1, 0)]))


def edge_timing(t, sa, tr_, sb, hi=math.inf):
    """max |t_edge(a) - t_edge(b)| over switching instants matched per switch
    and new state (nearest, both ways); ``inf`` when an instant before ``hi``
    has no partner (an edge at the last sample has none in a trace that ends
    there), None without comparable records."""
    if sa is None or sb is None or sa.shape[1] != sb.shape[1]:
        return None
    worst = 0.0
    for j in range(sa.shape[1]):
        for v in np.union1d(np.unique(sa[:, j]), np.unique(sb[:, j])):
            ea, eb = _instants(sa, t, j, v), _instants(sb, tr_, j, v)
            for x, y in ((ea, eb), (eb, ea)):
                x = x[x < hi]
                if x.size:
                    worst = max(worst, float(_nearest(x, y).max()))
    return worst


def diagnose(tr, ref, cols, scale, forced=(), step=None):
    """The edge diagnostic of one cell (never the gate)."""
    t, _, sa = tr
    tr_, _, sb = ref
    lo, hi = max(t[np.searchsorted(t, t[0], side="right")], tr_[0]), min(t[-1], tr_[-1])
    excl = switching_steps(sa, t) + switching_steps(sb, tr_)
    frac = _covered(excl, lo, hi) / (hi - lo) if hi > lo else 1.0
    worst = max(cols, key=lambda c: _errs(tr, ref, c, scale)[0])
    fa, fb = (int(np.sum(s[1:] != s[:-1])) if s is not None else 0 for s in (sa, sb))
    d = dict(worst=worst, excluded_frac=frac, edge_rms=None,
             edge_timing_s=edge_timing(t, sa, tr_, sb, hi), edge_timing_steps=None)
    busy = sa if fa >= fb else sb
    if sa is not None and sb is not None and min(fa, fb) == 0 and max(fa, fb) > 1:
        # one trace never switches: chatter if the other mostly undoes each change
        # at the next sample, else it switches where this one does not
        rev = int(np.sum((busy[1:-1] != busy[:-2]) & (busy[2:] == busy[:-2])))
        d["edge"] = "chatter" if 2 * rev > max(fa, fb) else "missed edges"
    elif min(fa, fb) > 0 and max(fa, fb) > CHATTER * min(fa, fb):
        d["edge"] = "chatter"
    elif frac > EDGE_CAP:
        d["edge"] = "undersampled"
    else:
        d["edge"] = "ok"
        d["edge_rms"] = max(_errs(tr, ref, c, scale, excl)[0]
                            for c in ([c for c in cols if c not in forced] or cols))
    if step and d["edge_timing_s"] is not None:
        d["edge_timing_steps"] = d["edge_timing_s"] / step
    return d


def classify(rec, res, case, cols, ref, tol, step=None):
    """Fill ``rec`` with the gate outcome and the edge diagnostic."""
    tr = traces(res, cols)
    er, em = compare(tr, ref, cols, 1e-3)
    rec.update(measure="interp", err_rms=er, err_max=em,
               outcome="agree" if er <= tol else "disagree",
               **diagnose(tr, ref, cols, 1e-3, source_forced(case), step))
    return rec


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
        tols = {e: entry.get("tol", {}).get(_alias(e), default_tol(e, switched)) for e in engines}
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
                classify(rec, res, case, cols, ref, tols[e], case.dt if e == "pwl" else None)
            known = entry.get("known", {}).get(_alias(e))
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
