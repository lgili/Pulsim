#!/usr/bin/env python3
"""Regenerate benchmarks/REFERENCE_STATUS.md: the differential matrix and
the analytic set per engine configuration.

    reference_status.py [--circuits GLOB,...] [--json out.json]
    reference_status.py --from-json out.json      # re-render only

Configurations: ``pwl`` (defaults), ``pwl+restart``
(``discontinuity_restart=True``), ``pwl+split`` (the rule and
``enable_substep_state_correction=True``), ``trbdf2`` and ``dsed`` (harness
defaults). Cells are classified exactly as differential_runner.py does,
against one reference per circuit (computed once, without the flag).
Text below the ``## Notes`` heading of the existing file is kept.
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import differential_runner as dr  # noqa: E402
import metrics  # noqa: E402
import v1_reader  # noqa: E402
from analytic import CASES  # noqa: E402

import pulsim  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "REFERENCE_STATUS.md"
ANALYTIC = yaml.safe_load((ROOT / "manifests" / "analytic.yaml").read_text())["cases"]
CONFIGS = {"pwl": ("pwl", {}), "pwl+restart": ("pwl", {"discontinuity_restart": True}),
           "pwl+split": ("pwl", {"discontinuity_restart": True,
                                 "enable_substep_state_correction": True}),
           "trbdf2": ("trbdf2", {}), "dsed": ("dsed", {})}
KINDS = ("agree", "disagree", "error", "refused")


@contextlib.contextmanager
def extra_kwargs(kw):
    """Inject ``kw`` into every pulsim.simulate call (the analytic cases
    take no keyword of their own)."""
    orig = pulsim.simulate
    pulsim.simulate = lambda *a, **k: orig(*a, **{**k, **kw})
    try:
        yield
    finally:
        pulsim.simulate = orig


def differential(patterns):
    triage = dr.load_triage()
    out = []
    for rel, entry in sorted(triage.items()):
        if entry["status"] != "runs" or not any(fnmatch.fnmatch(rel, p) for p in patterns):
            continue
        case = v1_reader.load(ROOT / rel)
        _, sw, _ = case.build()
        switched = sw is not None or any(k in json.dumps(case._components)
                                         for k in ('"diode"', '"pwm"', '"pulse"'))
        cols = dr.observables_of(case)
        tols = {e: entry.get("tol", {}).get(e, dr.default_tol(e, switched)) for e in dr.ENGINES}
        ref_name, ref = dr.reference(case, cols, min(tols.values()))
        for cfg, (eng, kw) in CONFIGS.items():
            with extra_kwargs(kw):
                kind, res, _, msg = dr.run_cell(case, eng)
            rec = dict(circuit=rel, config=cfg, reference=ref_name, tol=tols[eng], message=msg,
                       known=entry.get("known", {}).get(eng))
            if kind != "ran":
                rec["outcome"] = kind
            elif ref is None:
                rec.update(outcome="error", message="no converged reference")
            else:
                dr.classify(rec, res, case, cols, ref, tols[eng], case.dt if eng == "pwl" else None)
            out.append(rec)
            print(f"{rec['outcome']:9s} {cfg:12s} {rel} {rec.get('err_rms', '')}", flush=True)
    return out


def analytic():
    out = []
    for c in CASES:
        for cfg, (eng, kw) in CONFIGS.items():
            spec = ANALYTIC[c.id].get(eng, {})
            rec = dict(case=c.id, config=cfg, frozen=next(
                (f"{k}: {spec[k]}" for k in ("ceiling", "known", "refused") if k in spec), ""))
            try:
                with extra_kwargs(kw):
                    r = c.run(eng)
                e = metrics.err_rms(np.asarray(r.times), c.observe(r), c.exact,
                                    abs_scale=c.abs_scale, exclude_t0=c.exclude_t0)
                rec.update(err_rms=e, outcome="agree" if e <= spec.get("ceiling", 1e-3) else "disagree")
            except dr.REFUSAL as ex:
                rec.update(outcome="refused", message=f"{type(ex).__name__}: {ex}"[:200])
            except Exception as ex:  # noqa: BLE001 - classified
                rec.update(outcome="error", message=f"{type(ex).__name__}: {ex}"[:200])
            out.append(rec)
            print(f"{rec['outcome']:9s} {cfg:12s} {c.id} {rec.get('err_rms', '')}", flush=True)
    return out


def _counts(recs):
    rows = []
    for cfg in CONFIGS:
        n = [sum(r["config"] == cfg and r["outcome"] == k for r in recs) for k in KINDS]
        rows.append(f"| {cfg} | " + " | ".join(map(str, n)) + " |")
    return ["| configuration | " + " | ".join(KINDS) + " |", "|---|---:|---:|---:|---:|", *rows]


def _e(r):
    return f"{r['err_rms']:.3g}" if "err_rms" in r else r["outcome"]


def _g(x, fmt=".2g"):
    return "" if x is None else ("unmatched" if x == float("inf") else format(x, fmt))


def _diag(r):
    """The edge diagnostic of a cell: edge error or its status word, excluded
    fraction, edge-timing mismatch (s, engine steps)."""
    if "edge" not in r:
        return "| | | | |"
    edge = _g(r["edge_rms"], ".3g") if r["edge"] == "ok" else r["edge"]
    return (f"| {r['worst']} | {edge} | {r['excluded_frac']:.1%} | {_g(r['edge_timing_s'])} "
            f"| {_g(r['edge_timing_steps'])} |")


def render(diff, ana, notes):
    by = {(r["circuit"], r["config"]): r for r in diff}
    circuits = sorted({r["circuit"] for r in diff})
    L = ["# Reference status", "",
         "Generated by `benchmarks/tools/reference_status.py` (do not edit above",
         "`## Notes`). Outcomes as in `differential_runner.py`: `err_rms` is the",
         "time-weighted RMS error over the observables' span against the reference",
         "(trbdf2 rtol 1e-8, else pwl dt/4), tolerance 2e-2 (pwl, switched) / 1e-3.",
         "Gate: the strict `interp` measure (every observable, every interval). The edge",
         "diagnostic below explains a disagreement and never decides one.",
         "", f"## Differential matrix ({len(circuits)} circuits)", "", *_counts(diff), "",
         "### Fixed engine: restart rule and split (circuits where any of the three does not agree)", "",
         "| circuit | tol | pwl | pwl+restart | pwl+split |", "|---|---:|---:|---:|---:|"]
    for c in circuits:
        a, b, sp = by[(c, "pwl")], by[(c, "pwl+restart")], by[(c, "pwl+split")]
        if a["outcome"] != "agree" or b["outcome"] != "agree" or sp["outcome"] != "agree":
            L.append(f"| {c.removeprefix('circuits/')} | {a['tol']:g} | {_e(a)} | {_e(b)} | {_e(sp)} |")
    L += ["", "### Every cell whose outcome differs from the frozen `known:` entry", "",
          "| circuit | engine | frozen | now | err_rms |", "|---|---|---|---|---:|"]
    for r in diff:
        if r["config"] in ("pwl+restart", "pwl+split"):
            continue
        frozen = r["known"] or "agree/refused"
        if (r["known"] or None) != (r["outcome"] if r["outcome"] in ("disagree", "error") else None):
            L.append(f"| {r['circuit'].removeprefix('circuits/')} | {r['config']} | {frozen} "
                     f"| {r['outcome']} | {_e(r) if 'err_rms' in r else ''} |")
    if L[-1].startswith("|---|---|---|---|"):
        L.append("| (none) | | | | |")
    L += ["", "### Edge diagnostic of every cell that disagrees on the gate", "",
          "`worst observable`: the one with the largest strict err_rms. `edge err`: err_rms",
          "with the steps holding a recorded switch-state change (either trace) and the",
          "source-forced nodes left out; `undersampled` when more than",
          f"{dr.EDGE_CAP:.0%} of the span is excluded, `chatter` when one trace records more than",
          f"{dr.CHATTER:g}x the other's switch-state changes, `missed edges` when one never switches. `timing`: max |t_edge - t_edge,ref| over",
          "matched switching instants (`unmatched`: an instant without a partner), in s and in",
          "the engine's `dt` (pwl). Edge err within the tolerance and timing <= 1 step: the",
          "cell disagrees by edge timing; edge err above it: the states differ.", "",
          "| circuit | config | tol | err_rms | worst observable | edge err | excluded | timing (s) "
          "| timing (steps) |", "|---|---|---:|---:|---|---:|---:|---:|---:|"]
    for c in circuits:
        for k in CONFIGS:
            r = by[(c, k)]
            if r["outcome"] == "disagree":
                L.append(f"| {c.removeprefix('circuits/')} | {k} | {r['tol']:g} | {_e(r)} " + _diag(r))
    L += ["", "### All cells (err_rms or outcome)", "",
          "| circuit | " + " | ".join(CONFIGS) + " |", "|---|" + "---:|" * len(CONFIGS)]
    for c in circuits:
        L.append(f"| {c.removeprefix('circuits/')} | " + " | ".join(_e(by[(c, k)]) for k in CONFIGS) + " |")
    L += ["", f"## Analytic set ({len(CASES)} cases; agree = within the frozen ceiling, 1e-3 when none)",
          "", *_counts(ana), "", "| case | " + " | ".join(CONFIGS) + " |", "|---|" + "---:|" * len(CONFIGS)]
    aby = {(r["case"], r["config"]): r for r in ana}
    for c in CASES:
        L.append(f"| {c.id} | " + " | ".join(
            f"{_e(aby[(c.id, k)])}{' ✗' if aby[(c.id, k)]['outcome'] in ('disagree', 'error') else ''}"
            for k in CONFIGS) + " |")
    return "\n".join(L) + "\n\n## Notes\n" + notes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--circuits", default="*")
    ap.add_argument("--json", help="also write the cells to this file")
    ap.add_argument("--from-json", help="re-render from a --json file, no runs")
    args = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    if args.from_json:
        d = json.loads(Path(args.from_json).read_text())
        if d.get("measure") != "interp":
            raise SystemExit(f"{args.from_json}: not written on the strict gate (measure "
                             f"{d.get('measure', d.get('metric'))!r}); re-run it")
        diff, ana = d["differential"], d["analytic"]
        triage = dr.load_triage()                  # freezes as they are now, not as saved
        for r in diff:
            r["known"] = (triage[r["circuit"]].get("known") or {}).get(CONFIGS[r["config"]][0])
    else:
        pats = [p if p.startswith("circuits/") else f"circuits/{p}" for p in args.circuits.split(",")]
        diff, ana = differential(pats), analytic()
    if args.json:
        Path(args.json).write_text(json.dumps(dict(differential=diff, analytic=ana,
                                                   measure="interp"), indent=1))
    old = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
    notes = old.split("\n## Notes\n", 1)[1] if "\n## Notes\n" in old else "\n(none)\n"
    OUT.write_text(render(diff, ana, notes), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
