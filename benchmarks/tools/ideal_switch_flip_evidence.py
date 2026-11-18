#!/usr/bin/env python3
"""Evidence for the default flip of add-branch-form-ideal-switch (task 9) and
add-complementarity-mode-selection (task 9): every reference circuit of the
differential matrix, every configuration of reference_status.py, four
variants of each cell:

* ``default``  today's defaults (conductance form, settle + guard);
* ``branch``   every ``add_switch`` / ``add_diode`` in the branch form with the
  same values (``R_on = 1/g_on``, the same ``g_off``): what design D6 makes of
  a netlist that gives its values (the v1 reader always does);
* ``lcp``      ``mode_selection='lcp'`` (only on a circuit with a PWL diode:
  without one the default selection has nothing to decide);
* ``branch+lcp`` both.

Each cell is graded on the strict gate against the circuit's one reference,
exactly as reference_status.py grades it. Written to
benchmarks/IDEAL_SWITCH_FLIP_EVIDENCE.md (and ``--json``):

    PYTHONPATH=python:benchmarks python3 benchmarks/tools/ideal_switch_flip_evidence.py \
        [--circuits GLOB,...] [--json out.json] [--from-json out.json]

Nothing here changes a default; the flip itself belongs to the release.
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import json
import statistics
import sys
import warnings
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import differential_runner as dr  # noqa: E402
import reference_status as rs  # noqa: E402
import v1_reader  # noqa: E402

import pulsim  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "IDEAL_SWITCH_FLIP_EVIDENCE.md"
VARIANTS = ("default", "branch", "lcp", "branch+lcp")
RANK = {"agree": 0, "unverified": 0, "disagree": 1, "refused": 2, "error": 3}


@contextlib.contextmanager
def branch_form():
    """add_switch / add_diode build the branch form with the same values."""
    cb = pulsim.CircuitBuilder
    sw, di = cb.add_switch, cb.add_diode

    def add_switch(self, name, a, b, g_on=1e3, g_off=1e-9, *args, **kw):
        if args or kw:
            return sw(self, name, a, b, g_on, g_off, *args, **kw)
        return self.add_ideal_switch(name, a, b, 1.0 / g_on, g_off)

    def add_diode(self, name, a, c, g_on=None, g_off=None, *args, **kw):
        if kw.get("form", "conductance") != "conductance" or args:
            return di(self, name, a, c, g_on, g_off, *args, **kw)
        kw.pop("form", None)
        g_on = 1e3 if g_on is None else g_on
        return di(self, name, a, c, None, 1e-9 if g_off is None else g_off,
                  form="branch", R_on=1.0 / g_on, **kw)

    cb.add_switch, cb.add_diode = add_switch, add_diode
    try:
        yield
    finally:
        cb.add_switch, cb.add_diode = sw, di


def _has_pwl_diode(case):
    return any(c.get("type") == "diode" and "is" not in c for c in case._components)


def run(patterns):
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
        diode = _has_pwl_diode(case)
        for cfg, (eng, kw) in rs.CONFIGS.items():
            for var in VARIANTS:
                rec = dict(circuit=rel, config=cfg, variant=var, tol=tols[eng])
                if "lcp" in var and not diode:
                    rec.update(outcome="n/a", message="no PWL diode: nothing to select")
                    out.append(rec)
                    continue
                k = dict(kw, **({"mode_selection": "lcp"} if "lcp" in var else {}))
                ctx = branch_form() if var.startswith("branch") else contextlib.nullcontext()
                with ctx, rs.extra_kwargs(k):
                    kind, res, wall, msg = dr.run_cell(case, eng)
                rec.update(wall_s=wall, message=msg)
                if kind != "ran":
                    rec["outcome"] = kind
                elif ref is None:
                    rec.update(outcome="error", message="no converged reference")
                else:
                    dr.classify(rec, res, case, cols, ref, tols[eng],
                                case.dt if eng == "pwl" else None)
                    dr.mark_suspect(rec, entry)
                out.append({k2: v for k2, v in rec.items()
                            if k2 in ("circuit", "config", "variant", "tol", "outcome",
                                      "err_rms", "wall_s", "message")})
                print(f"{rec['outcome']:9s} {cfg:12s} {var:11s} {rel} {rec.get('err_rms', '')}",
                      flush=True)
    return out


def _cmp(a, b):
    """better / same / worse / refused of cell b against cell a."""
    if b["outcome"] == "n/a":
        return "n/a"
    if b["outcome"] in ("refused", "error") and a["outcome"] not in ("refused", "error"):
        return b["outcome"]
    ra, rb = RANK.get(a["outcome"], 3), RANK.get(b["outcome"], 3)
    if rb != ra:
        return "better" if rb < ra else "worse"
    ea, eb = a.get("err_rms"), b.get("err_rms")
    if ea is None or eb is None or ea == eb:
        return "same"
    # within 1 % (or 1e-12) of each other: the same answer
    if abs(eb - ea) <= max(1e-2 * max(ea, eb), 1e-12):
        return "same"
    return "better" if eb < ea else "worse"     # graded the same way both directions


def render(recs):
    by = {(r["circuit"], r["config"], r["variant"]): r for r in recs}
    circuits = sorted({r["circuit"] for r in recs})
    cfgs = [c for c in rs.CONFIGS if any(r["config"] == c for r in recs)]
    L = ["# Ideal switch / complementarity: default-flip evidence", "",
         "Generated by `benchmarks/tools/ideal_switch_flip_evidence.py` (do not edit).",
         "Every reference circuit of the differential matrix, every configuration, graded",
         "on the strict gate against the circuit's one reference (as",
         "`REFERENCE_STATUS.md`). Variants: `branch` = every `add_switch` / `add_diode` in",
         "the branch form with the same values (`R_on = 1/g_on`, same `g_off`, design D6);",
         "`lcp` = `mode_selection='lcp'` (only where a PWL diode exists; `n/a` otherwise);",
         "`branch+lcp` both. Comparison of each variant against `default` in the same",
         "configuration: `same` (same outcome and err_rms within 1 %), `better` / `worse`",
         "(outcome rank, else err_rms beyond 1 %), `refused` / `error` (the default ran).",
         "`agree` counts `unverified` (a suspect reference) too. The branch variant has",
         "finite R_on = 1/g_on (each netlist gives g_on): no R_on = 0 port appears",
         "here; the ideal (R_on = 0) cases are in the unit tests. MOSFET / IGBT helpers and",
         "Shockley diodes keep their own models (not switched by the variant).",
         "The wall ratio is variant / default over one run per cell (a noisy",
         "signal; the median-of-7 cost gate is",
         "`openspec/changes/add-branch-form-ideal-switch/probes/measurement_gate.py`).", "",
         f"## Summary ({len(circuits)} circuits)", "",
         "| configuration | variant | agree | disagree | error | refused | n/a | better | same | worse "
         "| newly refused / error | median wall ratio |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    worse = []
    for cfg in cfgs:
        for var in VARIANTS:
            rows = [by[(c, cfg, var)] for c in circuits if (c, cfg, var) in by]
            n = {k: sum(r["outcome"] == k for r in rows)
                 for k in ("agree", "disagree", "error", "refused", "n/a")}
            n["agree"] += sum(r["outcome"] == "unverified" for r in rows)
            cmp = {k: 0 for k in ("better", "same", "worse", "new")}
            ratios = []
            for r in rows:
                d = by[(r["circuit"], cfg, "default")]
                if var == "default":
                    continue
                c = _cmp(d, r)
                if c == "n/a":
                    continue
                key = "new" if c in ("refused", "error") else ("same" if c.startswith("same") else c)
                cmp[key] += 1
                if key in ("worse", "new"):
                    worse.append((r, d, c))
                if (d.get("wall_s") and r.get("wall_s") and r["outcome"] in ("agree", "disagree")
                        and d["outcome"] in ("agree", "disagree", "unverified")):
                    ratios.append(r["wall_s"] / d["wall_s"])
            ratio = f"{statistics.median(ratios):.3f}" if ratios else ""
            if var == "default":
                L.append(f"| {cfg} | default | {n['agree']} | {n['disagree']} | {n['error']} "
                         f"| {n['refused']} | {n['n/a']} | | | | | |")
            else:
                L.append(f"| {cfg} | {var} | {n['agree']} | {n['disagree']} | {n['error']} "
                         f"| {n['refused']} | {n['n/a']} | {cmp['better']} | {cmp['same']} "
                         f"| {cmp['worse']} | {cmp['new']} | {ratio} |")
    L += ["", "## Every cell that gets worse or is refused under a variant", "",
          "| circuit | config | variant | default | variant | err default | err variant | cause |",
          "|---|---|---|---|---|---:|---:|---|"]
    for r, d, c in worse:
        msg = (r.get("message") or "").replace("|", "/").replace("\n", " ")[:160]
        L.append(f"| {r['circuit'].removeprefix('circuits/')} | {r['config']} | {r['variant']} "
                 f"| {d['outcome']} | {r['outcome']} | {rs._e(d) if 'err_rms' in d else ''} "
                 f"| {rs._e(r) if 'err_rms' in r else ''} | {msg} |")
    if not worse:
        L.append("| (none) | | | | | | | |")
    better = [(by[(c, cfg, "default")], by[(c, cfg, v)]) for c in circuits for cfg in cfgs
              for v in VARIANTS[1:] if (c, cfg, v) in by
              and _cmp(by[(c, cfg, "default")], by[(c, cfg, v)]) == "better"]
    L += ["", "## Every cell that gets better", "",
          "| circuit | config | variant | default | variant | err default | err variant |",
          "|---|---|---|---|---|---:|---:|"]
    for d, r in better:
        L.append(f"| {r['circuit'].removeprefix('circuits/')} | {r['config']} | {r['variant']} "
                 f"| {d['outcome']} | {r['outcome']} | {rs._e(d) if 'err_rms' in d else ''} "
                 f"| {rs._e(r) if 'err_rms' in r else ''} |")
    if not better:
        L.append("| (none) | | | | | | |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--circuits", default="*")
    ap.add_argument("--json")
    ap.add_argument("--from-json")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    if a.from_json:
        recs = json.loads(Path(a.from_json).read_text())
    else:
        pats = [p if p.startswith("circuits/") else f"circuits/{p}" for p in a.circuits.split(",")]
        recs = run(pats)
        if a.json:
            Path(a.json).write_text(json.dumps(recs, indent=1))
    Path(a.out).write_text(render(recs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
