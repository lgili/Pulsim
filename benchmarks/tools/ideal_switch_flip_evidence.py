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
* ``branch+lcp`` both;
* ``auto``     ``mode_selection='auto'`` (every circuit: ``'lcp'`` where it
  applies, the settle where ``'lcp'`` refuses);
* ``ideal``    every ``add_switch`` / ``add_diode`` an IDEAL branch-form element:
  ``R_on = 0`` and ``V_f = 0`` (the netlist's ``g_off`` and ``V_th`` kept).
  That is another circuit, so its cells are graded against a reference
  regenerated for it whose convergence is shown (``converged_reference``:
  trbdf2 and pwl + restart each refined, the uncertainty is a Richardson
  estimate with the measured order);
  a cell whose tol/10 is below that uncertainty is ``not comparable`` with
  the reason;
* ``ideal+lcp`` the ideal twin with ``mode_selection='lcp'``.

Each cell is graded on the strict gate against the circuit's one reference,
exactly as reference_status.py grades it. Written to
benchmarks/IDEAL_SWITCH_FLIP_EVIDENCE.md (and ``--json``):

    PYTHONPATH=python:benchmarks python3 benchmarks/tools/ideal_switch_flip_evidence.py \
        [--resume DIR] [--circuits GLOB,...] [--json out.json] [--from-json out.json]

One command regenerates every variant on one tree; ``--resume DIR`` keeps each
circuit's cells in ``DIR`` as they are measured, so a killed run started again
measures only what is missing. Every reference carries its uncertainty
(the References section).

Nothing here changes a default; the flip itself belongs to the release.
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import differential_runner as dr  # noqa: E402
import reference_status as rs  # noqa: E402
import v1_reader  # noqa: E402

import pulsim  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "IDEAL_SWITCH_FLIP_EVIDENCE.md"
VARIANTS = ("default", "branch", "lcp", "branch+lcp", "auto", "ideal", "ideal+lcp")
# the structural refusals of an ideal circuit, by name (UnsupportedError.feature)
STRUCTURAL = ("ideal_voltage_loop", "floating_node", "dependent_states", "impulse")
RANK = {"agree": 0, "unverified": 0, "disagree": 1, "refused": 2, "error": 3,
        "not comparable": 3}


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


@contextlib.contextmanager
def ideal_form():
    """add_switch / add_diode build IDEAL branch-form elements: R_on = 0,
    V_f = 0; the netlist's g_off and V_th are kept."""
    cb = pulsim.CircuitBuilder
    sw, di = cb.add_switch, cb.add_diode

    def add_switch(self, name, a, b, g_on=1e3, g_off=1e-9, *args, **kw):
        if args or kw:
            return sw(self, name, a, b, g_on, g_off, *args, **kw)
        return self.add_ideal_switch(name, a, b, 0.0, g_off)

    def add_diode(self, name, a, c, g_on=None, g_off=None, V_th=0.0, V_f=0.0, *args, **kw):
        if kw.get("form", "conductance") != "conductance" or args:
            return di(self, name, a, c, g_on, g_off, V_th, V_f, *args, **kw)
        kw.pop("form", None)
        return di(self, name, a, c, None, 1e-9 if g_off is None else g_off,
                  V_th, 0.0, form="branch", R_on=0.0, **kw)

    cb.add_switch, cb.add_diode = add_switch, add_diode
    try:
        yield
    finally:
        cb.add_switch, cb.add_diode = sw, di


def _run_cell(case, eng):
    """dr.run_cell, keeping the refusal's feature name."""
    t0 = time.perf_counter()
    try:
        res = case.run(eng)
        return "ran", res, time.perf_counter() - t0, None, None
    except dr.REFUSAL as ex:
        return ("refused", None, time.perf_counter() - t0, f"{type(ex).__name__}: {ex}"[:300],
                getattr(ex, "feature", None))
    except Exception as ex:  # noqa: BLE001 - classified, never swallowed
        return ("error", None, time.perf_counter() - t0, f"{type(ex).__name__}: {ex}"[:300],
                getattr(ex, "feature", None))


def _has_pwl_diode(case):
    return any(c.get("type") == "diode" and "is" not in c for c in case._components)


# (tool, settings, refinement factor between successive settings)
LADDERS = (("trbdf2", (1e-8, 1e-9, 1e-10), 10.0),
           ("pwl+restart", (8, 16, 32, 64, 128, 256), 2.0))


def _ladder(case, cols, target, ladder):
    """Refine one tool until its estimated error is at most ``target`` or the
    ladder ends. Returns (name, trace, u): the finest setting run and its
    error estimate (Richardson, from the last two moves: order
    p = log_f(e1 / e2), u = e2 / (f^p - 1); inf with fewer than three settings
    or a move that does not shrink). The order is measured, not assumed: the
    interp measure converges at p ~ 0.5 across a jump (a ramp over one step)
    — taking the last move as the error would understate it 2.4x."""
    tool, levels, f = ladder
    prev, name, moves, u = None, None, [], math.inf
    for lv in levels:
        try:
            if tool == "trbdf2":
                res, nm = case.run("trbdf2", rtol=lv, atol=lv * 1e-4), f"trbdf2 rtol={lv:g}"
            else:
                with rs.extra_kwargs({"discontinuity_restart": True}):
                    res, nm = case.run("pwl", dt=case.dt / lv), f"pwl+restart dt/{lv}"
            cur = dr.traces(res, cols)
        except Exception:  # noqa: BLE001 - a tool that cannot run ends its ladder
            break
        if prev is not None:
            moves.append(max(dr._sym(prev[1], cur, c, 1e-3) for c in cols))
        prev, name = (nm, cur), nm
        if len(moves) >= 2:
            e1, e2 = moves[-2:]
            if e2 == 0.0:      # identical outputs: a saturated ladder, not a proof of 0
                u = e1
            elif e1 > e2:
                u = e2 / (f ** math.log(e1 / e2, f) - 1.0)
            else:
                u = math.inf
            if u <= target:
                break
    return (name, prev[1] if prev else None, u)


def converged_reference(case, cols, tol):
    """A reference whose own convergence is shown, with its uncertainty.

    Each tool (trbdf2 by rtol, pwl + restart by dt) is refined until its
    estimated error (``_ladder``: Richardson with the measured order) is at
    most tol/10 or its ladder ends. The reference is the finest setting of the
    tool with the smaller estimate ``u``. When the other tool's own estimate
    is below half of the two tools' distance ``e_cross``, that distance is a
    real disagreement (a tool can converge to a wrong answer: ll07's ideal twin
    on trbdf2) and ``u`` is at least ``e_cross``. A cell is graded against it
    only when ``u <= tol_cell / 10``. Returns (trace or None, metadata)."""
    got = {lad[0]: _ladder(case, cols, tol / 10, lad) for lad in LADDERS}
    meta = {f"e_{k}": v[2] for k, v in got.items()}
    ran = sorted((v[2], k) for k, v in got.items() if v[1] is not None and v[2] < math.inf)
    if not ran:
        meta.update(name=None, uncertainty=None, why="no tool has a finite error estimate (three settings, shrinking moves)")
        return None, meta
    u, best = ran[0]
    name, tr, _ = got[best]
    if len(ran) == 2:
        other = got[ran[1][1]]
        x = max(dr._sym(tr, other[1], c, 1e-3) for c in cols)
        meta["e_cross"] = x
        if other[2] <= x / 2:
            u = max(u, x)
    meta.update(name=name, uncertainty=u)
    return tr, meta


def analytic_reference(case, cols):
    """The circuit's closed form as a trace, when its validation block declares
    it exact for the netlist as built (``exact: true``; benchmark_runner's model,
    the source's edge included), else None. Uncertainty 0. Dense on the edge and
    geometric after it (ratio 1 + 7e-5): its linear interpolation adds < 1e-9 of
    the span. rl_step / inductor_chain_step: the converged ladder interpolated its
    own 31 / 156 ns grid across the 1 ns edge and graded the exact engine
    (1e-11 of the closed form) a disagreement (exact engine session 21)."""
    val = case.validation
    if val.get("type") != "analytical" or not val.get("exact") or cols != [val.get("observable")]:
        return None
    import numpy as np

    import benchmark_runner as br
    if val.get("model") != "rl_step":
        return None
    p = val.get("params") or {}
    tr = br.parse_value(p.get("t_rise", 0.0))
    t1 = case.t_end
    t = np.union1d(np.linspace(0.0, tr, 201) if tr > 0 else [0.0],
                   np.geomspace(max(tr, t1 * 1e-9), t1, 200001))
    y = np.asarray(br.analytical_rl_step(list(t), br.parse_value(p["v0"]), br.parse_value(p["r"]),
                                         br.parse_value(p["l"]), tr))
    return (t, {cols[0]: y}, None)


def default_reference(case, cols, tol):
    """The gate's own reference (dr.reference, unchanged) and its uncertainty:
    trbdf2 rtol 1e-8 against 1e-9, or pwl dt/2 against dt/4 (the symmetric
    measure)."""
    name, ref = dr.reference(case, cols, tol)
    meta = {"name": name, "uncertainty": None}
    if ref is None:
        return None, meta
    try:
        if name.startswith("trbdf2"):
            fine = dr.traces(case.run("trbdf2", rtol=1e-9, atol=1e-13), cols)
        else:
            fine = dr.traces(case.run("pwl", dt=case.dt / 2), cols)
        meta["uncertainty"] = max(dr._sym(ref, fine, c, 1e-3) for c in cols)
    except Exception as ex:  # noqa: BLE001 - stated, never assumed
        meta["why"] = f"uncertainty not measured: {type(ex).__name__}"
    return ref, meta


def _tree():
    """(commit, digest of the uncommitted tracked changes or '') of the tree this
    run measures: two different sets of edits on one commit are two trees."""
    root = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "--short=10", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        diff = subprocess.run(["git", "-C", str(root), "diff", "HEAD", "--binary"],
                              capture_output=True, check=True).stdout
    except Exception:  # noqa: BLE001 - not a checkout
        return "unknown", "unknown"
    return head, hashlib.sha256(diff).hexdigest()[:12] if diff else ""


def _cell_file(cache, rel):
    return Path(cache) / (rel.replace("/", "__") + ".json")


def run_circuit(rel, entry, want=VARIANTS, have=None):
    """Every (configuration, variant) cell of one circuit not already in
    ``have`` (resumed cells), and the circuit's references with their
    uncertainties."""
    have = have or {}
    case = v1_reader.load(ROOT / rel)
    _, sw, _ = case.build()
    switched = sw is not None or any(k in json.dumps(case._components)
                                     for k in ('"diode"', '"pwm"', '"pulse"'))
    cols = dr.observables_of(case)
    tols = {e: entry.get("tol", {}).get(e, dr.default_tol(e, switched)) for e in dr.ENGINES}
    tol = min(tols.values())
    variants = [v for v in VARIANTS if v in want]
    ref, rmeta = default_reference(case, cols, tol)
    iref, imeta = None, {"name": None, "uncertainty": None}
    an = analytic_reference(case, cols) if any(v.startswith("ideal") for v in variants) else None
    if an is not None:
        iref, imeta = an, {"name": f"analytic {case.validation['model']} (closed form)",
                           "uncertainty": 0.0}
    elif any(v.startswith("ideal") for v in variants):
        try:
            with ideal_form():
                iref, imeta = converged_reference(case, cols, tol)
        except Exception as ex:  # noqa: BLE001 - the reason is the table entry
            imeta = {"name": None, "uncertainty": None,
                     "why": f"ideal twin has no reference: {type(ex).__name__}: {ex}"[:200]}
    diode = _has_pwl_diode(case)
    out = []
    for cfg, (eng, kw) in rs.CONFIGS.items():
        for var in variants:
            if (cfg, var) in have:
                out.append(have[(cfg, var)])
                continue
            rec = dict(circuit=rel, config=cfg, variant=var, tol=tols[eng])
            ideal = var.startswith("ideal")
            meta = imeta if ideal else rmeta
            rec.update(reference=meta.get("name"), reference_uncertainty=meta.get("uncertainty"))
            if "lcp" in var and not diode:
                rec.update(outcome="n/a", message="no PWL diode: nothing to select")
                out.append(rec)
                continue
            ms = "lcp" if "lcp" in var else "auto" if var == "auto" else None
            k = dict(kw, **({"mode_selection": ms} if ms else {}))
            ctx = (branch_form() if var.startswith("branch") else
                   ideal_form() if ideal else contextlib.nullcontext())
            rf = iref if ideal else ref
            with ctx, rs.extra_kwargs(k):
                kind, res, wall, msg, feat = _run_cell(case, eng)
            rec.update(wall_s=wall, message=msg, feature=feat)
            if kind != "ran":
                rec["outcome"] = kind
            elif rf is None:
                rec.update(outcome="not comparable" if ideal else "error",
                           message=meta.get("why") or "no converged reference")
            elif ideal and not meta["uncertainty"] <= tols[eng] / 10:
                rec.update(outcome="not comparable",
                           message=f"reference uncertainty {meta['uncertainty']:.3g} > tol/10 "
                                   f"({tols[eng] / 10:.3g}): {meta['name']}")
            else:
                dr.classify(rec, res, case, cols, rf, tols[eng],
                            case.dt if eng == "pwl" else None)
                dr.mark_suspect(rec, entry)
            out.append({k2: v for k2, v in rec.items()
                        if k2 in ("circuit", "config", "variant", "tol", "outcome", "err_rms",
                                  "edge", "edge_rms", "wall_s", "message", "feature",
                                  "reference", "reference_uncertainty")})
            print(f"{rec['outcome']:9s} {cfg:12s} {var:11s} {rel} {rec.get('err_rms', '')}",
                  flush=True)
    return out, {"default": rmeta, "ideal": imeta}


def run(patterns, want=VARIANTS, cache=None):
    """Every circuit matching ``patterns``. With ``cache``, each circuit's cells
    are written to ``cache/<circuit>.json`` once that circuit is done, and a
    later run measures only the cells missing there (resumable per circuit; a
    cell measured on another tree is measured again; cached cells of variants
    not in ``want`` are kept)."""
    tree = _tree()
    out, refs = [], {}
    for rel, entry in sorted(dr.load_triage().items()):
        if entry["status"] != "runs" or not any(fnmatch.fnmatch(rel, p) for p in patterns):
            continue
        have, f = {}, _cell_file(cache, rel) if cache else None
        if f is not None and f.exists():
            old = json.loads(f.read_text(encoding="utf-8"))
            if old.get("tree") == list(tree):
                have = {(r["config"], r["variant"]): r for r in old["cells"]}
        if have and all((c, v) in have for c in rs.CONFIGS for v in want):
            recs, meta = [have[(c, v)] for c in rs.CONFIGS for v in VARIANTS if v in want], old["refs"]
        else:
            recs, meta = run_circuit(rel, entry, want, have)
            if have and not any(v.startswith("ideal") for v in want):
                meta["ideal"] = old["refs"].get("ideal", meta["ideal"])   # kept with its cells
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(".tmp")
            kept = [r for k, r in have.items() if k[1] not in want]
            tmp.write_text(json.dumps({"tree": list(tree), "refs": meta, "cells": recs + kept},
                                      indent=1),
                           encoding="utf-8")
            tmp.replace(f)
        out += recs
        refs[rel] = meta
    return out, refs, tree


def _cmp(a, b):
    """better / same / worse / refused of cell b against cell a."""
    if b["outcome"] in ("n/a", "not comparable"):
        return b["outcome"]
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


def render(recs, refs=None, tree=("unknown", "")):
    by = {(r["circuit"], r["config"], r["variant"]): r for r in recs}
    circuits = sorted({r["circuit"] for r in recs})
    cfgs = [c for c in rs.CONFIGS if any(r["config"] == c for r in recs)]
    L = ["# Ideal switch / complementarity: default-flip evidence", "",
         "Generated by `benchmarks/tools/ideal_switch_flip_evidence.py` (do not edit), every",
         f"variant in one run on one tree: `{tree[0]}`{f' (with uncommitted changes {tree[1]})' if tree[1] else ''}.",
         "Regenerate: `PYTHONPATH=python:benchmarks python3 benchmarks/tools/"
         "ideal_switch_flip_evidence.py --resume <dir>`.", "",
         "Every reference circuit of the differential matrix, every configuration, graded",
         "on the strict gate against the circuit's one reference (as",
         "`REFERENCE_STATUS.md`). Variants: `branch` = every `add_switch` / `add_diode` in",
         "the branch form with the same values (`R_on = 1/g_on`, same `g_off`, design D6);",
         "`lcp` = `mode_selection='lcp'` (only where a PWL diode exists; `n/a` otherwise);",
         "`branch+lcp` both; `auto` = `mode_selection='auto'` (every circuit). Comparison of each variant against `default` in the same",
         "configuration: `same` (same outcome and err_rms within 1 %), `better` / `worse`",
         "(outcome rank, else err_rms beyond 1 %), `refused` / `error` (the default ran).",
         "`agree` counts `unverified` (a suspect reference) too. The branch variant has",
         "finite R_on = 1/g_on (each netlist gives g_on); the R_on = 0 twin is the `ideal`",
         "section below. MOSFET / IGBT helpers and",
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
                if c in ("n/a", "not comparable"):
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
    L += _render_ideal(by, circuits, cfgs)
    L += _render_refs(refs or {})
    return "\n".join(L) + "\n"


def _u(m):
    return f"{m['uncertainty']:.2g}" if m.get("uncertainty") is not None else "not stated"


def _render_refs(refs):
    """Every reference and its uncertainty (a reference without one grades nothing)."""
    if not refs:
        return []
    L = ["", "## References and their uncertainty", "",
         "`default`: the gate's reference (`differential_runner.reference`, unchanged); its",
         "uncertainty is its distance to the next refinement (trbdf2 rtol 1e-9, or pwl dt/2",
         "against dt/4). `ideal twin`: each tool refined",
         "until its error estimate is at most tol/10 or its ladder ends (trbdf2 rtol 1e-8 → 1e-10,",
         "pwl + restart dt/8 → dt/256); the estimate is Richardson's with the order measured from",
         "the last two moves (the interp measure converges at order ~0.5 across a jump). The",
         "reference is the tool with the smaller estimate u, raised to the two tools' distance",
         "`e_cross` when the other tool's estimate is below half of it. A cell is graded only",
         "when u <= its tol/10, else `not comparable`.",
         "All distances are the symmetric measure (`metrics.err_rms_symmetric`).", "",
         "| circuit | default reference | u | ideal-twin reference | u | u trbdf2 | u pwl | e cross |",
         "|---|---|---:|---|---:|---:|---:|---:|"]
    g = lambda m, k: f"{m[k]:.2g}" if m.get(k) not in (None, math.inf) else ""  # noqa: E731
    for rel in sorted(refs):
        d, i = refs[rel].get("default", {}), refs[rel].get("ideal", {})
        L.append(f"| {rel.removeprefix('circuits/')} | {d.get('name') or 'none'} | {_u(d)} "
                 f"| {i.get('name') or 'none'} | {_u(i)} | {g(i, 'e_trbdf2')} "
                 f"| {g(i, 'e_pwl+restart')} | {g(i, 'e_cross')} |")
    return L


def _render_ideal(by, circuits, cfgs):
    """The R_on = 0 section: what the release decides the flip on."""
    if not any(v.startswith("ideal") for (_, _, v) in by):
        return []
    L = ["", "## The ideal switch (R_on = 0, V_f = 0)", "",
         "Each `ideal` cell is graded against the ideal twin's own reference, one whose",
         "convergence is shown (see References: its uncertainty is a Richardson estimate),",
         "then compared with the `default` cell of the same configuration (graded against the",
         "original reference) as above. `not comparable`: the ideal twin's reference uncertainty",
         "is above the cell's tol/10 (the reason is in the cause column). Structural refusals",
         "are counted by `UnsupportedError.feature`.", "",
         "| configuration | variant | better | same | worse | refused | error | not comparable "
         "| " + " | ".join(STRUCTURAL) + " | other refusal | median wall ratio |",
         "|---|---|---:|---:|---:|---:|---:|---:|" + "---:|" * len(STRUCTURAL) + "---:|---:|"]
    for cfg in cfgs:
        for var in ("ideal", "ideal+lcp"):
            n = dict.fromkeys(("better", "same", "worse", "refused", "error", "not comparable"), 0)
            feats = dict.fromkeys(STRUCTURAL, 0)
            other, ratios, rows = 0, [], 0
            for c in circuits:
                r, d = by.get((c, cfg, var)), by.get((c, cfg, "default"))
                if r is None or d is None or r["outcome"] == "n/a":
                    continue
                rows += 1
                k = _cmp(d, r)
                k = "same" if k.startswith("same") else k
                if r["outcome"] in ("refused", "error"):
                    k = r["outcome"]
                    if r.get("feature") in feats:
                        feats[r["feature"]] += 1
                    else:
                        other += 1
                n[k] = n.get(k, 0) + 1
                if (d.get("wall_s") and r.get("wall_s") and r["outcome"] in ("agree", "disagree")
                        and d["outcome"] in ("agree", "disagree", "unverified")):
                    ratios.append(r["wall_s"] / d["wall_s"])
            if not rows:
                continue
            ratio = f"{statistics.median(ratios):.3f}" if ratios else ""
            L.append(f"| {cfg} | {var} | {n['better']} | {n['same']} | {n['worse']} | {n['refused']} "
                     f"| {n['error']} | {n['not comparable']} | "
                     + " | ".join(str(feats[f]) for f in STRUCTURAL) + f" | {other} | {ratio} |")
    L += ["", "### Every ideal cell that is worse, refused, errors or is not comparable", "",
          "| circuit | config | variant | default | ideal | err default | err ideal | feature | cause |",
          "|---|---|---|---|---|---:|---:|---|---|"]
    for c in circuits:
        for cfg in cfgs:
            for var in ("ideal", "ideal+lcp"):
                r, d = by.get((c, cfg, var)), by.get((c, cfg, "default"))
                if r is None or d is None or r["outcome"] == "n/a":
                    continue
                k = _cmp(d, r)
                if k in ("worse", "refused", "error", "not comparable") or r["outcome"] in (
                        "refused", "error"):
                    msg = (r.get("message") or "").replace("|", "/").replace("\n", " ")[:140]
                    L.append(f"| {c.removeprefix('circuits/')} | {cfg} | {var} | {d['outcome']} "
                             f"| {r['outcome']} | {rs._e(d) if 'err_rms' in d else ''} "
                             f"| {rs._e(r) if 'err_rms' in r else ''} | {r.get('feature') or ''} "
                             f"| {msg} |")
    return L


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--circuits", default="*")
    ap.add_argument("--json", help="also write every cell (and the references) here")
    ap.add_argument("--from-json", help="render an earlier --json, measure nothing")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--resume", metavar="DIR",
                    help="per-circuit cell cache: a killed run started again with the same "
                         "DIR measures only the missing cells (same tree only)")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    if a.from_json:
        blob = json.loads(Path(a.from_json).read_text(encoding="utf-8"))
        recs, refs, tree = blob["cells"], blob["refs"], tuple(blob["tree"])
    else:
        pats = [p if p.startswith("circuits/") else f"circuits/{p}" for p in a.circuits.split(",")]
        recs, refs, tree = run(pats, tuple(a.variants.split(",")), a.resume)
        if a.json:
            Path(a.json).write_text(json.dumps({"tree": list(tree), "refs": refs, "cells": recs},
                                               indent=1), encoding="utf-8")
    Path(a.out).write_text(render(recs, refs, tree), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
