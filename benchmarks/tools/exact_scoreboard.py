#!/usr/bin/env python3
"""The exact scoreboard: every circuit the exact truth solves (conductance
form) x the five harness configurations and the extra ones below, graded with the ``exact`` measure
against the truth, beside the strict ``interp`` verdict of the harness.

    exact_scoreboard.py run [--circuits GLOB,...]   # per-circuit records (resumable)
    exact_scoreboard.py twins                       # ideal twins: which converged tool is wrong
    exact_scoreboard.py render [--previous REV]     # EXACT_SCOREBOARD.md + exact_scoreboard.json
                                                    # (+ cells moved > 2x against REV's json)

The ``exact`` measure (``metrics.err_exact``): per graded column, the
time-weighted RMS over the engine's OWN samples of the distance to the
truth evaluated exactly at those times (``exact_truth.Solver.at_float``),
divided by the truth's peak-to-peak over the run (floored like the gate at
1e-3 of its peak; raised to the truth's range at the engine's samples).
Every truth segment within DELTA = 1e-3 dt of a sample is a candidate (its
formula continued to the sample time), so a sample on an event is compared
with its own one-sided limit, left (dsed) or right; a sample that matches a
branch across an event is charged the event's displacement, jump^2 x gap
(its true L2 cost), so the window hides no timing error. The first sample
is excluded (as the gate). Only the samples are graded: a `measure` verdict
says the engine's samples are right, not that it resolves the waveform
between them. Input columns (a node a source
drives, no circuit unknown in it, e.g. V(ctrl)) are not graded. The cell is
the worst graded column. The strict gate, its tolerances and freezes are
untouched: this is a second measure next to it.
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

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import exact_truth as et  # noqa: E402
import metrics  # noqa: E402
import next_defaults as nd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / ".scratch" / "exact_scoreboard"
OUT_MD = ROOT / "EXACT_SCOREBOARD.md"
OUT_JSON = ROOT / "exact_scoreboard.json"
DELTA = 1e-3                     # snap window, in netlist steps
SELF_N = 5000                    # self-score: the smooth part's rows, strided to at most this many
ONSET = 0.05                     # the error's onset: first sample off by 5 % of tol
# solved directly (20,000 schedule events, ~20 s); `truth` writes no record for it (its dt/32 CSV is 3.2 M rows)
EXTRA = ("circuits/thermal/long_run_drift_buck.yaml",)
# Scored here only, never by the strict reference gate (reference_status.CONFIGS):
# (engine, simulate() keywords, branch form via next_defaults.builder_form).
# ESDIRK4(3) on the variable-step engine (add-high-order-stiff-integrator), and
# `<engine>+next` = the release profile of that engine (next_defaults.py: its
# keywords + form='branch' + the PLAN library defaults), the same cells as the
# rehearsal's `<engine>+next`.
EXTRA_CONFIGS = {"trbdf2+esdirk43": ("trbdf2", dict(nd.EXTRA["esdirk43"][1]), False),
                 **{f"{e}+next": (e, nd.profile_kwargs(e), True) for e in ("pwl", "trbdf2", "dsed")}}
THRESHOLDS = (1e-3, 1e-6, 1e-9)
TWINS = ("circuits/diodes/diode_rectifier.yaml", "circuits/switching/isolated/flyback_converter.yaml")


def all_configs():
    """label -> (engine, simulate() keywords, branch form): the strict five, then the extras."""
    import reference_status as rs
    return {**{k: (e, dict(kw), False) for k, (e, kw) in rs.CONFIGS.items()}, **EXTRA_CONFIGS}


def solved_circuits():
    out = []
    for f in sorted((ROOT / "truth").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("form") == "conductance" and d.get("supported"):
            out.append(d["circuit"])
    out += [c for c in EXTRA if c not in out]
    return out


def truth_dps(rel, form="conductance"):
    """The working precision the stored truth record used (45 where 30 cannot select the mode)."""
    f = ROOT / "truth" / f"{Path(rel).stem}{'' if form == 'conductance' else '.' + form}.json"
    return json.loads(f.read_text(encoding="utf-8"))["dps"][0] if f.exists() else 30


def graded_columns(sol, cols):
    """(graded, inputs, unknown): input = no circuit unknown in the column."""
    g, inp, unk = [], [], []
    for c in cols:
        if c not in sol.ck.observables:
            unk.append(c)
            continue
        r, waves = sol.obs_row(c)
        (inp if waves and not any(r) else g).append(c)
    return g, inp, unk


def spans(sol, cols):
    """The truth's peak-to-peak per column (dt/8 grid + every event both
    sides), floored at 1e-3 of its peak and 1e-9."""
    ck = sol.ck
    n = int(round(float(ck.t_end) / float(ck.dt) * 8))
    g = np.union1d(np.linspace(0.0, float(ck.t_end), n + 1),
                   [float(s.t0) for s in sol.segs])
    v = sol.at_float(cols, g, 0.0)
    out = {}
    for c in cols:
        x = v[c][np.isfinite(v[c])]
        out[c] = max(float(np.ptp(x)), 1e-3 * float(np.abs(x).max()), 1e-9)
    return out


def event_near(sol, t):
    """The last truth event at or before t: time, cause, flips."""
    ev = [e for e in sol.events[1:] if float(e["t"]) <= t]
    if not ev:
        return None
    e = ev[-1]
    return dict(t=float(e["t"]), cause=e["cause"], flips=et.flips(sol.ck, e["prev"], e["mode"]))


def _old_exact(sol, tr, cols, sp, dt):
    """The previous definition (``metrics.err_exact`` on the two-sided trace,
    snap window DELTA, 0+ row graded), kept for one release beside the new
    one. Returns (per-column score, span per column)."""
    t = tr[0]
    keep = t <= float(sol.ck.t_end) + DELTA * dt
    t = t[keep]
    cand, gap = sol.at_float(cols, t, DELTA * dt, gaps=True)
    own = np.argmin(np.where(np.isnan(gap), np.inf, gap), axis=1)
    sp = {c: max(sp[c], float(np.ptp(cand[c][np.arange(t.size), own]))) for c in cols}  # peaks between span-grid points
    return {c: metrics.err_exact(t, tr[1][c][keep], cand[c], span=sp[c], gap=gap) for c in cols}, sp


def _one_sided(sol, cols, ts):
    """{col: (left, right)} of the truth at float instants ``ts`` (equal where it is continuous)."""
    ts = np.asarray(ts, dtype=float)
    if not ts.size:
        return {c: (np.zeros(0), np.zeros(0)) for c in cols}
    cand = sol.at_float(cols, ts, 0.0)
    out = {}
    for c in cols:
        v = cand[c]
        last = v.shape[1] - 1 - np.argmax(np.isfinite(v[:, ::-1]), axis=1)
        out[c] = (v[:, 0], v[np.arange(ts.size), last])
    return out


def switch_names(sol):
    """Device name of each switch-record column (``builder.switch_index_of``)."""
    import v1_reader
    b, _, _ = v1_reader.load(sol.ck.path).build()
    names = {}
    for dv in sol.ck.devs:
        try:
            names[b.switch_index_of(dv["name"])] = dv["name"]
        except Exception:  # noqa: BLE001 - a device the builder does not index: its flips are not matched
            pass
    return names


def event_record(sol, res, cols, t_end, names):
    """The truth's device events and the engine's, with this column's
    one-sided values, for ``metrics.exact_parts``.

    Truth: every event (``sol.events``) where a device flips; a source break
    with no flip is not an event here (the engine is handed its instant, and
    the rows around it stay in the smooth part). Engine: every change of the
    switch record between consecutive rows of ``with_event_limits(res)``
    (stored rows, the split's inner events, right limits one ulp after);
    the instant is the earlier row's time (the step after it ran in the new
    state), the left value that row, the right value the later row when it
    is a right limit (one ulp later), else not recorded. Every other right
    limit is a ``right_rows`` entry. Returns per column the keyword
    arguments of ``exact_parts`` besides the rows."""
    import v1_reader
    from pulsim._result_views import with_event_limits
    ck = sol.ck
    tev = [(float(e["t"]), f"{dv['name']}{'+' if b else '-'}")
           for e in sol.events[1:] if float(e["t"]) <= t_end
           for dv, a, b in zip(ck.devs, e["prev"], e["mode"]) if a != b]
    view = with_event_limits(res)
    tv = np.asarray(view.times, dtype=float)
    sv = getattr(view, "switch_states", None)
    sv = None if sv is None else np.asarray(sv, dtype=bool)
    available = sv is not None and sv.ndim == 2 and sv.shape[0] == tv.size and sv.shape[1] > 0
    is_right = np.r_[False, tv[1:] == np.nextafter(tv[:-1], np.inf)]
    eev, paired = [], set()
    if available:
        for i in np.flatnonzero(np.any(sv[1:] != sv[:-1], axis=1)):
            if tv[i] > t_end:
                continue
            for j in np.flatnonzero(sv[i + 1] != sv[i]):
                if int(j) in names:
                    eev.append((float(tv[i]), f"{names[int(j)]}{'+' if sv[i + 1, j] else '-'}",
                                int(i), int(i + 1) if is_right[i + 1] else None))
                    if is_right[i + 1]:
                        paired.add(int(i + 1))
    rr = [int(r) for r in np.flatnonzero(is_right) if int(r) not in paired and tv[r - 1] <= t_end]
    yv = {c: np.asarray(v1_reader.observable(view, c), dtype=float) for c in cols}
    tl = _one_sided(sol, cols, sorted({x[0] for x in tev}))
    tl_at = {c: dict(zip(sorted({x[0] for x in tev}), zip(*tl[c]))) for c in cols}
    inst = sorted({e[0] for e in eev} | {float(tv[r - 1]) for r in rr})
    el = _one_sided(sol, cols, inst)
    el_at = {c: dict(zip(inst, zip(*el[c]))) for c in cols}
    out = {}
    for c in cols:
        y = yv[c]
        out[c] = dict(
            truth_events=[(te, k, *tl_at[c][te]) for te, k in tev],
            engine_events=[(te, k, float(y[i]), None if r is None else float(y[r]), *el_at[c][te])
                           for te, k, i, r in eev],
            right_rows=[(float(tv[r - 1]), float(y[r]), el_at[c][float(tv[r - 1])][1]) for r in rr],
            events_available=bool(available or not tev))
    return out


def grade(sol, tr, cols, sp, dt, tol, self_score=True, names=None):
    """The exact measure of one run: ``tr`` = ``differential_runner.traces(res, cols)``.

    ``exact`` = the cell's summary: per column ``metrics.exact_parts`` (smooth,
    instants, limits; the largest), the worst column; ``exact_part`` names the
    part that set it. ``exact_old`` = the previous definition (one release)."""
    import mpmath as mp
    import v1_reader
    res = tr.result
    old, sp = _old_exact(sol, tr, cols, sp, dt)
    t_end = float(sol.ck.t_end)
    t = np.asarray(res.times, dtype=float)
    keep = t <= t_end + DELTA * dt                  # rows past the run's end (a float t_end) are not graded
    t = t[keep]
    cand = sol.at_float(cols, t, 0.0)
    ev = event_record(sol, res, cols, t_end, switch_names(sol) if names is None else names)
    per, dep, worst_pt = {}, None, 0.0
    for c in cols:
        y = np.asarray(v1_reader.observable(res, c), dtype=float)[keep]
        p = metrics.exact_parts(t, y, cand[c], span=sp[c], h=dt, **ev[c])
        kept = p.pop("kept")             # the same rows for every column (events are per circuit)
        per[c] = p
        e = metrics.exact_dev(y, cand[c]) / sp[c]
        e = np.where(kept, e, 0.0)
        worst_pt = max(worst_pt, float(e.max()), p["limits"] or 0.0)
        over = np.flatnonzero(e > tol)
        if over.size and (dep is None or t[over[0]] < dep["t"]):
            dep = dict(t=float(t[over[0]]), col=c, dev=float(e[over[0]]))
        low = np.flatnonzero(e > ONSET * tol)
        if over.size and low.size and (dep.get("onset_t") is None or t[low[0]] < dep["onset_t"]):
            dep["onset_t"] = float(t[low[0]])
    if dep:
        dep["event"] = event_near(sol, dep["t"])
        if dep.get("onset_t") is not None:
            dep["onset_event"] = event_near(sol, dep["onset_t"])
    worst = max(per, key=lambda c: per[c]["summary"])
    w = per[worst]

    def part_max(name):
        xs = [(p[name], c) for c, p in per.items() if p.get(name) is not None]
        return max(xs) if xs else (None, None)
    rec = dict(exact=w["summary"], exact_col=worst, exact_part=w["part"],
               exact_per_col={c: p["summary"] for c, p in per.items()},
               smooth=part_max("smooth")[0], instants=part_max("instants")[0], limits=part_max("limits")[0],
               instants_available=w["instants_available"],
               max_dt_h=max((p.get("max_dt_h", 0.0) for p in per.values()), key=abs),
               limits_worst=dict(zip(("value", "col"), part_max("limits"))) | (
                   {"t": per[part_max("limits")[1]]["limits_worst_t"]} if part_max("limits")[1] else {}),
               n_matched=w["n_matched"], n_initial=w["n_initial"], smooth_rows=w["smooth_rows"],
               missing=w["missing"][:20], n_missing=len(w["missing"]),
               spurious=w["spurious"][:20], n_spurious=len(w["spurious"]),
               exact_old=old[max(old, key=old.get)], exact_old_col=max(old, key=old.get), exact_old_per_col=old,
               exact_max_pt=worst_pt, n_samples=int(t.size), departure=dep)
    if self_score:
        # the truth itself (mpmath `sample`, a different code path) on the smooth part's
        # rows, strided to <= SELF_N (expm segments cost one expm each)
        idx = np.flatnonzero(kept)
        idx = np.r_[0, idx[:: max(1, idx.size // SELF_N)]]
        ts = t[idx]
        rec["self"], rec["self_n"] = None, int(ts.size)
        if ts.size >= 2:
            v = sol.sample(cols, [mp.mpf(float(x)) for x in ts])
            rec["self"] = max(metrics.exact_parts(ts, np.array([float(x) for x in v[c]]), cand[c][idx],
                                                  span=sp[c], h=dt)["smooth"] for c in cols)
    return rec


def run_circuit(rel):
    import differential_runner as dr
    import reference_status as rs
    import v1_reader
    import yaml
    entry = yaml.safe_load((ROOT / "manifests" / "v2_triage.yaml").read_text(encoding="utf-8"))["circuits"][rel]
    t0 = time.time()
    sol = et.Solver(et.Circuit(ROOT / rel), truth_dps(rel)).run()
    rec = dict(circuit=rel, truth_s=round(time.time() - t0, 1), n_events=len(sol.events) - 1)
    case = v1_reader.load(ROOT / rel)
    _, sw, _ = case.build()
    switched = sw is not None or any(k in json.dumps(case._components) for k in ('"diode"', '"pwm"', '"pulse"'))
    cols = dr.observables_of(case)
    g, inp, unk = graded_columns(sol, cols)
    rec.update(dt=case.dt, graded=g, inputs=inp, not_in_truth=unk)
    if not g:
        rec["skip"] = "no graded column"
        return rec
    sp = spans(sol, g)
    tols = {e: entry.get("tol", {}).get(e, dr.default_tol(e, switched)) for e in dr.ENGINES}
    ref_name, ref = dr.reference(case, cols, min(tols.values()))
    rec["reference"] = dict(name=ref_name)
    if ref is not None:
        rec["reference"].update(grade(sol, ref, g, sp, case.dt, min(tols.values())))
    rec["cells"] = {}
    for cfg, (eng, kw, form) in all_configs().items():
        with rs.extra_kwargs(kw), nd.builder_form(form), nd.plan_defaults(form):  # form: the +next cells
            kind, res, wall, msg = dr.run_cell(case, eng)
        cell = dict(config=cfg, circuit=rel, reference=ref_name, tol=tols[eng], message=msg, wall_s=round(wall, 3),
                    engine=eng, kwargs=kw, form="branch" if form else None)
        if kind != "ran":
            cell["outcome"] = kind
        elif ref is None:
            cell.update(outcome="error", message="no converged reference")
        else:
            dr.classify(cell, res, case, cols, ref, tols[eng], case.dt if eng == "pwl" else None)
        rs._annotate(cell, entry)
        if kind == "ran":
            cell.update(grade(sol, dr.traces(res, cols), g, sp, case.dt, tols[eng]))
        rec["cells"][cfg] = {k: cell[k] for k in cell if k not in ("circuit", "reference", "worst")}
    return rec


def twins():
    """Ideal twins of the two circuits where converged tools disagree: each
    tool's refinement graded against the ideal truth."""
    import differential_runner as dr
    import ideal_switch_flip_evidence as ev
    import reference_status as rs
    import v1_reader
    out = {}
    for rel in TWINS:
        sol = et.Solver(et.Circuit(ROOT / rel, "ideal"), truth_dps(rel, "ideal")).run()
        case = v1_reader.load(ROOT / rel)
        g, _, _ = graded_columns(sol, dr.observables_of(case))
        sp = spans(sol, g)
        rows = []
        runs = [("trbdf2", dict(rtol=r, atol=1e-12), {}, f"rtol {r:g}") for r in (1e-6, 1e-8, 1e-10)]
        runs += [("pwl", dict(dt=case.dt / k), {"discontinuity_restart": True}, f"pwl+restart dt/{k}")
                 for k in (1, 4, 16)]
        for eng, kw, extra, label in runs:
            with ev.ideal_form(), rs.extra_kwargs(extra):
                try:
                    res = case.run(eng, **kw)
                except Exception as ex:  # noqa: BLE001 - recorded
                    rows.append(dict(run=label, error=f"{type(ex).__name__}: {ex}"[:200]))
                    continue
            r = grade(sol, dr.traces(res, g), g, sp, case.dt, 1e-3, self_score=False)
            rows.append(dict(run=label, exact=r["exact"], exact_col=r["exact_col"], exact_part=r["exact_part"],
                             exact_old=r["exact_old"], n=r["n_samples"], departure=r["departure"]))
            print(rel, label, r["exact"], flush=True)
        out[rel] = rows
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "_twins.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


# ----------------------------------------------------------------- render --

def _cls(c):
    """'engine error' / 'measure' for a disagree cell, 'contradicted' for an
    agree cell the truth puts above tol, '' otherwise."""
    if "exact" not in c:
        return ""
    over = c["exact"] > c["tol"]
    if c["outcome"] == "disagree":
        return "engine error" if over else "measure"
    if c["outcome"] == "agree" and over:
        return "contradicted"
    if c["outcome"] == "unverified" and over:
        return "engine error (unverified cell)"
    return ""


def _g(x):
    return "" if x is None else f"{x:.2g}"


def _git(*args):
    import subprocess
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def render(previous=None):
    recs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(RAW.glob("*.json"))
            if not f.name.startswith("_")]
    recs = [r for r in recs if "cells" in r]
    twin = json.loads((RAW / "_twins.json").read_text(encoding="utf-8")) if (RAW / "_twins.json").exists() else {}
    defs = all_configs()
    cfgs = [k for k in defs if any(k in r["cells"] for r in recs)]
    for r in recs:
        for c in r["cells"].values():
            c["class"] = _cls(c)
    head = _git("rev-parse", "--short=8", "HEAD").strip()
    configs = {k: dict(engine=defs[k][0], kwargs=defs[k][1], form="branch" if defs[k][2] else None) for k in cfgs}
    OUT_JSON.write_text(json.dumps(dict(delta_steps=DELTA, commit=head, configs=configs, circuits=recs, twins=twin),
                                   indent=1) + "\n", encoding="utf-8")
    cells = [(r, cfg, r["cells"][cfg]) for r in recs for cfg in cfgs if cfg in r["cells"]]
    selfs = [c.get("self") for _, _, c in cells if c.get("self") is not None]
    selfs += [r["reference"]["self"] for r in recs if "self" in r.get("reference", {})]
    L = ["# Exact scoreboard", "",
         f"Generated by `benchmarks/tools/exact_scoreboard.py render`, rendered on `{head}` from `run` records "
         "(the tree the runs used is named under `## Notes`; do not edit above `## Notes`). Every circuit the exact truth solves (conductance form) x "
         "the five harness configurations of `reference_status.py` and the extra configurations "
         "below, graded with the `exact` measure against the truth "
         "(`benchmarks/tools/exact_truth.py`), beside the strict `interp` verdict (unchanged).", "",
         "| configuration | engine | simulate() keywords | builder form |", "|---|---|---|---|",
         *[f"| {k} | {v['engine']} | {', '.join(f'`{a}={b!r}`' for a, b in v['kwargs'].items()) or '(defaults)'} "
           f"| {v['form'] or '(default)'} |" for k, v in configs.items()], "",
         "`exact`: per graded column, sqrt(sum w_i e_i^2 / sum w_i) / span over the engine's own "
         "samples (w_i trapezoid time weights, first sample excluded), e_i the distance to the "
         f"nearest one-sided truth value at t_i (every truth segment within {DELTA:g} dt, continued "
         "to t_i; a sample matched across an event is charged jump^2 x its time gap, the true L2 "
         "of that displacement, beyond 4 ulp), span the truth's peak-to-peak (floored at 1e-3 of "
         "its peak). Input columns (only a source in them) are not graded. Cell = worst column; a "
         "non-finite sample scores inf. No interpolation of the engine trace: what it does between "
         "its samples is not graded, so `measure` below means the engine's samples are right "
         "(the strict gate's interpolation floor, or its reference, is what disagrees).", "",
         f"No floor: the truth itself (mpmath `Solver.sample`) graded on each engine's own grid "
         f"scores at most {max(selfs):.1e} over {len(selfs)} grids (median {np.median(selfs):.1e}).", ""]
    L += findings(recs, cfgs)
    L += ["", "## Summary per configuration", "",
          "`graded` = cells that ran and were graded; `not graded` = refused / error (no `exact`).", "",
          "| configuration | graded | not graded | exact < 1e-3 | < 1e-6 | < 1e-9 | >= 1e-2 | worst five |",
          "|---|---:|---:|---:|---:|---:|---:|---|"]
    for cfg in cfgs:
        xs = sorted(((c["exact"], r["circuit"]) for r, k, c in cells if k == cfg and "exact" in c), reverse=True)
        e = np.array([x for x, _ in xs])
        ng = sum(1 for _, k, c in cells if k == cfg and "exact" not in c)
        top = ", ".join(f"{Path(n).stem} {x:.2g}" for x, n in xs[:5])
        L.append(f"| {cfg} | {len(xs)} | {ng} | " + " | ".join(str(int((e < t).sum())) for t in THRESHOLDS)
                 + f" | {int((e >= 1e-2).sum())} | {top} |")
    if previous:
        L += moved_section(recs, cfgs, previous)
    dis = [(r, k, c) for r, k, c in cells if c["outcome"] == "disagree"]
    ee = sum(c["class"] == "engine error" for _, _, c in dis)
    ms = sum(c["class"] == "measure" for _, _, c in dis)
    con = [(r, k, c) for r, k, c in cells if c["class"] == "contradicted"]
    L += ["", "## Strict `disagree` cells classified", "",
          f"{len(dis)} `disagree` cells on solved circuits: {ee} engine error (exact > tol), "
          f"{ms} measure (exact <= tol: the engine's samples are right; the gate's floor or reference is not). "
          f"{len(con)} `agree` cells contradicted by the truth (exact > tol).", "",
          "| circuit | configuration | tol | interp (gate) | exact | class |", "|---|---|---:|---:|---:|---|"]
    for r, k, c in sorted(dis + con, key=lambda x: (x[0]["circuit"], cfgs.index(x[1]))):
        L.append(f"| {r['circuit'].removeprefix('circuits/')} | {k} | {c['tol']:g} | {_g(c.get('err_rms'))} "
                 f"| {_g(c['exact'])} | {c['class'] or c['outcome']} |")
    L += ["", "## The stored reference against the truth", "",
          "The harness reference (`differential_runner.reference`: trbdf2 rtol 1e-8, else pwl dt/4), "
          "graded with the same measure.", "",
          "| circuit | reference | exact | worst column | self |", "|---|---|---:|---|---:|"]
    for r in recs:
        rf = r.get("reference", {})
        L.append(f"| {r['circuit'].removeprefix('circuits/')} | {rf.get('name')} | {_g(rf.get('exact'))} "
                 f"| {rf.get('exact_col', '')} | {_g(rf.get('self'))} |")
    if twin:
        L += ["", "## Ideal twins: the two converged tools that disagree", "",
              "Each tool's refinement on the ideal twin (`ideal_switch_flip_evidence.ideal_form`), "
              "graded against the ideal truth (`departs`: first sample off by more than 1e-3 of span, "
              "and the truth event before it).", "",
              ]
        for rel, rows in twin.items():
            fin = {fam: [x for x in rows if x["run"].startswith(fam) and "exact" in x][-1:]
                   for fam in ("rtol", "pwl+restart")}
            if all(fin.values()):
                a, b = fin["rtol"][0], fin["pwl+restart"][0]
                bad, good = (a, b) if a["exact"] > b["exact"] else (b, a)
                verdict = (f"{'trbdf2' if bad is a else 'pwl+restart'} converges to a wrong waveform"
                           if bad["exact"] > 1e-3 else "both finest runs are within 1e-3 of the ideal truth")
                for fam in ("rtol", "pwl+restart"):
                    xs = [x for x in rows if x["run"].startswith(fam) and "exact" in x]
                    if len(xs) > 1 and xs[-1]["exact"] > 2 * xs[0]["exact"]:
                        verdict += (f"; the {fam} error grows under refinement ({xs[0]['run']} "
                                    f"{xs[0]['exact']:.2g} -> {xs[-1]['run']} {xs[-1]['exact']:.2g})")
                L += [f"- {rel.removeprefix('circuits/')}: the finest {bad['run']} is {bad['exact']:.2g} from "
                      f"the ideal truth, the finest {good['run']} {good['exact']:.2g}: {verdict}."]
        L += ["", "| circuit | run | exact | worst column | departs at [s] | truth event |", "|---|---|---:|---|---:|---|"]
        for rel, rows in twin.items():
            for x in rows:
                d = x.get("departure") or {}
                e = d.get("event") or {}
                L.append(f"| {rel.removeprefix('circuits/')} | {x['run']} | {_g(x.get('exact'))} "
                         f"| {x.get('exact_col', x.get('error', ''))} | {_g(d.get('t'))} "
                         f"| {_g(e.get('t'))} {e.get('cause', '')} {e.get('flips', '')} |")
    L += ["", "## Full table", "", "exact (strict verdict, interp err).", "",
          "| circuit | ref exact | " + " | ".join(cfgs) + " |", "|---|---:|" + "---|" * len(cfgs)]
    for r in recs:
        row = []
        for k in cfgs:
            c = r["cells"].get(k)
            if c is None:
                row.append("")
            elif "exact" not in c:
                row.append(c["outcome"])
            else:
                row.append(f"{c['exact']:.2g} ({c['outcome'][:3]} {_g(c.get('err_rms'))})")
        L.append(f"| {r['circuit'].removeprefix('circuits/')} | {_g(r.get('reference', {}).get('exact'))} | "
                 + " | ".join(row) + " |")
    skipped = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(RAW.glob("*.json")) if not f.name.startswith("_")]
    skipped = [s for s in skipped if "cells" not in s]
    if skipped:
        L += ["", "Not graded: " + ", ".join(f"{s['circuit']} ({s.get('skip')})" for s in skipped) + "."]
    old = OUT_MD.read_text(encoding="utf-8") if OUT_MD.exists() else ""
    notes = old.split("\n## Notes\n", 1)[1] if "\n## Notes\n" in old else "\n(none)\n"
    OUT_MD.write_text("\n".join(L) + "\n\n## Notes\n" + notes, encoding="utf-8")


def moved_section(recs, cfgs, rev):
    """Cells whose `exact` moved by more than 2x against the json stored at git ``rev``."""
    old = json.loads(_git("show", f"{rev}:benchmarks/exact_scoreboard.json"))
    before = {(r["circuit"], k): c for r in old["circuits"] for k, c in r.get("cells", {}).items()}
    rows, same, quiet, quiet_worse, new = [], 0, 0, 0, 0
    rank = {"agree": 0, "unverified": 0, "disagree": 1, "refused": 2, "error": 3}
    for r in recs:
        for k in cfgs:
            a, b = before.get((r["circuit"], k)), r["cells"].get(k)
            if b is None:
                continue
            if a is None:
                new += 1
                continue
            ea, eb = a.get("exact"), b.get("exact")
            if ea is None or eb is None or a["outcome"] != b["outcome"]:
                if (ea is None) != (eb is None) or a["outcome"] != b["outcome"]:
                    ra, rb = rank.get(a["outcome"], 3), rank.get(b["outcome"], 3)
                    worse = rb > ra if rb != ra else eb is None
                    rows.append((r["circuit"], k, a, b, "worse" if worse else "better"))
                else:
                    same += 1
                continue
            if max(ea, eb) > 2 * max(min(ea, eb), 1e-300):
                if max(ea, eb) < 1e-12:      # both at the truth's own resolution (self-score ~1e-13)
                    quiet += 1
                    quiet_worse += eb > ea
                    continue
                rows.append((r["circuit"], k, a, b, "worse" if eb > ea else "better"))
            else:
                same += 1

    def v(c):
        return f"{c['exact']:.3g} ({c['outcome'][:3]})" if "exact" in c else c["outcome"]
    L = ["", "## Cells moved by more than 2x against the previous scoreboard", "",
         f"Previous: `benchmarks/exact_scoreboard.json` at `{rev}` "
         f"({old.get('commit', 'commit not recorded in that file')}). "
         f"{sum(x[4] == 'worse' for x in rows)} worse, {sum(x[4] == 'better' for x in rows)} better, "
         f"{same} within 2x (same outcome), {quiet} moved > 2x with both values < 1e-12 (not listed; "
         f"{quiet_worse} of them worse), {new} cells with no previous value (configuration new since), "
         f"{len(before) - len([1 for r in recs for k in cfgs if (r['circuit'], k) in before])} previous cells "
         "with no current one. A change of strict outcome is listed whatever the ratio. Worse first; what is "
         "known about the causes is in `## Notes`.", "",
         "| circuit | configuration | before | after | ratio |", "|---|---|---:|---:|---:|"]
    for circ, k, a, b, d in sorted(rows, key=lambda x: (x[4] != "worse", x[0], cfgs.index(x[1]))):
        ratio = f"{b['exact'] / a['exact']:.3g}" if "exact" in a and "exact" in b and a["exact"] > 0 else ""
        L.append(f"| {circ.removeprefix('circuits/')} | {k} | {v(a)} | {v(b)} | {ratio} |")
    return L


def findings(recs, cfgs):
    L = ["## Findings for the engine tracks", "",
         "The five largest engine errors per configuration (exact > tol). `onset`: the first "
         f"sample whose own deviation exceeds {ONSET:g} tol (fraction of span), and the last truth "
         "event at or before it (time, cause, the devices that flip there): where the engine "
         "starts to depart. `departs`: the same at tol.", ""]
    for cfg in cfgs:
        xs = [(c["exact"], r, c) for r in recs for k, c in r["cells"].items()
              if k == cfg and "exact" in c and c["exact"] > c["tol"]]
        xs.sort(key=lambda x: -x[0])
        L += [f"### {cfg}", ""]
        if not xs:
            L += ["None above tol.", ""]
            continue
        L += ["| circuit | exact | column | onset [s] | truth event at onset | departs [s] | truth event there |",
              "|---|---:|---|---:|---|---:|---|"]

        def ev(e):
            return f"{_g(e.get('t'))} {e.get('cause', '')}: {e.get('flips', '')}" if e else "start"
        for x, r, c in xs[:5]:
            d = c.get("departure") or {}
            L.append(f"| {r['circuit'].removeprefix('circuits/')} | {x:.3g} | {c['exact_col']} | "
                     f"{_g(d.get('onset_t'))} | {ev(d.get('onset_event'))} | {_g(d.get('t'))} | "
                     f"{ev(d.get('event'))} |")
        L.append("")
    return L


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--circuits", default="*")
    r.add_argument("--force", action="store_true")
    sub.add_parser("twins")
    rd = sub.add_parser("render")
    rd.add_argument("--previous", help="git revision whose exact_scoreboard.json the diff section compares with")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    sys.stdout.reconfigure(line_buffering=True)
    if a.cmd == "run":
        RAW.mkdir(parents=True, exist_ok=True)
        pats = [p if p.startswith("circuits/") else f"circuits/{p}" for p in a.circuits.split(",")]
        for rel in solved_circuits():
            f = RAW / f"{Path(rel).stem}.json"
            if not any(fnmatch.fnmatch(rel, p) for p in pats) or (f.exists() and not a.force):
                continue
            t0 = time.time()
            try:
                rec = run_circuit(rel)
            except Exception as ex:  # noqa: BLE001 - recorded, the sweep goes on
                rec = dict(circuit=rel, skip=f"{type(ex).__name__}: {ex}"[:300])
            f.write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
            print(f"{time.time() - t0:7.1f}s {rel} " + " ".join(
                f"{k}={c.get('exact', c['outcome']):.2g}" if "exact" in c else f"{k}={c['outcome']}"
                for k, c in rec.get("cells", {}).items()), flush=True)
    elif a.cmd == "twins":
        twins()
    else:
        render(a.previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
