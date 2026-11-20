#!/usr/bin/env python3
"""The exact scoreboard: every circuit the exact truth solves (conductance
form) x the five harness configurations and the extra ones below, graded with the ``exact`` measure
against the truth, beside the strict ``interp`` verdict of the harness.

    exact_scoreboard.py run [--circuits GLOB,...]   # per-circuit records (resumable)
    exact_scoreboard.py twins                       # ideal twins: which converged tool is wrong
    exact_scoreboard.py render [--previous REV]     # EXACT_SCOREBOARD.md + exact_scoreboard.json
                                                    # (+ cells moved > 2x against REV's json)

The ``exact`` measure (``metrics.exact_parts``, one implementation for every
tool that grades against the truth): per graded column, three parts, each a
fraction of the truth's peak-to-peak (floored like the gate at 1e-3 of its
peak; raised to the truth's range at the engine's samples):

- smooth: time-weighted RMS over the engine's STORED rows of the distance to
  the truth at those times (``exact_truth.Solver.at_float``, no snap window),
  leaving out the initial instant, the 0+ row and the rows inside a matched
  event's displacement [min(t_engine, t_truth), max(...)];
- instants: each engine device event matched to the truth's (same device and
  direction, in order, within EVENT_WINDOW netlist steps); the displacement's
  L2 cost sqrt(sum jump^2 |delta| / T) (a missing truth event charged as
  displaced to that device's next event, at most the window), and max
  |delta| / dt;
- limits: max distance of every one-sided value the engine records (left at
  its event instants, every right limit) from the truth's limits at the
  truth's instant (the engine's own when unmatched).

The cell is the largest part of the worst column (``exact``, ``exact_part``).
Unmatched events are listed, not priced. ``exact_old`` is the previous
definition (``metrics.err_exact``, snap window DELTA = 1e-3 dt), kept for one
release. Input columns (a node a source drives, no circuit unknown in it,
e.g. V(ctrl)) are not graded. The strict gate, its tolerances and freezes
are untouched: this is a second measure next to it.
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
DELTA = 1e-3                     # the OLD measure's snap window (exact_old only), in netlist steps
SELF_N = 5000                    # self-score: the smooth part's rows, strided to at most this many
ONSET = 0.05                     # the error's onset: first sample off by 5 % of tol
# rule (1): each engine's declared parasitic window, in netlist steps. The exact engine projects
# every mode with Re(λ) < -100 / dt_max at an event (dsed/scheduler.hpp `project_`; the harness
# passes dt, so dt_max = dt and the output rule's cut does not apply): a transient shorter than
# dt / 100. The fixed engine's L-stable step collapses one shorter than its step h. trbdf2 records
# no right limit (nothing to apply the rule to).
PARASITIC = {"dsed": 1e-2, "pwl": 1.0}
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
MEASURE = "three-part 2026-11"   # json tag: a --previous file without it holds the old definition
PARTS = ("smooth", "instants", "limits")
MEASURE_TEXT = [
    "## The measure", "",
    "`exact` (`benchmarks/tools/metrics.py`, `exact_parts`), per graded column, three parts, each a "
    "fraction of `span` (the truth's peak-to-peak, floored at 1e-3 of its peak):", "",
    "- **smooth**: sqrt(sum w_i e_i^2 / sum w_i) over the engine's STORED rows, e_i the distance to the "
    "truth at t_i (two one-sided candidates only when t_i is a truth event instant; either is right), w_i "
    "trapezoid time weights; left out: the initial instant, the right limit at t = 0+ (stored rows only), "
    "and the rows inside a matched event's displacement [min(t_engine, t_truth), max(...)] (the engine is on "
    "the other branch there: `instants` prices it; the row at the engine's instant is its left limit: "
    "`limits` grades it). Every other row is graded, the first one solved after an event included. No snap "
    "window, no timing charge.",
    "- **instants**: engine device events (changes of the switch record of `pulsim.with_event_limits(res)`: "
    "stored rows, the split's inner events, right limits; the instant is the row before the change) "
    "matched to the truth's by device and direction, in order, the nearest within "
    "2 netlist steps; cost sqrt(sum_e jump_e^2 |delta_e| / T), the true L2 of each displacement (jump from "
    "the truth's two limits; |delta| within 4 ulp counts 0); a missing truth event is charged as displaced to "
    "that device's next truth event, at most the window (beyond it, the rows carry the error). `max δ/h`: the largest |delta| in units of the "
    "netlist dt, the same unit for every engine (the fixed engine's step; the output grid of the "
    "variable-step and exact engines, which place events by root finding). Missing and spurious events are "
    "counted and named in the json; a spurious one is not charged a duration (no row is left out around it: "
    "its effect is in `smooth`, its values in `limits`); an event displaced by more than the window is a "
    "missing + spurious pair. `n/a`: the run exposes no switch record (every row is then graded).",
    "- **limits**: the largest distance of a one-sided value the engine records after t = 0 (its left "
    "value at each of its event instants, every right-limit row, matched or not) from the truth's limit "
    "on that side at the truth's instant (at the engine's own instant when there is no match). An "
    "engine without right limits (trbdf2) is graded on its left values only.", "",
    "Cell = the largest part of the worst column (`by` names it); a non-finite value anywhere scores inf. "
    "Classes (< 1e-3 / 1e-6 / 1e-9) are read from it. Events at the run's first or last instant are not "
    "graded. Source breaks without a device flip are not events here: the engine is given their instant, "
    "and the rows around them are graded in `smooth`; a non-finite truth value scores inf. Input columns (only a source in them) are not graded.", "",
    "**What changed and why (this release).** The previous `exact` (kept as `exact_old`, column `old`) was "
    "one time-weighted RMS over every row of the two-sided trace with a snap window of 1e-3 dt. Two flaws: "
    "(1) it dropped only rows at t == t[0], so the right limit at t = 0+ (5e-324) was graded against the "
    "truth's initial value: on `three_phase_diode_rectifier` pwl+restart that one row (V(out) 17.22 V, a "
    "quadrature line start, against 0 V) was 99 % of the score, 0.0127 against 0.000998 without it; the "
    "strict gate already drops it. (2) An event the engine places delta away from the truth's, beyond "
    "the snap window, was charged jump^2 x the row's trapezoid weight instead of jump^2 x |delta|: "
    "`vcswitch_resistor` pwl+split switches 0.005 h early and read 0.0316 (half a step, 100x the true "
    "cost in the square; 0 before the right-limit rows existed), against 0.0032 now; adding or "
    "removing a row at the event changed the charge by a factor unrelated to accuracy. Both flaws "
    "mispriced all engines whenever an event is displaced; the snap window and its charge are gone.",
]
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
    is a right limit (one ulp later), else not recorded; the 7th element is
    True when the engine reports a projection there (a ``reinit_records``
    entry within 4 ulp of the instant: rule (1)). Every other right
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
    proj_t = np.array(sorted(float(r.t) for r in (getattr(res, "reinit_records", None) or ())))

    def reported(te):
        return bool(proj_t.size) and bool(np.any(np.abs(proj_t - te) <= 4 * np.spacing(abs(te))))
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
    # truth events within 4 ulp of each other (a complementary pair one ulp apart) are one
    # instant for the limits: left of the first, right of the last (an engine flips them at once)
    ut = sorted({x[0] for x in tev})
    tl = _one_sided(sol, cols, ut)
    first, last = list(range(len(ut))), list(range(len(ut)))
    for k in range(1, len(ut)):
        if ut[k] - ut[k - 1] <= 4 * np.spacing(abs(ut[k])):
            first[k] = first[k - 1]
    for k in range(len(ut) - 2, -1, -1):
        if ut[k + 1] - ut[k] <= 4 * np.spacing(abs(ut[k + 1])):
            last[k] = last[k + 1]
    tl_at = {c: {u: (tl[c][0][first[k]], tl[c][1][last[k]]) for k, u in enumerate(ut)} for c in cols}
    inst = sorted({e[0] for e in eev} | {float(tv[r - 1]) for r in rr})
    el = _one_sided(sol, cols, inst)
    el_at = {c: dict(zip(inst, zip(*el[c]))) for c in cols}
    out = {}
    for c in cols:
        y = yv[c]
        out[c] = dict(
            truth_events=[(te, k, *tl_at[c][te]) for te, k in tev],
            engine_events=[(te, k, float(y[i]), None if r is None else float(y[r]), *el_at[c][te],
                            reported(te)) for te, k, i, r in eev],
            right_rows=[(float(tv[r - 1]), float(y[r]), el_at[c][float(tv[r - 1])][1]) for r in rr],
            events_available=bool(available or not tev))
    return out


def grade(sol, tr, cols, sp, dt, tol, self_score=True, names=None, parasitic=0.0):
    """The exact measure of one run: ``tr`` = ``differential_runner.traces(res, cols)``.

    ``exact`` = the cell's summary: per column ``metrics.exact_parts`` (smooth,
    instants, limits; the largest), the worst column; ``exact_part`` names the
    part that set it. ``exact_old`` = the previous definition (one release).
    ``parasitic``: the engine's declared parasitic window in seconds (rule (1),
    ``PARASITIC``; 0 disables it); ``n_projected`` counts the events it applied to."""
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
        p = metrics.exact_parts(t, y, cand[c], span=sp[c], h=dt, parasitic=parasitic, limits_tol=tol,
                                truth_after=lambda te, s, c=c: _one_sided(sol, [c], te + s)[c][1], **ev[c])
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
               n_projected=max(p.get("n_projected", 0) for p in per.values()),
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
            cell.update(grade(sol, dr.traces(res, cols), g, sp, case.dt, tols[eng],
                              parasitic=PARASITIC.get(eng, 0.0) * case.dt))
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
    OUT_JSON.write_text(json.dumps(dict(measure=MEASURE, event_window_steps=metrics.EVENT_WINDOW,
                                        old_delta_steps=DELTA, commit=head, configs=configs, circuits=recs,
                                        twins=twin), indent=1) + "\n", encoding="utf-8")
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
         *MEASURE_TEXT, "",
         f"No floor: the truth itself (mpmath `Solver.sample`) graded by the smooth part on each engine's "
         f"own rows scores at most {max(selfs):.1e} over {len(selfs)} grids (median {np.median(selfs):.1e}).", ""]
    L += findings(recs, cfgs)
    L += ["", "## Summary per configuration", "",
          "`graded` = cells that ran and were graded; `not graded` = refused / error (no `exact`). "
          "`old` = the same cells under the previous definition (`exact_old`). `set by` = how many cells' "
          "summary each part set (smooth / instants / limits).", "",
          "| configuration | graded | not graded | exact < 1e-3 | < 1e-6 | < 1e-9 | >= 1e-2 "
          "| old < 1e-3 / 1e-6 / 1e-9 | set by s / i / l | worst five |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for cfg in cfgs:
        xs = sorted(((c["exact"], r["circuit"], c) for r, k, c in cells if k == cfg and "exact" in c),
                    key=lambda x: -x[0])
        e = np.array([x for x, _, _ in xs])
        o = np.array([c["exact_old"] for _, _, c in xs])
        ng = sum(1 for _, k, c in cells if k == cfg and "exact" not in c)
        top = ", ".join(f"{Path(n).stem} {x:.2g} {c['exact_part'][0]}" for x, n, c in xs[:5])
        by = " / ".join(str(sum(c["exact_part"] == p for _, _, c in xs)) for p in PARTS)
        L.append(f"| {cfg} | {len(xs)} | {ng} | " + " | ".join(str(int((e < t).sum())) for t in THRESHOLDS)
                 + f" | {int((e >= 1e-2).sum())} | " + " / ".join(str(int((o < t).sum())) for t in THRESHOLDS)
                 + f" | {by} | {top} |")
    L += class_changes(cells, cfgs)
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
    L += ["", "## Full table", "",
          "`exact` = the summary, `by` = the part that set it (s smooth, i instants, l limits); `max δ/h` = the "
          "largest matched displacement in netlist steps (signed: engine − truth); `unmatched` = missing / "
          "spurious device events (`-` instants: no event record); `old` = `exact_old`; strict = the gate's "
          "verdict and interp error.", "",
          "| circuit | configuration | exact | by | smooth | instants | max δ/h | limits | unmatched | old | strict |",
          "|---|---|---:|---|---:|---:|---:|---:|---|---:|---|"]
    for r in recs:
        for k in cfgs:
            c = r["cells"].get(k)
            if c is None:
                continue
            name = r["circuit"].removeprefix("circuits/")
            if "exact" not in c:
                L.append(f"| {name} | {k} | {c['outcome']} |||||||||")
                continue
            proj = f" p{c['n_projected']}" if c.get("n_projected") else ""
            L.append(f"| {name} | {k} | {_g(c['exact'])} | {c['exact_part'][0]}{proj} | {_g(c['smooth'])} "
                     f"| {_g(c['instants']) if c['instants_available'] else 'n/a'} | {_g(c['max_dt_h'])} "
                     f"| {_g(c['limits'])} | {c['n_missing']} / {c['n_spurious']} | {_g(c['exact_old'])} "
                     f"| {c['outcome'][:3]} {_g(c.get('err_rms'))} |")
    L += ["", "Reference column (the harness reference graded the same way), per circuit: "
          + "; ".join(f"{Path(r['circuit']).stem} {_g(r.get('reference', {}).get('exact'))}" for r in recs) + "."]
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
    same_def = old.get("measure") == MEASURE        # else compare the previous definition with itself
    if not same_def:
        recs = [dict(r, cells={k: (dict(c, exact=c["exact_old"]) if "exact" in c else c)
                               for k, c in r["cells"].items()}) for r in recs]
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
         + ("" if same_def else "That file holds the previous definition: compared here with this run's "
            "`exact_old` (the same definition), so this section shows engine changes only; the change of "
            "definition is the section above. ") +
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


def _bucket(x):
    return sum(x >= t for t in THRESHOLDS)    # 0: < 1e-9 ... 3: >= 1e-3


def class_changes(cells, cfgs):
    """Every cell whose class (< 1e-9 / 1e-6 / 1e-3 / above) differs between ``exact_old`` and ``exact``."""
    rows = [(r, k, c) for r, k, c in cells if "exact" in c and _bucket(c["exact"]) != _bucket(c["exact_old"])]
    worse = [x for x in rows if x[2]["exact"] > x[2]["exact_old"]]

    def why(c):
        if c["exact"] < c["exact_old"]:
            return (f"the old score came from rows the smooth part leaves out (the 0+ row, rows next to an "
                    f"event charged at their weight); smooth {_g(c['smooth'])}")
        p = c["exact_part"]
        if p == "instants":
            return f"instants: max δ = {c['max_dt_h']:.3g} h"
        if p == "limits":
            lw = c.get("limits_worst") or {}
            return f"limits: worst one-sided value at t = {_g(lw.get('t'))} s ({lw.get('col')})"
        return f"{p}: {c['n_missing']} missing / {c['n_spurious']} spurious events" if p == "smooth" else p
    L = ["", "## Cells whose class changed with the definition", "",
         f"{len(rows)} cells change class between `exact_old` and `exact` ({len(worse)} worse, "
         f"{len(rows) - len(worse)} better). A worse cell is the measure now seeing a timing or one-sided "
         "error the old one did not price: a candidate finding, not an artefact (`## Notes`).", "",
         "| circuit | configuration | old | exact | by | why |", "|---|---|---:|---:|---|---|"]
    for r, k, c in sorted(rows, key=lambda x: (x[2]["exact"] < x[2]["exact_old"], x[0]["circuit"], cfgs.index(x[1]))):
        L.append(f"| {r['circuit'].removeprefix('circuits/')} | {k} | {_g(c['exact_old'])} | {_g(c['exact'])} "
                 f"| {c['exact_part']} | {why(c)} |")
    return L


def findings(recs, cfgs):
    L = ["## Findings for the engine tracks", "",
         "The five largest engine errors per configuration (exact > tol). `onset`: the first "
         f"smooth-part row whose own deviation exceeds {ONSET:g} tol (fraction of span), and the last truth "
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
        L += ["| circuit | exact | part | column | onset [s] | truth event at onset | departs [s] | truth event there |",
              "|---|---:|---|---|---:|---|---:|---|"]

        def ev(e):
            return f"{_g(e.get('t'))} {e.get('cause', '')}: {e.get('flips', '')}" if e else "start"
        for x, r, c in xs[:5]:
            d = c.get("departure") or {}
            L.append(f"| {r['circuit'].removeprefix('circuits/')} | {x:.3g} | {c['exact_part']} | {c['exact_col']} | "
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
