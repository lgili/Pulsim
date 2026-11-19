#!/usr/bin/env python3
"""Rehearsal of the number-changing release: every reference circuit under
today's defaults, each next-default option alone, all of them together, and
``engine='auto'`` under the profile (``next_defaults.py`` holds the profile).

    next_defaults_rehearsal.py run [--circuits GLOB,...] [--configs NAME,...]
                                   [--jobs 2] [--cell-timeout 300] [--ref-timeout 1200]
    next_defaults_rehearsal.py render        # benchmarks/NEXT_DEFAULTS_REHEARSAL.md

Configurations: (a) the five of ``reference_status.CONFIGS``, unchanged;
(b) ``<engine>:<option>`` = one profile option on top of today's defaults
(``pwl:restart`` is the (a) cell ``pwl+restart``, the same run; ``pwl:split`` is
``pwl+split``: the split alone is refused, it requires the restart rule); (c)
``<engine>+next`` = the whole profile for that engine; (d) ``auto`` (today) and
``auto+next`` (the release route: ``pulsim.exact_route`` says every segment is
exact -> the exact engine, otherwise trbdf2; both without ``dt``, as a script
that passes none; with a ``dt`` the route is pwl and the cell is ``pwl+next``);
extra: ``trbdf2+esdirk43`` and ``trbdf2+next+esdirk43`` (opt-in, not a default).

Each cell is graded exactly as ``reference_status.py`` grades it (one reference
per circuit computed with today's defaults, the strict gate, the harness's own
tolerances, ``reference: suspect`` -> unverified) and, where the independent
mpmath truth solves the circuit (``exact_scoreboard.solved_circuits``), with the
floor-free ``exact`` measure (``exact_scoreboard.grade``, no self-score). The
analytic set is graded against its closed forms as ``reference_status`` does.

Resumable: one JSON file per circuit under ``.scratch/rehearsal/cells`` keyed by
configuration; a run measures only the missing cells. Bounded: each circuit
runs in a child process that a supervisor kills when one cell passes
``--cell-timeout`` (the cell is recorded ``timeout``); at most ``--jobs``
children; run the whole command under ``nice -n 15``.
"""

from __future__ import annotations

import argparse
import collections
import fnmatch
import json
import os
import queue
import subprocess
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import differential_runner as dr  # noqa: E402
import next_defaults as nd  # noqa: E402
import reference_status as rs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / ".scratch" / "rehearsal" / "cells"
PYTEST_SUMMARY = ROOT.parent / ".scratch" / "rehearsal" / "pytest_profile.json"
OUT = ROOT / "NEXT_DEFAULTS_REHEARSAL.md"
RANK = {"agree": 0, "unverified": 0, "disagree": 1, "refused": 2, "error": 2, "timeout": 2}


def expected_a(section: str) -> dict:
    """The strict counts of REFERENCE_STATUS.md's first table under ``section``."""
    text = rs.OUT.read_text(encoding="utf-8").split(section, 1)[1]
    out = {}
    for ln in text.splitlines():
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if cells and cells[0] in rs.CONFIGS and len(cells) == 6:
            out[cells[0]] = tuple(int(c) for c in cells[1:])
        elif out and not ln.startswith("|"):
            break
    return out
ENGINES = ("pwl", "trbdf2", "dsed")


def configs() -> dict:
    """name -> dict(engine, kw, form, group, option, route, profile)."""
    C = {k: dict(engine=e, kw=dict(kw), form=False, group="a") for k, (e, kw) in rs.CONFIGS.items()}
    for eng in ENGINES:
        for k, e in {**nd.PROFILE[eng], **nd.BUILDER}.items():
            C[f"{eng}:{e.short}"] = dict(engine=eng, kw={} if k == "form" else {k: e.value},
                                         form=k == "form", group="b", option=k)
        C[f"{eng}+next"] = dict(engine=eng, kw=nd.profile_kwargs(eng), form=True, group="c")
    _, kw, _ = nd.EXTRA["esdirk43"]
    C["trbdf2+esdirk43"] = dict(engine="trbdf2", kw=dict(kw), form=False, group="extra")
    C["trbdf2+next+esdirk43"] = dict(engine="trbdf2", kw={**nd.profile_kwargs("trbdf2"), **kw},
                                     form=True, group="extra")
    C["auto"] = dict(engine="auto", kw={}, form=False, group="d")
    # the release route alone (today's options) isolates the route from the options
    C["auto:route"] = dict(engine="auto", kw={}, form=False, group="b", option="route", route=True)
    C["auto+next"] = dict(engine="auto", kw={}, form=True, group="d", route=True, profile=True)
    return C


CONFIGS = configs()
# the same run as an (a) cell; the split alone is refused by run_transient (it needs the rule)
ALIAS = {"pwl:restart": "pwl+restart", "pwl:split": "pwl+split"}


def units(patterns) -> list:
    tri = dr.load_triage()
    out = [rel for rel, e in sorted(tri.items())
           if e["status"] == "runs" and any(fnmatch.fnmatch(rel, p) for p in patterns)]
    from analytic import CASES
    out += [f"analytic:{c.id}" for c in CASES
            if any(fnmatch.fnmatch(f"analytic:{c.id}", p) for p in patterns)]
    return out


def _cache_file(unit):
    return CACHE / (unit.replace("/", "__").replace(":", "__") + ".json")


def load_unit(unit) -> dict:
    f = _cache_file(unit)
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"unit": unit, "cells": {}}


def save_unit(d):
    f = _cache_file(d["unit"])
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=0, default=float), encoding="utf-8")
    os.replace(tmp, f)


# ---------------------------------------------------------------- one cell ---
def _no_dt_run(case, engine, **kw):
    """BenchCase.run without the netlist dt (what a script without dt gets)."""
    import pulsim
    b, sw, obs = case.build()
    args = dict(engine=engine, t_start=case.t_start)
    if sw is not None:
        args["switch_fn"] = sw
    if obs is not None:
        args["step_observer"] = obs
    x0 = None if case.uic else case._dc_start(b, sw)
    if x0 is not None:
        args["initial_state"] = x0
    return pulsim.simulate(b, case.t_end, **{**args, **kw})


def route_next(build, t_end, profile=True, **kw):
    """(engine, reason): the release route of engine='auto' without dt."""
    import pulsim
    b, sw = build()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = pulsim.exact_route(b, t_end, switch_fn=sw, **kw,
                                   **(nd.profile_kwargs("dsed") if profile else {}))
    except dr.REFUSAL as e:
        return "trbdf2", f"exact_route refused: {getattr(e, 'feature', '')}: {str(e).splitlines()[0][:120]}"
    return ("dsed", None) if r["exact"] else ("trbdf2", f"segment not exact: {r['reason']}")


def _diag(res):
    d = getattr(res, "diagnostics", None) or {}
    m = d.get("mode_selection_auto") if isinstance(d, dict) else None
    return dict(engine_used=getattr(res, "engine_used", None),
                mode_chosen=(m or {}).get("chosen"), n_samples=int(len(res.times)))


def run_cell(case, name):
    """(kind, res, wall, msg, info)."""
    spec = CONFIGS[name]
    info, t0 = {}, time.perf_counter()
    try:
        with warnings.catch_warnings(), nd.builder_form(spec["form"]):
            warnings.simplefilter("ignore")
            if spec.get("route"):
                def build():
                    b, sw, _ = case.build()
                    return b, sw
                _, _, obs = case.build()
                x0 = None
                if not case.uic:
                    b0, sw0, _ = case.build()
                    x0 = case._dc_start(b0, sw0)
                kw = dict(t_start=case.t_start, **({"step_observer": obs} if obs else {}),
                          **({"initial_state": x0} if x0 is not None else {}))
                prof = spec.get("profile", False)
                eng, why = route_next(build, case.t_end, prof, **kw)
                info.update(route=eng, route_reason=why)
                res = _no_dt_run(case, eng, **(nd.profile_kwargs(eng) if prof else {}))
            elif spec["engine"] == "auto":
                res = case.run("auto")
            else:
                with rs.extra_kwargs(spec["kw"]):
                    res = case.run(spec["engine"])
        info.update(_diag(res))
        return "ran", res, time.perf_counter() - t0, None, info
    except dr.REFUSAL as ex:
        return "refused", None, time.perf_counter() - t0, f"{type(ex).__name__}: {ex}"[:300], info
    except Exception as ex:  # noqa: BLE001 - classified, never swallowed
        return "error", None, time.perf_counter() - t0, f"{type(ex).__name__}: {ex}"[:300], info


def circuit_worker(unit, names, emit):
    import exact_scoreboard as sb
    import v1_reader
    entry = dr.load_triage()[unit]
    case = v1_reader.load(ROOT / unit)
    _, sw, _ = case.build()
    switched = sw is not None or any(k in json.dumps(case._components) for k in ('"diode"', '"pwm"', '"pulse"'))
    cols = dr.observables_of(case)
    tols = {e: entry.get("tol", {}).get(e, dr.default_tol(e, switched)) for e in dr.ENGINES}
    emit("PHASE reference")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref_name, ref = dr.reference(case, cols, min(tols.values()))
    sol = None
    if unit in sb.solved_circuits():
        import exact_truth as et
        sol = et.Solver(et.Circuit(ROOT / unit), sb.truth_dps(unit)).run()
        g, _, _ = sb.graded_columns(sol, cols)
        sp = sb.spans(sol, g) if g else None
    d = load_unit(unit)
    d.update(reference=ref_name, truth=sol is not None)
    for name in names:
        emit(f"PHASE {name}")
        kind, res, wall, msg, info = run_cell(case, name)
        eng = info.get("engine_used") or CONFIGS[name]["engine"]
        eng = dr._alias(eng) if eng in dr.ENGINES or eng == "exact" else CONFIGS[name]["engine"]
        tol = tols.get(eng, min(tols.values()))
        rec = dict(config=name, tol=tol, message=msg, wall_s=round(wall, 3), **info)
        if kind != "ran":
            rec["outcome"] = kind
        elif ref is None:
            rec.update(outcome="error", message="no converged reference")
        else:
            dr.classify(rec, res, case, cols, ref, tol, case.dt if eng == "pwl" else None)
        dr.mark_suspect(rec, entry)
        if kind == "ran" and sol is not None and g:
            try:
                rec["exact"] = sb.grade(sol, dr.traces(res, cols), g, sp, case.dt, tol,
                                        self_score=False)["exact"]
            except Exception as ex:  # noqa: BLE001 - the measure failed, said so
                rec["exact_error"] = f"{type(ex).__name__}: {ex}"[:200]
        rec.pop("worst", None)
        d["cells"][name] = {k: v for k, v in rec.items() if v is not None}
        save_unit(d)
        emit(f"DONE {name}")


def analytic_worker(unit, names, emit):
    import metrics
    from analytic import CASES
    c = next(x for x in CASES if x.id == unit.split(":", 1)[1])
    d = load_unit(unit)
    for name in names:
        emit(f"PHASE {name}")
        spec, info, t0 = CONFIGS[name], {}, time.perf_counter()
        eng = spec["engine"]
        rec = dict(config=name)
        try:
            if eng == "auto" and not spec.get("route"):
                rec.update(outcome="n/a", message="the analytic cases name their engine")
                d["cells"][name] = rec
                save_unit(d)
                emit(f"DONE {name}")
                continue
            with warnings.catch_warnings(), nd.builder_form(spec["form"]):
                warnings.simplefilter("ignore")
                kw = spec["kw"]
                if spec.get("route"):
                    prof = spec.get("profile", False)
                    eng, why = route_next(lambda: c.build(1.0), c.t_end, prof)
                    info.update(route=eng, route_reason=why)
                    kw = nd.profile_kwargs(eng) if prof else {}
                with rs.extra_kwargs(kw):
                    r = c.run(eng)
            ceil = rs.ANALYTIC[c.id].get(eng, {}).get("ceiling", 1e-3)
            e = metrics.err_rms(np.asarray(r.times), c.observe(r), c.exact,
                                abs_scale=c.abs_scale, exclude_t0=c.exclude_t0)
            rec.update(err_rms=e, outcome="agree" if e <= ceil else "disagree", tol=ceil, **_diag(r))
        except dr.REFUSAL as ex:
            rec.update(outcome="refused", message=f"{type(ex).__name__}: {ex}"[:300])
        except Exception as ex:  # noqa: BLE001 - classified
            rec.update(outcome="error", message=f"{type(ex).__name__}: {ex}"[:300])
        rec.update(wall_s=round(time.perf_counter() - t0, 3), **info)
        d["cells"][name] = {k: v for k, v in rec.items() if v is not None}
        save_unit(d)
        emit(f"DONE {name}")


def worker_main(unit, names):
    warnings.simplefilter("ignore")

    def emit(s):
        print(s, flush=True)
    (analytic_worker if unit.startswith("analytic:") else circuit_worker)(unit, names, emit)
    return 0


# -------------------------------------------------------------- supervisor ---
def missing(unit, names):
    have = load_unit(unit)["cells"]
    out = [n for n in names if n not in have and n not in ALIAS]
    return out


def supervise(unit, names, cell_timeout, ref_timeout, log):
    """Run the unit's missing cells in a child; a cell past its limit is `timeout`."""
    while True:
        todo = missing(unit, names)
        if not todo:
            return
        cmd = [sys.executable, str(Path(__file__).resolve()), "_worker", unit, ",".join(todo)]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True,
                             env={**os.environ, "PYTHONWARNINGS": "ignore"})
        q: queue.Queue = queue.Queue()
        threading.Thread(target=lambda: [q.put(x) for x in p.stdout] + [q.put(None)],
                         daemon=True).start()
        phase, killed = None, False
        while True:
            limit = ref_timeout if phase in (None, "reference") else cell_timeout
            try:
                line = q.get(timeout=limit)
            except queue.Empty:
                p.kill()
                p.wait()
                killed = True
                break
            if line is None:
                break
            if line.startswith("PHASE "):
                phase = line.split(None, 1)[1].strip()
        p.wait()
        d = load_unit(unit)
        if killed or p.returncode != 0:
            why = (f"timeout after {limit:g} s" if killed else f"worker exit {p.returncode}")
            if phase in (None, "reference"):
                for n in todo:            # nothing graded without a reference
                    d["cells"][n] = dict(config=n, outcome="timeout" if killed else "error",
                                         message=f"reference phase: {why}")
            else:
                d["cells"].setdefault(phase, dict(config=phase, outcome="timeout" if killed else "error",
                                                  message=why, wall_s=limit if killed else None))
            save_unit(d)
        print(f"{unit}: {len(d['cells'])} cells", flush=True)


def cmd_run(args):
    pats = [p if p.startswith(("circuits/", "analytic:")) else f"circuits/{p}"
            for p in args.circuits.split(",")]
    names = list(CONFIGS) if args.configs == "all" else args.configs.split(",")
    us = [u for u in units(pats) if missing(u, names)]
    print(f"{len(us)} units with missing cells", flush=True)
    log_path = CACHE.parent / "worker_stderr.log"
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log, ThreadPoolExecutor(args.jobs) as ex:
        list(ex.map(lambda u: supervise(u, names, args.cell_timeout, args.ref_timeout, log), us))
    return 0


# ------------------------------------------------------------------ report ---
KINDS = ("agree", "disagree", "error", "refused", "unverified", "timeout")


def collect():
    diff, ana = {}, {}
    for f in sorted(CACHE.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        cells = dict(d["cells"])
        for a, b in ALIAS.items():
            if b in cells:
                cells[a] = dict(cells[b], config=a)
        (ana if d["unit"].startswith("analytic:") else diff)[d["unit"]] = dict(d, cells=cells)
    return diff, ana


def _counts(units_, name):
    cells = [u["cells"][name] for u in units_.values() if name in u["cells"]]
    return [sum(c["outcome"] == k for c in cells) for k in KINDS], len(cells)


def _err(c):
    if c is None:
        return "not run"
    if c["outcome"] in ("refused", "error", "timeout", "n/a"):
        return c["outcome"]
    s = f"{c['err_rms']:.3g}" if "err_rms" in c else c["outcome"]
    return s + (f" / ex {c['exact']:.2g}" if "exact" in c else "")


def compare(a, b):
    """'worse' / 'better' / None for cell b against cell a (same circuit)."""
    if a is None or b is None or "n/a" in (a["outcome"], b["outcome"]):
        return None
    ra, rb = RANK[a["outcome"]], RANK[b["outcome"]]
    if rb > ra:
        return "worse"
    if rb < ra:
        return "better"
    ea, eb = a.get("exact"), b.get("exact")
    if ea is not None and eb is not None and max(ea, eb) > 1e-12:
        if eb > 2 * ea:
            return "worse"
        if ea > 2 * eb:
            return "better"
    return None


def _why(c):
    return (c.get("message") or "").replace("|", "/").replace("\n", " ")[:160]


def _repro(unit, eng, opt_names):
    form = "form" in opt_names
    if "enable_substep_state_correction" in opt_names:
        opt_names = set(opt_names) | {"discontinuity_restart"}
    kw = {k: v for k, v in nd.profile_kwargs(eng).items() if k in opt_names}
    return (f"`with nd.builder_form({form}): v1_reader.load('benchmarks/{unit}').run('{eng}'"
            + "".join(f", {k}={v!r}" for k, v in kw.items()) + ")`")


def changes(units_, eng):
    """Rows (unit, verdict, base, full, responsible singles) for engine eng
    ('auto': auto+next against auto, responsible singles = the route alone)."""
    base_n, full = (eng, f"{eng}+next")
    singles = [n for n, s in CONFIGS.items() if s["group"] == "b" and s["engine"] == eng]
    out = []
    for unit, u in sorted(units_.items()):
        base, cf = u["cells"].get(base_n), u["cells"].get(full)
        v = compare(base, cf)
        if v:
            resp = [n for n in singles if compare(base, u["cells"].get(n)) == v]
            out.append((unit, v, base, cf, resp))
    return out


WORSE_CLASSES = ("fails", "samples wrong (truth)", "disagree, no truth", "less accurate, still agrees",
                 "strict gate only (samples within tol of the truth)")


def worse_class(b):
    """Why a worse cell is worse, ranked (index into WORSE_CLASSES)."""
    if b["outcome"] in ("refused", "error", "timeout"):
        return 0
    ex = b.get("exact")
    if b["outcome"] == "disagree":
        if ex is None:
            return 2
        return 1 if ex > b.get("tol", 1e-3) else 4
    return 3


def render():
    diff, ana = collect()
    L = ["## Method", "",
         "Generated by `benchmarks/tools/next_defaults_rehearsal.py render` (do not edit above `## Notes`) from the",
         "cell cache of `run`; the profile is `benchmarks/tools/next_defaults.py` (`--list`).",
         "Nothing here changes a library default: every option is applied by the harness.", ""]
    L += ["## Profile", "", "| engine | option | value | change | decision |", "|---|---|---|---|---|"]
    L += [f"| {e} | `{k}` | `{x.value!r}` | `{x.change}` | {x.decision} |" for e, k, x in nd.table()]
    L += ["", "Not applied (not expressible through the harness on this tree):", ""]
    L += [f"- {e}: {w} (`{c}`): {y}" for e, w, c, y in nd.NOT_APPLIED]
    L += ["", f"## Counts, differential matrix ({len(diff)} circuits)", "",
          "Graded as `REFERENCE_STATUS.md` (strict gate, harness tolerances, suspect -> unverified).",
          "`timeout`: the cell passed the per-cell limit and was killed.", "",
          "| configuration | group | " + " | ".join(KINDS) + " | cells | wall (s) |",
          "|---|---|" + "---:|" * (len(KINDS) + 2)]
    for n, s in CONFIGS.items():
        cnt, tot = _counts(diff, n)
        wall = sum(u["cells"][n].get("wall_s") or 0 for u in diff.values() if n in u["cells"])
        L.append(f"| {n} | {s['group']} | " + " | ".join(map(str, cnt)) + f" | {tot} | {wall:.1f} |")
    exp = expected_a("## Differential matrix")
    ok = len(exp) == 5 and all(tuple(_counts(diff, n)[0][:5]) == v and _counts(diff, n)[0][5] == 0
                               for n, v in exp.items())
    L += ["", f"Configuration (a) reproduces REFERENCE_STATUS.md's strict counts: **{'yes' if ok else 'NO'}**"
          f" (read from that file: {exp}).", ""]
    if ana:
        L += [f"## Counts, analytic set ({len(ana)} cases; closed-form truth, frozen ceilings)", "",
              "| configuration | " + " | ".join(KINDS) + " |", "|---|" + "---:|" * len(KINDS)]
        for n in CONFIGS:
            cnt, tot = _counts(ana, n)
            if sum(cnt):
                L.append(f"| {n} | " + " | ".join(map(str, cnt)) + " |")
        exp = expected_a("## Analytic set")
        ok = len(exp) == 5 and all(tuple(_counts(ana, n)[0][:5]) == v for n, v in exp.items())
        L += ["", "`auto` (today) has no analytic rows: the cases name their engine; the route rows route them.",
              f"Configuration (a) reproduces REFERENCE_STATUS.md's analytic counts: **{'yes' if ok else 'NO'}**.", ""]
    allu = {**diff, **ana}
    L += ["## Each configuration against today's default of its engine", "",
          "Same rule as below (outcome rank, else `exact` beyond 2x); `auto+next` against `auto`,",
          "the esdirk columns against `trbdf2`. Circuits and analytic cases together.", "",
          "`moved` = cells whose err_rms is not bit-equal to the base's (the option reached the",
          "run); `mode` = the `mode_selection='auto'` choices recorded by the runs (lcp / legacy).", "",
          "| configuration | base | better | worse | compared | moved | mode |", "|---|---|---:|---:|---:|---:|---|"]
    for n, s in CONFIGS.items():
        if s["group"] == "a":
            continue
        base = "auto" if s["engine"] == "auto" else s["engine"]
        v = [compare(u["cells"].get(base), u["cells"].get(n)) for u in allu.values()]
        nc = sum(base in u["cells"] and n in u["cells"] and "n/a" not in
                 (u["cells"][base]["outcome"], u["cells"][n]["outcome"]) for u in allu.values())
        pairs = [(u["cells"][base], u["cells"][n]) for u in allu.values()
                 if base in u["cells"] and n in u["cells"]]
        moved = sum("err_rms" in x and "err_rms" in y and x["err_rms"] != y["err_rms"] for x, y in pairs)
        modes = collections.Counter(y.get("mode_chosen") for _, y in pairs if y.get("mode_chosen"))
        L.append(f"| {n} | {base} | {v.count('better')} | {v.count('worse')} | {nc} | {moved} | "
                 + ", ".join(f"{k} {m}" for k, m in sorted(modes.items())) + " |")
    L.append("")
    rows = {e: changes(allu, e) for e in (*ENGINES, "auto")}
    L += ["## Worse under the full profile than under today's default for that engine", "",
          "Worse = outcome rank up (agree/unverified < disagree < refused/error/timeout), or the",
          "`exact` error (truth) more than 2x larger. Responsible = the single options (b) that",
          "alone make the same cell worse; `combination` when none does alone.", "",
          "| circuit | engine | today | full profile | responsible | class; message |", "|---|---|---|---|---|---|"]
    for e in rows:
        for unit, v, a, b, resp in rows[e]:
            if v == "worse":
                L.append(f"| {unit.removeprefix('circuits/')} | {e} | {a['outcome']} {_err(a)} | "
                         f"{b['outcome']} {_err(b)} | {', '.join(resp) or 'combination'} | "
                         f"{WORSE_CLASSES[worse_class(b)]}; {_why(b)} |")
    L += ["", "## Better under the full profile", "",
          "Better = outcome rank down against the harness reference, or the `exact` error (truth)",
          "more than 2x smaller.", "",
          "| circuit | engine | today | full profile | responsible |", "|---|---|---|---|---|"]
    for e in rows:
        for unit, v, a, b, resp in rows[e]:
            if v == "better":
                L.append(f"| {unit.removeprefix('circuits/')} | {e} | {a['outcome']} {_err(a)} | "
                         f"{b['outcome']} {_err(b)} | {', '.join(resp) or 'combination'} |")
    L += ["", "## engine='auto' under the profile", "",
          "`auto` = today (no dt: trbdf2, or a refusal naming the blocker); `auto+next` = the",
          "release route (no dt). The exact engine without dt samples at its 10 us default ceiling,",
          "so its strict cell can disagree where `dsed+next` (netlist dt as the output grid) agrees.", "",
          "| circuit | today: engine / outcome | next: engine / outcome | route reason |", "|---|---|---|---|"]
    moved = 0
    for unit, u in sorted(diff.items()):
        a, b = u["cells"].get("auto"), u["cells"].get("auto+next")
        if a is None or b is None:
            continue
        ea, eb = a.get("engine_used") or "-", b.get("route") or b.get("engine_used") or "-"
        moved += ea != eb
        L.append(f"| {unit.removeprefix('circuits/')} | {ea} / {a['outcome']} {_err(a)} | "
                 f"{eb} / {b['outcome']} {_err(b)} | {(b.get('route_reason') or '').replace('|', '/')[:100]} |")
    L += ["", f"Circuits whose engine changes under `auto`: **{moved}**.", ""]
    L += ["## Every cell (err_rms / exact, or outcome)", "",
          "| circuit | " + " | ".join(CONFIGS) + " |", "|---|" + "---:|" * len(CONFIGS)]
    for unit, u in sorted(allu.items()):
        L.append(f"| {unit.removeprefix('circuits/')} | "
                 + " | ".join(_err(u["cells"].get(n)) for n in CONFIGS) + " |")
    body = "\n".join(L) + "\n"
    head = blockers(allu, rows) + pytest_section()
    old = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
    notes = old.split("\n## Notes\n", 1)[1] if "\n## Notes\n" in old else "\n(none)\n"
    OUT.write_text(head + body + "\n## Notes\n" + notes, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


def blockers(allu, rows):
    L = ["# Next-defaults rehearsal", "",
         "Rehearsal of the number-changing release on `feat/core-integration` fe75ee01 (method and",
         "tables below the blocker list).", "",
         "## Ranked blocker list", "",
         "### 1. Cells worse under the profile than under today's default, by class (classes 1-3 are engine fixes before the release)", ""]
    worse = [(u, e, a, b, r) for e in rows for (u, v, a, b, r) in rows[e] if v == "worse"]
    worse.sort(key=lambda x: (worse_class(x[3]), x[1], x[0]))
    i, cls = 0, None
    gate_only = [x for x in worse if worse_class(x[3]) == 4]
    for u, e, a, b, resp in worse:
        if worse_class(b) != cls:
            cls = worse_class(b)
            L += ["", f"**{WORSE_CLASSES[cls]}**", ""]
        i += 1
        if cls == 4:            # one cause, one entry: the samples are right, the grid is not
            ex = max(x[3].get("exact", 0.0) for x in gate_only)
            L.append(f"{i}. {len(gate_only)} cells ({', '.join(sorted({x[1] for x in gate_only}))}): "
                     f"the samples are within the tolerance of the truth (largest `exact` {ex:.2g}), "
                     "the strict gate (linear interpolation between samples) is not. On `auto` this is "
                     "the exact engine's output grid without dt (10 us default ceiling). Circuits: "
                     + ", ".join(f"`{x[0].removeprefix('circuits/')}`" for x in gate_only)
                     + ". Repro (any of them): "
                     + (f"`next_defaults_rehearsal._no_dt_run(v1_reader.load('benchmarks/{gate_only[0][0]}'), 'dsed')`"
                        " then compare `res.times` spacing with the netlist dt."))
            break
        if e == "auto":
            route = "route alone" if resp else "the profile on the routed engine"
            repro = (f"`with nd.builder_form({not resp}): next_defaults_rehearsal._no_dt_run("
                     f"v1_reader.load('benchmarks/{u}'), '{b.get('route', 'dsed')}'"
                     + ("" if resp else f", **nd.profile_kwargs('{b.get('route', 'dsed')}')") + ")`")
            L.append(f"{i}. `{u.removeprefix('circuits/')}` on auto -> {b.get('route')}: {a['outcome']} "
                     f"({_err(a)}) -> {b['outcome']} ({_err(b)}); responsible: {route}. {_why(b)} "
                     f"Repro: {repro}")
            continue
        opts = {CONFIGS[n]["option"] for n in resp} or set(nd.profile_kwargs(e)) | {"form"}
        L.append(f"{i}. `{u.removeprefix('circuits/')}` on {e}: {a['outcome']} ({_err(a)}) -> "
                 f"{b['outcome']} ({_err(b)}); responsible: {', '.join(resp) or 'combination'}. "
                 f"{_why(b)} Repro: {_repro(u, e, opts)}")
    if not worse:
        L.append("(none)")
    better = [(u, e, a, b, r) for e in rows for (u, v, a, b, r) in rows[e] if v == "better"]
    L += ["", f"### 2. Accepted changes of number ({len(better)} cells better under the profile)", "",
          "Each is listed in 'Better under the full profile' below; the evidence is the harness",
          "reference (outcome) or the mpmath truth (`exact` > 2x smaller). The CHANGELOG migration",
          "note of the release cites these.", ""]
    L += ["### 3. Owner decisions (add-kernel-simulation-object DECISIONS 23, branch "
          "feat/core-p3-simulation-object; undriven switches first)", "",
          "1. No `switch_fn` (measured on fe75ee01): pwl all CLOSED + a warning (v(o) = 0.999 V), "
          "trbdf2 refuses (`UnsupportedError`, 'the two engines' undriven-switch defaults "
          "differ'), dsed all CLOSED + a warning, `auto` without dt refuses (ValueError). Under "
          "the release route `auto` would send it to dsed: CLOSED with a warning instead of a "
          "refusal. One policy (task 11.3 there). Repro: "
          "`b = pulsim.CircuitBuilder(); b.add_voltage_source('V', 'a', '0', 1.0); "
          "b.add_switch('S', 'a', 'o', 1e3, 1e-9); b.add_resistor('R', 'o', '0', 1.0)`, then "
          "`pulsim.simulate(b, 1e-5, 1e-7, engine=e)` for e in pwl / trbdf2 / dsed.",
          "2. `mmc_arms` / `on_cache` compose on the run's own cache (pwl only).",
          "3. `progress` / `live_stream` wired after (1) on pwl.",
          "4. `resume_from` continues time on pwl / trbdf2 only (dsed refuses it).",
          "5. `mode_selection='auto'`: resolved while routing on pwl / trbdf2, in the "
          "dispatcher on dsed; on trbdf2 it is the settle (no 'auto' rule there).",
          "6. `on_impulse='error'`: pwl refuses at the event, dsed after the run, trbdf2 never records.",
          "7. Option value checks in Python for pwl only.",
          "8. `sweep()`'s recipe read once while planning.",
          "", "Also open for the owner: one voltage-sanity check (task 6.2 there); the composite "
          "diode drop's release (its DECISIONS OQ6); the TR-BDF2 step grid (factor-reuse 5.6: "
          "accuracy sign-off).", ""]
    return "\n".join(L) + "\n"


def pytest_section():
    L = ["## Pytest blast radius under the profile", ""]
    if not PYTEST_SUMMARY.exists():
        return "\n".join(L + ["Not run.", ""]) + "\n"
    s = json.loads(PYTEST_SUMMARY.read_text(encoding="utf-8"))
    L += [s.get("header", ""), "", "| class | failures |", "|---|---:|"]
    L += [f"| {k} | {v} |" for k, v in s["classes"].items()]
    L += ["", "| test file | failures | classes |", "|---|---:|---|"]
    L += [f"| {f} | {n} | {c} |" for f, n, c in s["files"][:25]]
    return "\n".join(L) + "\n\n"


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["_worker"]:
        return worker_main(argv[1], argv[2].split(","))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--circuits", default="*,analytic:*")
    r.add_argument("--configs", default="all")
    r.add_argument("--jobs", type=int, default=2, choices=(1, 2))
    r.add_argument("--cell-timeout", type=float, default=300.0)
    r.add_argument("--ref-timeout", type=float, default=1200.0)
    sub.add_parser("render")
    args = ap.parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else render()


if __name__ == "__main__":
    raise SystemExit(main())
