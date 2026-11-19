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
RANK = {"agree": 0, "unverified": 0, "disagree": 1, "refused": 2, "error": 3, "timeout": 4}


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


def signature(name) -> str:
    """What a cached cell was measured with: its configuration and the profile."""
    import hashlib
    blob = json.dumps([CONFIGS[name], [(e, k, repr(x.value)) for e, k, x in nd.table()]],
                      sort_keys=True, default=repr)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
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
                b1, sw1, obs = case.build()     # one build: a closed loop's sw and observer share a PI

                def build():
                    return b1, sw1
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
        d["cells"][name] = {k: v for k, v in rec.items() if v is not None} | {"sig": signature(name)}
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
                d["cells"][name] = rec | {"sig": signature(name)}
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
        d["cells"][name] = {k: v for k, v in rec.items() if v is not None} | {"sig": signature(name)}
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
    return [n for n in names if n not in ALIAS
            and (n not in have or have[n].get("sig") != signature(n))]


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
                                         message=f"reference phase: {why}", sig=signature(n))
            else:
                d["cells"].setdefault(phase, dict(config=phase, outcome="timeout" if killed else "error",
                                                  message=why, wall_s=limit if killed else None,
                                                  sig=signature(phase)))
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


STALE: list = []


def collect():
    """Cells by unit; a cell whose signature is not today's is left out (STALE).
    An analytic cell's err_rms is against its closed form: it is its `exact`."""
    diff, ana = {}, {}
    STALE.clear()
    for f in sorted(CACHE.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        cells = {}
        for n, c in d["cells"].items():
            if n not in CONFIGS or c.get("sig") != signature(n):
                STALE.append((d["unit"], n))
                continue
            if d["unit"].startswith("analytic:") and "err_rms" in c:
                c = dict(c, exact=c["err_rms"])
            cells[n] = c
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


def judge(a, b):
    """(verdict, basis) for cell b against cell a of the same circuit: verdict
    'worse' / 'better' / None; basis 'outcome' (the harness reference), 'truth'
    (`exact` beyond 2x; an improvement only when the strict error did not grow
    past 2x, so a sparser grid cannot count as better) or 'reference' (the strict
    err_rms beyond 2x of the base's, above 1 % of the tolerance)."""
    if a is None or b is None or "n/a" in (a["outcome"], b["outcome"]):
        return None, None
    ra, rb = RANK[a["outcome"]], RANK[b["outcome"]]
    if rb != ra:
        return ("worse" if rb > ra else "better"), "outcome"
    sa, sb = a.get("err_rms"), b.get("err_rms")
    big = sa is not None and sb is not None and max(sa, sb) > (b.get("tol") or 1e-3) / 100
    up, down = big and sb > 2 * sa, big and sa > 2 * sb
    ea, eb = a.get("exact"), b.get("exact")
    if ea is not None and eb is not None and max(ea, eb) > 1e-12:
        if eb > 2 * ea:
            return "worse", "truth"
        if ea > 2 * eb and not up:
            return "better", "truth"
    if up:
        return "worse", "reference"
    if down:
        return "better", "reference"
    return None, None


def compare(a, b):
    """'worse' / 'better' / None (judge without the basis)."""
    return judge(a, b)[0]


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
    """Rows (unit, verdict, base, full, responsible singles, basis) for engine eng
    ('auto': auto+next against auto, the single = the route alone)."""
    base_n, full = (eng, f"{eng}+next")
    singles = [n for n, s in CONFIGS.items() if s["group"] == "b" and s["engine"] == eng]
    out = []
    for unit, u in sorted(units_.items()):
        base, cf = u["cells"].get(base_n), u["cells"].get(full)
        v, basis = judge(base, cf)
        if v:
            resp = [n for n in singles if compare(base, u["cells"].get(n)) == v]
            out.append((unit, v, base, cf, resp, basis))
    return out


WORSE_CLASSES = ("fails (refused / error / timeout)", "samples wrong (truth)", "disagree, no truth",
                 "less accurate at the samples (truth, > 2x), still agrees",
                 "strict gate only (disagrees; RMS error of the samples against the truth within tol)",
                 "strict error > 2x the base's, still agrees")


def worse_class(b, basis="truth"):
    """Why a worse cell is worse, ranked (index into WORSE_CLASSES)."""
    if b["outcome"] in ("refused", "error", "timeout"):
        return 0
    ex = b.get("exact")
    if b["outcome"] == "disagree":
        if ex is None:
            return 2
        return 1 if ex > b.get("tol", 1e-3) else 4
    return 3 if basis == "truth" else 5


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
    L += ["", "## Counts, differential matrix", "", f"{len(diff)} circuits.",
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
        L += ["## Counts, analytic set", "", f"{len(ana)} cases; closed-form truth, frozen ceilings.", "",
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
          "Same rule as below (outcome rank, else `exact` or the strict err_rms beyond 2x); `auto+next` against `auto`,",
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
          "Worse = outcome rank up (agree/unverified < disagree < refused < error < timeout), or the",
          "`exact` error (truth; for the analytic set the closed-form err_rms) more than 2x larger,",
          "or the strict err_rms more than 2x the base's (above 1 % of the tolerance). Responsible =",
          "the single options (b) that alone make the same cell worse; `combination` when none does.", "",
          "| circuit | engine | today | full profile | responsible | class; message |", "|---|---|---|---|---|---|"]
    for e in rows:
        for unit, v, a, b, resp, basis in rows[e]:
            if v == "worse":
                L.append(f"| {unit.removeprefix('circuits/')} | {e} | {a['outcome']} {_err(a)} | "
                         f"{b['outcome']} {_err(b)} | {', '.join(resp) or 'combination'} | "
                         f"{WORSE_CLASSES[worse_class(b, basis)]}; {_why(b)} |")
    L += ["", "## Better under the full profile", "",
          "Better = outcome rank down (basis `outcome`, the harness reference), or the `exact` error",
          "more than 2x smaller with the strict err_rms not grown past 2x (basis `truth`), or the",
          "strict err_rms more than 2x smaller (basis `reference`, no truth or truth unchanged).", "",
          "| circuit | engine | today | full profile | responsible | basis |", "|---|---|---|---|---|---|"]
    for e in rows:
        for unit, v, a, b, resp, basis in rows[e]:
            if v == "better":
                L.append(f"| {unit.removeprefix('circuits/')} | {e} | {a['outcome']} {_err(a)} | "
                         f"{b['outcome']} {_err(b)} | {', '.join(resp) or 'combination'} | {basis} |")
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
    import subprocess
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=8", "HEAD"], capture_output=True,
                          text=True, encoding="utf-8", check=True).stdout.strip()
    L = ["# Next-defaults rehearsal", "",
         f"Rehearsal of the number-changing release, rendered on `{head}` from the cell cache of `run`",
         "(method and tables below the blocker list; the pytest blast radius states its own run).", ""]
    if not any("output_rtol" in d for d in nd.PROFILE.values()):
        L += ["The profile at this head has no `output_rtol` (the exact engine's output-grid rule is not",
              "integrated yet): `auto+next` runs the exact engine without dt at its default output",
              "ceiling, so its strict counts are expected to stay where the previous rehearsal put them",
              "(80 / 28 / 1); the measured counts are in the tables below.", ""]
    L += [
         "## Ranked blocker list", "",
         "### 1. Cells worse under the profile than under today's default, by class",
         "", "The first three classes (fails, samples wrong, disagree without a truth) are engine",
         "fixes before the release; the last three need a decision or a re-baseline with evidence.", ""]
    worse = [(u, e, a, b, r, worse_class(b, bs)) for e in rows for (u, v, a, b, r, bs) in rows[e]
             if v == "worse"]
    worse.sort(key=lambda x: (x[5], x[1], x[0]))
    i, cls = 0, None
    for u, e, a, b, resp, k in worse:
        if k != cls:
            cls = k
            L += ["", f"**{WORSE_CLASSES[cls]}**", ""]
            if cls in (4, 5):      # one cause per class here: one entry listing the circuits
                grp = [x for x in worse if x[5] == cls]
                i += 1
                exs = [x[3]["exact"] for x in grp if "exact" in x[3]]
                L.append(f"{i}. {len(grp)} cells ({', '.join(sorted({x[1] for x in grp}))}). "
                         + (f"Largest `exact` (RMS of the samples against the truth) {max(exs):.2g}. "
                            if exs else "") +
                         ("On `auto` the exact engine runs without dt: 10 us default output ceiling; "
                          "the strict gate interpolates linearly between those samples. "
                          if all(x[1] == "auto" for x in grp) else "") +
                         "Circuits (today -> profile): "
                         + "; ".join(f"`{x[0].removeprefix('circuits/')}` {x[2]['outcome']} {_err(x[2])} -> "
                                     f"{x[3]['outcome']} {_err(x[3])}" for x in grp)
                         + (f". Repro (any of them): `next_defaults_rehearsal._no_dt_run(v1_reader.load("
                            f"'benchmarks/{grp[0][0]}'), '{grp[0][3].get('route', 'dsed')}')`"
                            if grp[0][1] == "auto" else ""))
        if cls in (4, 5):
            continue
        i += 1
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
    L += suite_refusals()
    better = collections.Counter((e, bs) for e in rows for (u, v, a, b, r, bs) in rows[e] if v == "better")
    L += ["", "### 2. Accepted changes of number", "",
          f"{sum(better.values())} cells better under the profile. By engine and basis: " + ", ".join(f"{e} {bs} {n}" for (e, bs), n in sorted(better.items()))
          + ". Each is listed in 'Better under the full profile' below with its basis; the CHANGELOG",
          "migration note of the release cites them.", ""]
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


# Refusals the profile raises in python/tests, by root cause (message substring, owning option).
SUITE_ROOTS = (
    ("period map / integrals", "form='branch'",
     "the exact period map / integrals (periodic_steady_state, sensitivities, shooting) refuse a "
     "branch-form diode: the flip of `add_diode` takes them away. Engine fix (branch current in the "
     "period map) or keep those diodes conductance-form"),
    ("mmc_arms=...", "discontinuity_restart",
     "`mmc_arms` refuses the restart rule (full-dt Thevenin stamp at dt/2): the flip breaks every MMC "
     "arm run. Engine fix or the rule off under `mmc_arms`"),
    ("cannot take the parameter", "form='branch'",
     "a branch-form switch refuses a bound `g_on` (`bind_param`): scoped / parametrised circuits break"),
    ("cannot simulate", "form='branch'",
     "bdf1 (retired) refuses a branch-form switch / diode by name and names pwl / trbdf2 "
     "(update-number-changing-release 1.7; ac_sweep runs them since): at the flip bdf1 keeps the "
     "conductance form or the test moves off bdf1"),
    ("requires discontinuity_restart", "enable_substep_state_correction",
     "a caller that sets `discontinuity_restart=False` is refused once the split defaults on: the split's "
     "default must follow the rule's"),
)


def suite_refusals():
    if not PYTEST_SUMMARY.exists():
        return ["", "### 1b. Refusals in the test suite: pytest under the profile not run", ""]
    fails = json.loads(PYTEST_SUMMARY.read_text(encoding="utf-8"))["failures"]
    L = ["", "### 1b. Refusals the profile raises in `python/tests` (engine fixes or owner decisions)", ""]
    for i, (pat, opt, what) in enumerate(SUITE_ROOTS, 1):
        hit = [f["nodeid"] for f in fails if pat in f["msg"]]
        if hit:
            L.append(f"{i}. {len(hit)} test{'s' * (len(hit) > 1)}, option `{opt}`: {what}. Repro: `{hit[0]}` under "
                     "`-p next_defaults_pytest`.")
    return L


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
