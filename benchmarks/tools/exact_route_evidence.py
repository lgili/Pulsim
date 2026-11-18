"""Evidence for the exact engine's default-route gate (add-exact-propagator-strategy D11, 13.3).

    python3 benchmarks/tools/exact_route_evidence.py [--out benchmarks/EXACT_ROUTE_EVIDENCE.md]
                                                     [--circuits GLOB,...] [--refs DIR]

For every reference circuit the v1 reader maps (benchmarks/manifests/v2_triage.yaml, status
'runs') it runs the exact engine (``engine='exact'``) and today's default (``engine='auto'``)
the way the reference harness runs them (``v1_reader.BenchCase.run``: netlist dt, DC start,
switch_fn, step_observer) and grades both against the harness's reference (trbdf2 rtol 1e-8,
else pwl dt/4) with its strict measure. For every shipped example (``examples/*.yaml``) it
runs ``pulsim.exact_route`` and both engines; the examples have no reference, so their rows
give the route, the wall times, the events and the largest exact-vs-auto difference.

Nothing here changes a route: ``engine='auto'`` is not flipped (task 19.1 belongs to the
number-changing release). Wall times are single runs on whatever machine generated the file.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "benchmarks" / "tools"), str(ROOT / "benchmarks")]
import differential_runner as dr  # noqa: E402
import v1_reader  # noqa: E402

import pulsim  # noqa: E402


def _ref(rel, case, cols, tol, ref_dir):
    f = None
    if ref_dir is not None:
        ref_dir.mkdir(parents=True, exist_ok=True)
        f = ref_dir / (rel.replace("/", "__") + ".pkl")
        if f.exists():
            return pickle.loads(f.read_bytes())
    out = dr.reference(case, cols, tol)
    if f is not None:
        f.write_bytes(pickle.dumps(out))
    return out


def _events(res):
    for name in ("event_times",):
        v = getattr(res, name, None)
        if v is not None:
            return int(len(v))
    n = getattr(res, "n_events", None)
    return int(n) if n is not None else None


def _cell(case, engine, cols, ref, tol):
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = case.run(engine)
    except dr.REFUSAL as e:
        return dict(outcome="refused", wall_s=time.perf_counter() - t0,
                    why=f"{getattr(e, 'feature', '')}: {str(e).splitlines()[0][:90]}")
    except Exception as e:  # noqa: BLE001 - every outcome is a row
        return dict(outcome="error", wall_s=time.perf_counter() - t0,
                    why=f"{type(e).__name__}: {str(e).splitlines()[0][:90]}")
    rec = dict(wall_s=time.perf_counter() - t0, events=_events(res),
               engine_used=getattr(res, "engine_used", engine))
    if ref is None:
        rec.update(outcome="unverified", why="no converged reference")
    else:
        dr.classify(rec, res, case, cols, ref, tol, None)
    ex = getattr(res, "exact", None) or {}
    if ex:
        rec["auto_would_route"] = ex.get("auto_would_route")
    return rec


def reference_rows(patterns, ref_dir):
    rows = []
    triage = dr.load_triage()
    for rel, entry in sorted(triage.items()):
        if entry["status"] != "runs" or not any(fnmatch.fnmatch(rel, p) for p in patterns):
            continue
        case = v1_reader.load(dr.ROOT / rel)
        _, sw, _ = case.build()
        blob = json.dumps(case._components)
        switched = sw is not None or any(k in blob for k in ('"diode"', '"pwm"', '"pulse"'))
        cols = dr.observables_of(case)
        tol = entry.get("tol", {}).get("dsed", dr.default_tol("dsed", switched))
        ref_name, ref = _ref(rel, case, cols, tol, ref_dir)
        if entry.get("suspect"):
            ref_name, ref = None, None
        row = dict(circuit=rel.replace("circuits/", ""), reference=ref_name, tol=tol,
                   exact=_cell(case, "exact", cols, ref, tol),
                   auto=_cell(case, "auto", cols, ref, tol))
        rows.append(row)
        print(f"{row['circuit']:60s} exact={row['exact']['outcome']:10s} "
              f"auto={row['auto']['outcome']}", flush=True)
    return rows


def example_rows():
    rows = []
    for f in sorted((ROOT / "examples").glob("*.yaml")):
        try:
            lc = pulsim.load_yaml_file(str(f))
        except Exception as e:  # noqa: BLE001 - a v1 file the 2.0 loader refuses
            rows.append(dict(example=f.stem, load=f"{type(e).__name__}"))
            continue
        row = dict(example=f.stem)
        t_end = getattr(lc, "t_end", 0.0) or None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rt = pulsim.exact_route(lc.builder, t_end) if t_end else None
            row["route"] = (rt["exact"], rt["reason"]) if rt else (None, "no t_end")
        except Exception as e:  # noqa: BLE001
            row["route"] = (False, f"{type(e).__name__}: {str(e).splitlines()[0][:60]}")
        res = {}
        for eng in ("exact", "auto"):
            t0 = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = pulsim.simulate(pulsim.load_yaml_file(str(f)), engine=eng)
                res[eng] = r
                row[eng] = dict(outcome="ran", wall_s=time.perf_counter() - t0,
                                events=_events(r), engine_used=getattr(r, "engine_used", eng))
            except Exception as e:  # noqa: BLE001
                row[eng] = dict(outcome="refused" if isinstance(e, dr.REFUSAL) else "error",
                                wall_s=time.perf_counter() - t0,
                                why=f"{type(e).__name__}: {str(e).splitlines()[0][:80]}")
        if len(res) == 2:
            a, b = res["exact"], res["auto"]
            ta, tb = np.asarray(a.times), np.asarray(b.times)
            xa, xb = np.asarray(a.states), np.asarray(b.states)
            n = min(xa.shape[1], xb.shape[1]) if xa.ndim == 2 and xb.ndim == 2 else 0
            if n and len(tb) > 1:
                # from the second sample: sample 0 differs by design (exact reports the
                # consistent algebraic state, the fixed-step engines their initial vector)
                d = max(float(np.max(np.abs(np.interp(tb[1:], ta, xa[:, j]) - xb[1:, j])))
                        / max(float(np.max(np.abs(xb[1:, j]))), 1e-12) for j in range(n))
                row["diff"] = d
        rows.append(row)
        print(f"{f.stem:40s} {row.get('route')}", flush=True)
    return rows


def _fmt(c):
    if c is None:
        return "", "", "", ""
    o = c["outcome"]
    err = f"{c['err_rms']:.2g}" if "err_rms" in c else c.get("why", "")
    ev = "" if c.get("events") is None else str(c["events"])
    return o, err, f"{c['wall_s']:.3g}", ev


def write(path, refs, exs):
    L = ["# Exact engine route evidence", "",
         "Generated by `benchmarks/tools/exact_route_evidence.py`; do not edit by hand.", "",
         "Exact = `engine='exact'`; auto = today's default `engine='auto'` (trbdf2 on these "
         "runs: the v1 harness passes no dt to auto). Error = the strict gate's err_rms "
         "against the harness reference (agree at <= tol). Wall times are one run each on a "
         "shared machine: an indication, not a benchmark. Nothing here changes a route.", ""]
    cnt = {}
    for r in refs:
        for k in ("exact", "auto"):
            cnt.setdefault(k, {}).setdefault(r[k]["outcome"], 0)
            cnt[k][r[k]["outcome"]] += 1
    L += ["## Totals (reference circuits)", "", "| engine | " + " | ".join(
        sorted({o for v in cnt.values() for o in v})) + " |"]
    outs = sorted({o for v in cnt.values() for o in v})
    L += ["|---|" + "---|" * len(outs)]
    for k in ("exact", "auto"):
        L += [f"| {k} | " + " | ".join(str(cnt[k].get(o, 0)) for o in outs) + " |"]
    both = [r for r in refs if r["exact"]["outcome"] == "agree" and "wall_s" in r["auto"]
            and r["auto"]["outcome"] in ("agree", "disagree")]
    if both:
        ratio = np.median([r["exact"]["wall_s"] / max(r["auto"]["wall_s"], 1e-9) for r in both])
        L += ["", f"Median wall-time ratio exact / auto where both ran and exact agrees: "
              f"{ratio:.3g} over {len(both)} circuits."]
    L += ["", "## Reference circuits", "",
          "| circuit | tol | exact | exact err | exact s | exact events | auto would route | "
          "auto | auto err | auto s | auto events |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in refs:
        e, a = _fmt(r["exact"]), _fmt(r["auto"])
        L += [f"| {r['circuit']} | {r['tol']:.0e} | {e[0]} | {e[1]} | {e[2]} | {e[3]} | "
              f"{r['exact'].get('auto_would_route', '')} | {a[0]} | {a[1]} | {a[2]} | {a[3]} |"]
    L += ["", "## Shipped examples (no reference)", "",
          "Run as the route survey runs them: `simulate(load_yaml_file(f), engine=...)`, no "
          "switch_fn (every switch closed). The difference is the largest over the state "
          "columns, relative to the column's peak: a node on a floating island (a flyback "
          "secondary, a bridge's DC bus) has no defined potential, so a large number there "
          "is a gauge difference until checked on a differential quantity.", "",
          "| example | exact_route | exact | exact s | exact events | auto | auto s | "
          "max rel diff exact vs auto (from sample 1) |", "|---|---|---|---|---|---|---|---|"]
    for r in exs:
        if "load" in r:
            L += [f"| {r['example']} | not loadable ({r['load']}) | | | | | | |"]
            continue
        rt = r.get("route", (None, ""))
        e, a = r.get("exact", {}), r.get("auto", {})
        L += [f"| {r['example']} | {'exact' if rt[0] else rt[1]} | "
              f"{e.get('outcome', '')} {e.get('why', '')} | {e.get('wall_s', 0):.3g} | "
              f"{e.get('events', '')} | {a.get('outcome', '')} {a.get('why', '')} | "
              f"{a.get('wall_s', 0):.3g} | {r.get('diff', ''):{'.2g' if 'diff' in r else ''}} |"]
    Path(path).write_text("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "EXACT_ROUTE_EVIDENCE.md"))
    ap.add_argument("--circuits", default="*")
    ap.add_argument("--refs", default=None, help="cache directory for the references")
    ap.add_argument("--no-examples", action="store_true")
    a = ap.parse_args(argv)
    refs = reference_rows(a.circuits.split(","), Path(a.refs) if a.refs else None)
    exs = [] if a.no_examples else example_rows()
    write(a.out, refs, exs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
