"""Task 13.3 (report only): would `engine='auto'` route each v2 netlist to the exact
engine? benchmarks/circuits/*.yaml are pulsim-v1 netlists the 2.0 loader refuses (counted, not
run), so the measured set is examples/*.yaml (the schematic templates are GUI templates,
not netlists). For each: the engine 'auto' picks today
(`result.engine_used`), and the exact engine's verdict `result.exact['auto_would_route']`
(every segment exact) with the first non-exact reason, from
`simulate(engine='exact', allow_implicit_segments=True)`. Each circuit runs in a child process
with a wall cap (a hung run is reported, never waited on). The default route does not change.

    PYTHONPATH=<worktree>/python python3 auto_route_survey.py [cap_seconds] > auto_route_survey.out
"""
import multiprocessing as mp
import os
import sys
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _one(path, q):
    warnings.simplefilter("ignore")
    import pulsim as p
    out = {"auto": "-", "would": "-", "why": ""}
    try:
        lc = p.load_yaml_file(path)
        try:
            r = p.simulate(lc)
            out["auto"] = getattr(r, "engine_used", "?")
        except Exception as e:  # noqa: BLE001
            out["auto"] = f"fails ({type(e).__name__})"
        lc = p.load_yaml_file(path)
        kw = {"engine": "exact", "allow_implicit_segments": True}
        r = p.simulate(lc, **kw)
        ex = getattr(r, "exact", None) or {}
        out["would"] = str(ex.get("auto_would_route", "-"))
        bad = [s for s in ex.get("segments", []) if not s["exact"]]
        if bad:
            s = bad[0]
            out["why"] = f"{s['reason']} {s['device']} {s['detail'][:60]}".strip()
    except p.UnsupportedError as e:
        out["would"] = "False"
        out["why"] = f"refused: {e.feature}" if e.feature else f"refused: device {e.device} ({e.kind})"
    except Exception as e:  # noqa: BLE001
        v1 = "pulsim v1 netlist" in str(e)
        out["would"] = "v1" if v1 else "error"
        out["why"] = "pulsim-v1 netlist, not loadable" if v1 else f"{type(e).__name__}: {str(e)[:110]}"
    q.put(out)


def main():
    cap = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    v1 = os.path.join(ROOT, "benchmarks", "circuits")
    n_v1 = sum(f.endswith(".yaml") for _, _, fs in os.walk(v1) for f in fs)
    print(f"benchmarks/circuits: {n_v1} pulsim-v1 YAMLs, not loadable by the 2.0 loader (not run)")
    base = ROOT
    ex = os.path.join(ROOT, "examples")
    paths = sorted(os.path.join(ex, f) for f in os.listdir(ex) if f.endswith(".yaml"))
    print(f"{len(paths)} YAMLs under examples/, cap {cap:.0f} s each")
    print(f"{'circuit':46s} {'auto today':14s} {'would exact':12s} first non-exact reason")
    counts = {}
    ctx = mp.get_context("spawn")
    for path in paths:
        q = ctx.Queue()
        pr = ctx.Process(target=_one, args=(path, q))
        pr.start()
        pr.join(cap)
        if pr.is_alive():
            pr.terminate()
            pr.join()
            out = {"auto": "?", "would": "timeout", "why": f"> {cap:.0f} s"}
        else:
            out = q.get() if not q.empty() else {"auto": "?", "would": "crash", "why": f"exit {pr.exitcode}"}
        rel = os.path.relpath(path, base)
        counts[out["would"]] = counts.get(out["would"], 0) + 1
        print(f"{rel:46s} {out['auto']:14s} {out['would']:12s} {out['why']}", flush=True)
    print("totals:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
