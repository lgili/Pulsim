"""Task 13.3 (report only): would `engine='auto'` route each netlist to the exact
engine? Two sets: the shipped examples/*.yaml (the 2.0 loader), and every
benchmarks/circuits/*.yaml the v1 reader (`benchmarks/tools/v1_reader.py`) maps, run as the
reference harness runs it (`BenchCase.run`: netlist dt, DC start, switch_fn). The schematic
templates are GUI templates, not netlists. For each: the engine 'auto' picks today
(`result.engine_used`), and the exact engine's verdict `result.exact['auto_would_route']`
(every segment exact) with the first non-exact reason, from
`simulate(engine='exact', allow_implicit_segments=True)`. Each circuit runs in a child process
with a wall cap (a hung run is reported, never waited on). The default route does not change.

    PYTHONPATH=<worktree>/python:<worktree>/benchmarks/tools python3 auto_route_survey.py [cap_seconds] > auto_route_survey.out
"""
import multiprocessing as mp
import os
import sys
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _runners(path, v1):
    """(run_auto, run_exact) for one netlist: the 2.0 loader, or the v1 reader's BenchCase."""
    import pulsim as p
    kw = {"allow_implicit_segments": True}
    if v1:
        import v1_reader
        case = v1_reader.load(path)
        return (lambda: case.run("auto")), (lambda: case.run("exact", **kw))
    return ((lambda: p.simulate(p.load_yaml_file(path))),
            (lambda: p.simulate(p.load_yaml_file(path), engine="exact", **kw)))


def _one(path, v1, q):
    warnings.simplefilter("ignore")
    import pulsim as p
    out = {"auto": "-", "would": "-", "why": ""}
    try:
        run_auto, run_exact = _runners(path, v1)
        try:
            r = run_auto()
            out["auto"] = getattr(r, "engine_used", "?")
        except Exception as e:  # noqa: BLE001
            out["auto"] = f"fails ({type(e).__name__})"
        r = run_exact()
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
        unmapped = type(e).__name__ == "Unmapped"
        old = "pulsim v1 netlist" in str(e)
        out["would"] = "unmapped" if unmapped else "v1" if old else "error"
        out["why"] = (f"v1 reader: {e}"[:110] if unmapped else "pulsim-v1 netlist, not loadable"
                      if old else f"{type(e).__name__}: {str(e)[:110]}")
    q.put(out)


def _survey(title, paths, v1, cap):
    print(f"\n{title}: {len(paths)} YAMLs, cap {cap:.0f} s each")
    print(f"{'circuit':58s} {'auto today':14s} {'would exact':12s} first non-exact reason")
    counts = {}
    ctx = mp.get_context("spawn")
    for path in paths:
        q = ctx.Queue()
        pr = ctx.Process(target=_one, args=(path, v1, q))
        pr.start()
        pr.join(cap)
        if pr.is_alive():
            pr.terminate()
            pr.join()
            out = {"auto": "?", "would": "timeout", "why": f"> {cap:.0f} s"}
        else:
            out = q.get() if not q.empty() else {"auto": "?", "would": "crash", "why": f"exit {pr.exitcode}"}
        rel = os.path.relpath(path, ROOT)
        counts[out["would"]] = counts.get(out["would"], 0) + 1
        print(f"{rel:58s} {out['auto']:14s} {out['would']:12s} {out['why']}", flush=True)
    print("totals:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))


def main():
    cap = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    ex = os.path.join(ROOT, "examples")
    _survey("examples/", sorted(os.path.join(ex, f) for f in os.listdir(ex) if f.endswith(".yaml")),
            False, cap)
    bc = os.path.join(ROOT, "benchmarks", "circuits")
    _survey("benchmarks/circuits/ (v1 reader)", sorted(
        os.path.join(d, f) for d, _, fs in os.walk(bc) for f in fs if f.endswith(".yaml")), True, cap)


if __name__ == "__main__":
    main()
