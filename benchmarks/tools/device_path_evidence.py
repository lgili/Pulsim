"""Regenerate benchmarks/DEVICE_PATH_EVIDENCE.md: the release evidence for
flipping ``device_path`` from ``'legacy'`` (hand stamps) to ``'interface'``
(every Newton kind's law on the device interface; add-unified-device-
interface DECISIONS #23).

    device_path_evidence.py [--circuits GLOB,...] [--repeats N]

Per circuit of the reference set (the differential matrix of
reference_status.py and the analytic cases) and per engine (``pwl``,
``trbdf2``): the cell outcome on each path, the error against the cell's
reference on each path, the largest state difference between the two
paths (of each state's scale) and the wall-time ratio interface / legacy
(minimum of ``--repeats`` interleaved runs; a machine-load number, not an
assertion). A circuit holding a kind not yet ported refuses
``'interface'`` by name and is listed as such.
"""
from __future__ import annotations

import argparse
import contextlib
import fnmatch
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import differential_runner as dr  # noqa: E402
import metrics  # noqa: E402
import v1_reader  # noqa: E402
from analytic import CASES  # noqa: E402

import pulsim  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "DEVICE_PATH_EVIDENCE.md"
ENGINES = ("pwl", "trbdf2")
PATHS = ("legacy", "interface")


@contextlib.contextmanager
def path_kw(path):
    orig = pulsim.simulate
    pulsim.simulate = lambda *a, **k: orig(*a, **{**k, "device_path": path})
    try:
        yield
    finally:
        pulsim.simulate = orig


def _timed(fn, path, repeats):
    """(kind, result, best wall, message) of ``fn`` on ``path``."""
    best, out = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        try:
            with path_kw(path):
                res = fn()
            out = ("ran", res, None)
        except dr.REFUSAL as ex:
            out = ("refused", None, f"{type(ex).__name__}: {ex}"[:160])
        except Exception as ex:  # noqa: BLE001 - classified
            # set_path's refusal of an unported kind is a ValueError
            kind = "refused" if "device_path='interface'" in str(ex) else "error"
            out = (kind, None, f"{type(ex).__name__}: {ex}"[:160])
        best = min(best, time.perf_counter() - t0)
        if out[0] != "ran":
            break
    return out[0], out[1], best, out[2]


def _diff(a, b):
    xa, xb = np.asarray(a.states), np.asarray(b.states)
    ta, tb = np.asarray(a.times), np.asarray(b.times)
    scale = np.maximum(np.max(np.abs(xa), axis=0), 1e-9)
    if xa.shape != xb.shape or not np.array_equal(ta, tb):
        # different accepted grids (TR-BDF2: a landing time moved by a
        # rounding): the times both runs accepted
        _, ia, ib = np.intersect1d(ta, tb, return_indices=True)
        if len(ia) == 0:
            return float("nan")
        xa, xb = xa[ia], xb[ib]
    return float(np.max(np.abs(xa - xb) / scale))


def _pair(fn, repeats, err_of):
    """Interleaved legacy / interface runs of one cell."""
    rec = {}
    runs = {}
    for path in PATHS:
        kind, res, wall, msg = _timed(fn, path, repeats)
        runs[path] = res
        rec[path] = dict(outcome=kind, wall=wall, message=msg)
        if res is not None:
            rec[path]["err"], rec[path]["verdict"] = err_of(res)
    if runs["legacy"] is not None and runs["interface"] is not None:
        rec["diff"] = _diff(runs["legacy"], runs["interface"])
        rec["speed"] = rec["interface"]["wall"] / max(rec["legacy"]["wall"], 1e-12)
    return rec


def differential(patterns, repeats):
    out = []
    for rel, entry in sorted(dr.load_triage().items()):
        if entry["status"] != "runs" or not any(fnmatch.fnmatch(rel, p) for p in patterns):
            continue
        case = v1_reader.load(ROOT / rel)
        _, sw, _ = case.build()
        switched = sw is not None or any(k in json.dumps(case._components)
                                         for k in ('"diode"', '"pwm"', '"pulse"'))
        cols = dr.observables_of(case)
        tols = {e: entry.get("tol", {}).get(e, dr.default_tol(e, switched)) for e in ENGINES}
        _, ref = dr.reference(case, cols, min(tols.values()))
        for eng in ENGINES:
            def err_of(res, eng=eng):
                if ref is None:
                    return None, "no reference"
                r = dict(circuit=rel)
                dr.classify(r, res, case, cols, ref, tols[eng], case.dt if eng == "pwl" else None)
                return r.get("err_rms"), r["outcome"]
            rec = _pair(lambda eng=eng: case.run(eng), repeats, err_of)
            rec.update(circuit=rel, engine=eng)
            out.append(rec)
            print(_line(rec), flush=True)
    return out


def analytic(repeats):
    out = []
    for c in CASES:
        for eng in ENGINES:
            def err_of(res, c=c):
                e = metrics.err_rms(np.asarray(res.times), c.observe(res), c.exact,
                                    abs_scale=c.abs_scale, exclude_t0=c.exclude_t0)
                return e, "ran"
            rec = _pair(lambda c=c, eng=eng: c.run(eng), repeats, err_of)
            rec.update(circuit=f"analytic:{c.id}", engine=eng)
            out.append(rec)
            print(_line(rec), flush=True)
    return out


def law_circuits(repeats):
    """Circuits that exercise every ported Newton law (the reference set's
    switched circuits use PWL devices and barely reach them): the
    equivalence circuits of python/tests/test_device_path_equivalence.py
    (MOSFET L1, Coss junction + VDMOS, saturable inductor, Shockley, IGBT L1
    with and without its tail, Lauritzen) on both Newton engines."""
    sys.path.insert(0, str(ROOT.parent / "python" / "tests"))
    import test_device_path_equivalence as T  # noqa: PLC0415 - test circuits
    out = []
    runs = {"pwl": dict(t_end=200e-6, dt=20e-9, engine="pwl"),
            "trbdf2": dict(t_end=200e-6, engine="trbdf2", rtol=1e-6, atol=1e-9)}
    for name, make in sorted(T.CIRCUITS.items()):
        for eng, kw in runs.items():
            def run(make=make, kw=kw):
                b = make()
                return pulsim.simulate(b, **T._gate(b.graph.num_switches), **kw)
            rec = _pair(run, repeats, lambda res: (None, "ran"))
            rec.update(circuit=f"law:{name}", engine=eng)
            out.append(rec)
            print(_line(rec), flush=True)
    return out


def _g(x):
    return "—" if x is None else f"{x:.3g}"


def _cell(r):
    o = r["outcome"]
    if o != "ran":
        return o
    return r.get("verdict") or "ran"


def _line(rec):
    leg, itf = rec["legacy"], rec["interface"]
    return (f"{rec['circuit']} | {rec['engine']} | {_cell(leg)} | {_cell(itf)} | "
            f"{_g(leg.get('err'))} | {_g(itf.get('err'))} | {_g(rec.get('diff'))} | "
            f"{_g(rec.get('speed'))}")


def render(recs):
    both = [r for r in recs if "diff" in r]
    refused = [r for r in recs if r["interface"]["outcome"] == "refused"
               and r["legacy"]["outcome"] != "refused"]
    same = [r for r in both if _cell(r["legacy"]) == _cell(r["interface"])
            and _g(r["legacy"].get("err")) == _g(r["interface"].get("err"))]
    lines = [
        "# Device-path evidence (legacy vs interface)",
        "",
        "Generated by `benchmarks/tools/device_path_evidence.py`; do not edit.",
        "add-unified-device-interface DECISIONS #23: the hand stamps (`legacy`, the",
        "default) against every Newton kind's law on the device interface",
        "(`interface`). `diff`: the largest state difference between the two paths,",
        "of each state's scale (TR-BDF2: at the times both runs accepted). `speed`: interface / legacy wall time, minimum of",
        "interleaved runs, on a shared machine (load-dependent).",
        "",
        f"- cells run on both paths: {len(both)}; same outcome and error to 3 "
        f"significant digits: {len(same)}",
        f"- cells `interface` refuses by name (a kind not ported yet): {len(refused)}",
        "",
        "| circuit | engine | legacy | interface | err legacy | err interface | diff | speed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += ["| " + _line(r) + " |" for r in recs]
    if refused:
        lines += ["", "## Refused on `interface`", ""]
        lines += [f"- {r['circuit']} ({r['engine']}): {r['interface']['message']}"
                  for r in refused]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", default="*")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    recs = (law_circuits(a.repeats) + differential(a.circuits.split(","), a.repeats)
            + analytic(a.repeats))
    OUT.write_text(render(recs))
    if a.json:
        Path(a.json).write_text(json.dumps(recs, indent=1, default=str))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
