"""Cost gate of mode_selection='lcp' on pwl and trbdf2 (--configs trbdf2) (DECISIONS #30 of add-branch-form-ideal-switch).

    PYTHONPATH=<worktree>/python:<worktree>/benchmarks python3 lcp_cost_gate.py [--reps 7]
        [--configs pwl,pwl+restart,pwl+split] [--save out.npz] [--compare base.npz] [--circuits GLOB]

Every reference circuit of the differential matrix with a PWL diode (the cells where 'lcp' applies),
default against 'lcp', REPS interleaved runs each; the time is the wall time of the simulate()
call alone (the netlist build and the DC start are outside it). Printed per circuit: the median
of each and their ratio; the gate is 'lcp' <= 1.2x the default. --save writes every 'lcp' run's
times / states / switch states; --compare checks a later build against such a file byte for byte.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4] / "benchmarks"
sys.path[:0] = [str(ROOT / "tools"), str(ROOT)]
import differential_runner as dr  # noqa: E402
import reference_status as rs  # noqa: E402
import v1_reader  # noqa: E402

import pulsim  # noqa: E402


def cells(pattern):
    for rel, entry in sorted(dr.load_triage().items()):
        if entry["status"] != "runs" or not fnmatch.fnmatch(rel, pattern):
            continue
        case = v1_reader.load(ROOT / rel)
        if any(c.get("type") == "diode" and "is" not in c for c in case._components):
            yield rel, case


def timed(case, kw, engine="pwl"):
    """(wall of simulate(), result or the error text)."""
    orig = pulsim.simulate
    box = {}

    def sim(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig(*a, **{**k, **kw})
        finally:
            box["t"] = time.perf_counter() - t0
    pulsim.simulate = sim
    try:
        res = case.run(engine)
    except Exception as ex:  # noqa: BLE001 - a refused / failing cell is reported, not timed
        res = f"{type(ex).__name__}: {ex}"[:120]
    finally:
        pulsim.simulate = orig
    return box.get("t", float("nan")), res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--configs", default="pwl,pwl+restart,pwl+split")
    ap.add_argument("--circuits", default="circuits/*")
    ap.add_argument("--save")
    ap.add_argument("--compare")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    load = os.getloadavg()
    print(f"pulsim from {pulsim.__file__}; load average {load[0]:.1f} {load[1]:.1f} {load[2]:.1f}")
    saved, base = {}, (dict(np.load(a.compare)) if a.compare else None)
    rows, ndiff = [], 0
    print(f"{'circuit':52s} {'config':12s} {'default ms':>10s} {'lcp ms':>9s} {'ratio':>6s}  "
          f"{'quiet+cached/sites':>18s} same")
    for rel, case in cells(a.circuits):
        for cfg in a.configs.split(","):
            eng, kw = rs.CONFIGS[cfg]
            td, tl, last = [], [], None
            for _ in range(a.reps):
                t, _r = timed(case, kw, eng)
                td.append(t)
                t, last = timed(case, dict(kw, mode_selection="lcp"), eng)
                tl.append(t)
            name = rel.removeprefix("circuits/")
            if isinstance(last, str):
                print(f"{name:52s} {cfg:12s} refused/error: {last}")
                continue
            key = f"{name}|{cfg}"
            arrs = {f"{key}|t": np.asarray(last.times), f"{key}|x": np.asarray(last.states),
                    f"{key}|s": np.asarray(last.switch_states)}
            saved.update(arrs)
            same = ""
            if base is not None:
                ok = all(k in base and base[k].shape == v.shape
                         and base[k].tobytes() == v.tobytes() for k, v in arrs.items())
                same = "yes" if ok else "NO"
                ndiff += not ok
            ms = last.diagnostics["mode_selection"]
            nsites = sum(ms["sites"].values())
            md, ml = statistics.median(td), statistics.median(tl)
            rows.append(dict(circuit=name, config=cfg, default_ms=md * 1e3, lcp_ms=ml * 1e3,
                             ratio=ml / md))
            print(f"{name:52s} {cfg:12s} {md * 1e3:10.2f} {ml * 1e3:9.2f} {ml / md:6.3f}  "
                  f"{ms.get('quiet_sites', 0):>5}+{ms.get('cached_sites', 0)}/{nsites:<6} {same}", flush=True)
    for cfg in a.configs.split(","):
        r = [x["ratio"] for x in rows if x["config"] == cfg]
        if r:
            print(f"{cfg}: {len(r)} circuits, ratio median {statistics.median(r):.3f}, "
                  f"max {max(r):.3f}, over 1.2: {sum(x > 1.2 for x in r)}")
    load = os.getloadavg()
    print(f"load average at the end {load[0]:.1f} {load[1]:.1f} {load[2]:.1f}")
    if base is not None:
        print(f"bit-identical to {a.compare}: {'ALL' if ndiff == 0 else f'{ndiff} cells differ'}")
    if a.save:
        np.savez(a.save, **saved)
    if os.environ.get("LCP_GATE_JSON"):
        Path(os.environ["LCP_GATE_JSON"]).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
