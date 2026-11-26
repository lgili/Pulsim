"""TEMPORARY CI diagnostic (ci/windows-ten-bridges-diag): ten branch-form bridges under the
exact engine's output rule, scaled 1..10, wall time + engine counters per run. Each run is a
subprocess with its own hard timeout so the job always finishes and prints its numbers.

usage: python ten_bridges_diag.py            (driver; writes diag_results.jsonl)
       python ten_bridges_diag.py child N T_END RULE GATE TIMEOUT
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("diag_results.jsonl")


def _scalars(obj):
    out = {}
    for k in dir(obj):
        if k.startswith("_"):
            continue
        try:
            v = getattr(obj, k)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(v, (bool, int, float, str)) and not callable(v):
            out[k] = v
    return out


def _jsonable(v, depth=0):
    if isinstance(v, dict):
        return {str(k): _jsonable(x, depth + 1) for k, x in v.items()}
    if isinstance(v, (list, tuple)) and v and all(isinstance(x, dict) for x in v):
        # per-mask records: their count and the sum of each numeric field
        tot = {}
        for x in v:
            for k, y in x.items():
                if isinstance(y, (int, float)) and not isinstance(y, bool) and y == y:
                    tot[k] = tot.get(k, 0) + y
        return {"len": len(v), "sum": tot}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x, depth + 1) for x in v[:20]] + (["...%d" % len(v)] if len(v) > 20 else [])
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    return str(v)[:200]


def child(n, t_end, rule, gate, timeout):
    import faulthandler
    import warnings
    faulthandler.dump_traceback_later(max(5.0, timeout - 10.0), exit=False)
    # appended, not prepended: the installed pulsim must win over the source tree
    sys.path += [str(ROOT / "python" / "tests"), str(ROOT / "python")]
    import numpy as np
    import pulsim as p
    from test_release_findings_f import _branch_bridges

    b = _branch_bridges(n)
    kw = {"output_rtol": 1e-5} if rule else {}
    t0 = time.perf_counter()
    if gate == "python":
        ns = b.graph.num_switches
        st = {"calls": 0, "last": t0, "tmax": 0.0}

        def fn(t):
            st["calls"] += 1
            st["tmax"] = max(st["tmax"], t)
            now = time.perf_counter()
            if now - st["last"] > 20.0:
                st["last"] = now
                print("PROGRESS wall=%.0f t=%.9e calls=%d" % (now - t0, st["tmax"], st["calls"]),
                      flush=True)
            return p.SwitchStateMask(ns)
        kw["switch_fn"] = fn
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = p.simulate(b, t_end=t_end, engine="exact", **kw)
    wall = time.perf_counter() - t0
    rec = {"n": n, "t_end": t_end, "rule": rule, "gate": gate, "wall": round(wall, 3),
           "n_times": len(r.times), "t_last": float(r.times[-1]),
           "n_events": len(r.event_times),
           "bus_end": float(np.asarray(r.v("p"))[-1] - np.asarray(r.v("n"))[-1]),
           "warnings": sorted({str(w.message)[:160] for w in caught})[:8],
           "scalars": _scalars(r)}
    for k in ("exact", "diagnostics"):
        try:
            rec[k] = _jsonable(getattr(r, k))
        except Exception as e:  # noqa: BLE001
            rec[k] = "ERR %r" % e
    if gate == "python":
        rec["gate_calls"] = st["calls"]
    print("RESULT " + json.dumps(rec, default=str), flush=True)


# (n, t_end, rule, gate, timeout s)
PLAN = [
    (1, 0.02, True, "constant", 300),
    (2, 0.02, True, "constant", 300),
    (4, 0.02, True, "constant", 600),
    (10, 0.02, False, "constant", 900),
    (7, 0.02, True, "constant", 1200),
    (10, 0.005, True, "constant", 900),
    (10, 0.02, True, "python", 1800),
]


def driver():
    import platform
    print("PLATFORM", platform.platform(), sys.version.replace("\n", " "), flush=True)
    for n, t_end, rule, gate, timeout in PLAN:
        tag = "n=%d t_end=%g rule=%s gate=%s" % (n, t_end, rule, gate)
        print("=== RUN " + tag, flush=True)
        t0 = time.perf_counter()
        try:
            cp = subprocess.run([sys.executable, __file__, "child", str(n), repr(t_end),
                                 str(int(rule)), gate, str(timeout)],
                                capture_output=True, text=True, timeout=timeout)
            out, err, status = cp.stdout, cp.stderr, "rc=%d" % cp.returncode
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
            status = "TIMEOUT"
        wall = time.perf_counter() - t0
        print(out[-20000:], flush=True)
        print("--- stderr (tail)\n" + err[-6000:], flush=True)
        print("=== END %s status=%s wall=%.1f" % (tag, status, wall), flush=True)
        res = [ln[7:] for ln in out.splitlines() if ln.startswith("RESULT ")]
        prog = [ln for ln in out.splitlines() if ln.startswith("PROGRESS")]
        with OUT.open("a") as f:
            f.write(json.dumps({"tag": tag, "status": status, "wall": round(wall, 1),
                                "result": json.loads(res[0]) if res else None,
                                "progress_tail": prog[-30:]}) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(int(sys.argv[2]), float(sys.argv[3]), bool(int(sys.argv[4])), sys.argv[5],
              float(sys.argv[6]))
    else:
        driver()
