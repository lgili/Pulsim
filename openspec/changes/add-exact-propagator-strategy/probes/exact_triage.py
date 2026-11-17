"""Root-cause triage of the exact engine (engine='dsed' == 'exact') on the reference set.

    python3 exact_triage.py run   OUT.json [--circuits GLOB,...]   # run the dsed cells
    python3 exact_triage.py table OUT.json                          # the cause table

Every cell is graded exactly as benchmarks/tools/reference_status.py grades the 'dsed'
configuration (same reference, tolerance, strict `interp` gate). The cause of a cell that does
not agree is decided by `cause()` below from the typed refusal (feature / device / kind), the
error's origin and the circuit's devices -- by mechanism, not by the message wording alone.
References are cached under .scratch/exact_triage_refs (pwl / trbdf2 do not move on this
branch: bit-identity is an invariant); pass --fresh to recompute them.
"""
from __future__ import annotations

import fnmatch
import json
import pickle
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "benchmarks" / "tools"), str(ROOT / "benchmarks")]
import differential_runner as dr  # noqa: E402
import v1_reader  # noqa: E402

import pulsim  # noqa: E402

REF_DIR = ROOT / ".scratch" / "exact_triage_refs"
NEWTON = {"mosfet", "igbt", "saturable_inductor", "diode_shockley", "shockley_diode", "bjt",
          "jfet", "thyristor", "triac"}


def _ref(rel, case, cols, tol, fresh):
    REF_DIR.mkdir(parents=True, exist_ok=True)
    f = REF_DIR / (rel.replace("/", "__") + ".pkl")
    if f.exists() and not fresh:
        return pickle.loads(f.read_bytes())
    out = dr.reference(case, cols, tol)
    f.write_bytes(pickle.dumps(out))
    return out


def run(out_path, patterns, fresh=False):
    triage = dr.load_triage()
    recs = []
    for rel, entry in sorted(triage.items()):
        if entry["status"] != "runs" or not any(fnmatch.fnmatch(rel, p) for p in patterns):
            continue
        case = v1_reader.load(dr.ROOT / rel)
        _, sw, _ = case.build()
        switched = sw is not None or any(k in json.dumps(case._components)
                                         for k in ('"diode"', '"pwm"', '"pulse"'))
        cols = dr.observables_of(case)
        tols = {e: entry.get("tol", {}).get(e, dr.default_tol(e, switched)) for e in dr.ENGINES}
        ref_name, ref = _ref(rel, case, cols, min(tols.values()), fresh)
        rec = dict(circuit=rel, reference=ref_name, tol=tols["dsed"],
                   types=sorted({c["type"] for c in case._components}), uic=case.uic)
        t0 = time.perf_counter()
        try:
            res = case.run("dsed")
            rec["wall_s"] = round(time.perf_counter() - t0, 3)
            if ref is None:
                rec.update(outcome="error", message="no converged reference")
            else:
                dr.classify(rec, res, case, cols, ref, tols["dsed"], None)
                ex = getattr(res, "exact", None) or {}
                rec["auto_would_route"] = ex.get("auto_would_route")
                rec["n_masks"] = len(ex.get("masks", []) or [])
        except Exception as e:  # noqa: BLE001 - every outcome is classified
            rec["wall_s"] = round(time.perf_counter() - t0, 3)
            rec["outcome"] = "refused" if isinstance(e, dr.REFUSAL) else "error"
            rec["exc"] = type(e).__name__
            rec["message"] = str(e)[:600]
            for a in ("engine", "feature", "device", "kind"):
                if hasattr(e, a):
                    rec[a] = getattr(e, a)
            # a full state vector (the DC start) handed to dsed?
            try:
                rec["dc_start"] = not case.uic and case._dc_start(*case.build()[:2]) is not None
            except Exception:  # noqa: BLE001 - informational only
                rec["dc_start"] = None
        dr.mark_suspect(rec, entry)
        recs.append(rec)
        print(f"{rec['outcome']:9s} {rel} {rec.get('err_rms', '')} "
              f"{rec.get('feature', '')} {rec.get('message', '')[:110]}", flush=True)
    Path(out_path).write_text(json.dumps(recs, indent=1, default=str))


def cause(r):
    """Root cause of a cell that does not agree (None for agree / unverified)."""
    if r["outcome"] in ("agree", "unverified"):
        return None
    msg, feat = r.get("message", "") or "", r.get("feature", "")
    if r["outcome"] in ("refused", "error"):
        if feat == "device" or (r.get("kind") and feat in ("", "device")):
            return f"Newton device refused by name ({r.get('kind') or r.get('device')})"
        if feat == "initial_state" or "initial_state" in msg and "full" in msg:
            return "DC start refused (non-zero consistent start)"
        if "inductors form a cycle" in msg or "cut-set" in msg or "cutset" in msg:
            return "extraction: inductor loop / cut-set"
        if "capacitors form a cycle" in msg:
            return "extraction: capacitor loop"
        if "coupled" in msg.lower() or "mutual" in msg.lower():
            return "extraction: coupled inductors"
        if "singular" in msg:
            return "extraction / factor: singular matrix in a mode"
        if "did not settle" in msg:
            return "settle: diode configuration did not settle"
        if "chatter" in msg.lower():
            return "chatter detector"
        if "progress" in msg:
            return "no progress"
        if feat:
            return f"refused: {feat}"
        return "error: " + msg.split(":")[0][:60]
    return "disagree (to classify)"


def table(path):
    recs = json.loads(Path(path).read_text())
    n = Counter(r["outcome"] for r in recs)
    print("outcomes:", dict(n))
    by = Counter()
    rows = []
    for r in recs:
        c = cause(r)
        if c is None:
            continue
        by[c] += 1
        rows.append((c, r["circuit"].removeprefix("circuits/"), r["outcome"],
                     f"{r['err_rms']:.3g}" if "err_rms" in r else (r.get("message", "")[:90])))
    for c, k in by.most_common():
        print(f"\n## {c}: {k}")
        for row in sorted(x for x in rows if x[0] == c):
            print("  ", row[1], row[2], row[3])


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    cmd, path = sys.argv[1], sys.argv[2]
    pats = ["*"]
    if "--circuits" in sys.argv:
        pats = [p if p.startswith("circuits/") else f"circuits/{p}"
                for p in sys.argv[sys.argv.index("--circuits") + 1].split(",")]
    if cmd == "run":
        run(path, pats, fresh="--fresh" in sys.argv)
    else:
        table(path)
