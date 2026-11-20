"""Property check of the exact engine's output rule (group F, session 4).

For each circuit: the auto+next run (the rule on, no dt) and the SAME route with a
``dt`` grid (the rule off: every stored sample is the exact flow at k·dt). At up to
``--points`` random grid instants strictly inside a rule chord that holds no event,
the chord's linear interpolation must be within the promise,
``|interp − flow| ≤ output_rtol·range + atol`` (range: the flow's own range over
the run, at least 1e-3 of its peak), for every observable. Events of the two runs
must agree (else the circuit is listed as not comparable, never as passing).

    python benchmarks/tools/output_rule_property.py run [--units FILE] [--dt-factor 100]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import differential_runner as dr  # noqa: E402
import next_defaults as nd  # noqa: E402
import next_defaults_rehearsal as R  # noqa: E402
import v1_reader  # noqa: E402


def check(unit: str, points: int = 200, dt_factor: float = 100.0, max_grid: int = 400_000,
          seed: int = 0) -> dict:
    warnings.simplefilter("ignore")
    case = v1_reader.load(R.ROOT / unit)
    cols = dr.observables_of(case)
    kind, res, _, msg, info = R.run_cell(case, "auto+next")
    rt = R._rule_rtol(res) if kind == "ran" else None
    if not rt:
        return dict(unit=unit, comparable=False, why=f"no rule cell ({kind}: {msg})")
    t, ys, sa = dr.traces(res, cols)
    gaps = np.diff(t)
    gaps = gaps[gaps > 0]
    dt = float(np.median(gaps)) / dt_factor
    span = float(t[-1] - t[0])
    if span / dt > max_grid:
        dt = span / max_grid
    eng = info.get("route") or "dsed"
    with warnings.catch_warnings(), nd.builder_form(True), nd.plan_defaults(True):
        warnings.simplefilter("ignore")
        kw = {k: v for k, v in nd.profile_kwargs(eng).items() if k != "output_rtol"}
        rg = R._no_dt_run(case, eng, dt=dt, **kw)
    if not np.array_equal(np.asarray(res.event_times), np.asarray(rg.event_times)):
        ea, eb = np.asarray(res.event_times), np.asarray(rg.event_times)
        n = min(ea.size, eb.size)
        return dict(unit=unit, comparable=False, dt=dt,
                    why=f"events differ: {ea.size} vs {eb.size}, max |dt_e| "
                        f"{float(np.abs(ea[:n] - eb[:n]).max()) if n else float('nan'):.3g}")
    tg = np.asarray(rg.times, dtype=float)
    yg = {c: np.asarray(v1_reader.observable(rg, c), dtype=float) for c in cols}
    ev = np.asarray(res.event_times, dtype=float)
    excl = dr.switching_steps(sa, t) + dr.event_steps(res)
    # grid instants strictly inside a chord of the rule trace with no event in it
    j = np.searchsorted(t, tg, side="right") - 1
    ok = (j >= 0) & (j < t.size - 1)
    j = np.where(ok, j, 0)
    lo, hi = t[j], t[np.minimum(j + 1, t.size - 1)]
    ok &= (tg > lo) & (tg < hi) & (hi > lo)
    ok &= ~np.isin(tg, ev)
    evp = np.r_[-np.inf, ev, np.inf]
    k = np.searchsorted(ev, tg, side="left")
    prev, nxt = evp[k], evp[k + 1]
    ok &= (prev < lo) & (nxt >= hi)   # no event strictly inside (lo, hi)
    for a, b in excl:
        ok &= ~((lo < b) & (hi > a))
    idx = np.flatnonzero(ok)
    if not idx.size:
        return dict(unit=unit, comparable=False, dt=dt, why="no grid instant inside a chord")
    rng = np.random.default_rng(seed)
    idx = rng.choice(idx, size=min(points, idx.size), replace=False)
    worst, where = 0.0, None
    for c in cols:
        y, f = ys[c], yg[c]
        r = max(float(np.ptp(f)), 1e-3 * float(np.abs(f).max()))
        tol = rt * r + 1e-9
        jj = j[idx]
        lin = y[jj] + (y[jj + 1] - y[jj]) * (tg[idx] - t[jj]) / (t[jj + 1] - t[jj])
        e = np.abs(lin - f[idx]) / tol
        q = int(np.argmax(e))
        if e[q] > worst:
            worst, where = float(e[q]), (c, float(tg[idx][q]))
    return dict(unit=unit, comparable=True, ratio=worst, at=where, n_points=int(idx.size), dt=dt,
                n_rule=int(t.size), rtol=rt)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--units", required=True, help="file with one benchmark unit per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", type=int, default=200)
    ap.add_argument("--dt-factor", type=float, default=100.0)
    a = ap.parse_args(argv)
    for u in Path(a.units).read_text(encoding="utf-8").split():
        try:
            rec = check(u, a.points, a.dt_factor)
        except Exception as ex:  # noqa: BLE001 - reported per circuit
            rec = dict(unit=u, comparable=False, why=f"{type(ex).__name__}: {ex}"[:200])
        with open(a.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
