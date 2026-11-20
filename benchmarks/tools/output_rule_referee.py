"""The referee of the exact engine's output rule (group F, session 7).

The rule promises: the straight line between two consecutive stored samples is within
``max(output_rtol * range, atol, floor)`` of the exact flow at EVERY instant of the line,
for every stored column (``range``: the column's range over the run, at least 1e-3 of its
peak; ``floor``: the column's rounding floor when the engine reports one,
``diagnostics['output_grid']['output_floor_by_column']``).

How it reads the flow, so that no stored line can go unread:
* the reference is the SAME route with the rule off, run as a chain of windows that tile the
  run at stored instants (``resume_from`` the previous window's snapshot); each window's
  ``dt`` is the shortest stored line in it over ``K`` -- every line holds at least ``K - 1``
  reference instants, wherever they fall (not only its midpoint, and never only where a
  dense run happens to have a sample: a dense run that shares the line reads 0 there);
* a line with no reference instant inside, a NaN, other events, a window that cannot resume
  or a column without a floor mapping is a FAILURE of the harness (``comparable`` False, the
  reason given), never a 0 or a pass;
* every line counts, the first one after an event too (the trace carries both limits of a
  jump; only reference instants within ``EVENT_TOL`` of an event are skipped);
* the denominator is the promise itself, no added slack.

    python benchmarks/tools/output_rule_referee.py run --units FILE --out OUT.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import differential_runner as dr  # noqa: E402
import next_defaults as nd  # noqa: E402
import next_defaults_rehearsal as R  # noqa: E402
import v1_reader  # noqa: E402

EVENT_TOL = 1e-12     # s: reference instants this close to an event are not read; events further apart differ
K = 6                 # each line holds >= K - 1 reference instants
WINDOW_POINTS = 40_000   # planned instants (a gate poll can add ~10x samples)
ATOL = 1e-9


class RefereeError(Exception):
    """The harness could not read the flow on every line (a failure, not a pass)."""


def lines_ratio(t, y, tg, yg, tol, *, events=()):
    """Worst ``|line(tg) - yg| / tol`` over the stored lines of (t, y), read at every
    reference instant ``tg`` strictly inside a line. Raises RefereeError when a line of
    non-zero length holds no reference instant, or on a NaN."""
    t, y, tg, yg = (np.asarray(a, dtype=float) for a in (t, y, tg, yg))
    if np.isnan(y).any() or np.isnan(yg).any() or not np.isfinite(tol) or tol <= 0:
        raise RefereeError("NaN in the trace or the reference, or no tolerance")
    ev = np.asarray(events, dtype=float)
    keep = np.flatnonzero(np.diff(t) > 0)
    j = np.searchsorted(t, tg, side="right") - 1
    inside = (j >= 0) & (j < t.size - 1)
    j = np.where(inside, j, 0)
    inside &= tg > t[j]
    inside &= tg < t[np.minimum(j + 1, t.size - 1)]
    if ev.size:
        k = np.searchsorted(ev, tg)
        near = np.zeros(tg.shape, bool)
        for o in (-1, 0):
            kk = np.clip(k + o, 0, ev.size - 1)
            near |= np.abs(tg - ev[kk]) <= EVENT_TOL
        inside &= ~near
    hit = np.zeros(t.size, bool)
    hit[j[inside]] = True
    # a line too short to hold an instant away from an event (a jump's ulp step) is not a line
    short = (t[keep + 1] - t[keep]) <= 2 * EVENT_TOL
    missing = keep[~hit[keep] & ~short]
    if missing.size:
        a = int(missing[0])
        raise RefereeError(f"{missing.size} line(s) without a reference instant, first "
                           f"[{t[a]:.17g}, {t[a + 1]:.17g}]")
    jj = j[inside]
    lin = y[jj] + (y[jj + 1] - y[jj]) * (tg[inside] - t[jj]) / (t[jj + 1] - t[jj])
    e = np.abs(lin - yg[inside])
    if not e.size:
        return 0.0, None
    q = int(np.argmax(e))
    return float(e[q] / tol), float(tg[inside][q])


def _windows(t, span_end):
    """Window ends (stored instants) and each window's dt: dt = shortest line / K, at most
    WINDOW_POINTS reference instants per window."""
    lines = np.diff(t)
    w0, dmin, out = t[0], np.inf, []
    for a in range(t.size - 1):
        L = lines[a]
        if L <= 2 * EVENT_TOL:
            continue
        nd_ = min(dmin, L / K)
        if (t[a + 1] - w0) / nd_ > WINDOW_POINTS and np.isfinite(dmin) and t[a] > w0:
            out.append((float(t[a]), dmin))
            w0, nd_ = t[a], L / K
        dmin = nd_
    out.append((float(span_end), dmin if np.isfinite(dmin) else (span_end - w0) / K))
    return out


def _column_floor(res, col, fl):
    """A column's rounding floor from the per-stored-column floors (|.|-summed through the
    observable's map; an observable the map cannot express raises)."""
    if fl is None or not np.any(fl):
        return 0.0
    from pulsim import SimulationResult
    proxy = SimulationResult()
    proxy._adapter_fill(np.zeros(1), np.abs(np.asarray(fl, dtype=float))[None, :], None, [])
    proxy._builder = res._builder
    kind, _, rest = col.partition("(")
    name = rest.rstrip(")")
    if kind.upper() == "V":
        nodes = [x.strip() for x in name.split(",")]
        return float(sum(abs(v1_reader.observable(proxy, f"V({n})")[0]) for n in nodes))
    raise RefereeError(f"no floor mapping for {col} (floored columns reported)")


def check(unit: str) -> dict:
    warnings.simplefilter("ignore")
    case = v1_reader.load(R.ROOT / unit)
    cols = dr.observables_of(case)
    kind, res, _, msg, info = R.run_cell(case, "auto+next")
    rt = R._rule_rtol(res) if kind == "ran" else None
    if not rt:
        return dict(unit=unit, comparable=False, why=f"no rule cell ({kind}: {msg})")
    og = res.diagnostics["output_grid"]
    if og.get("n_output_capped") or og.get("n_output_unbounded") or og.get("output_budget_t") is not None:
        return dict(unit=unit, comparable=False, why=f"rule suspended (cap/unbounded/budget): "
                    f"{ {k: og.get(k) for k in ('n_output_capped', 'n_output_unbounded', 'output_budget_t')} }")
    t, ys, _ = dr.traces(res, cols)
    ev = np.asarray(res.event_times if res.event_times is not None else [], dtype=float)
    eng = info.get("route") or "dsed"
    t0 = time.perf_counter()
    # the reference: the same route, rule off, chained windows (one build: a closed loop's
    # controller lives in its Python objects and continues across windows)
    b, sw, obs = case.build()
    args = dict(engine=eng, t_start=case.t_start)
    if sw is not None:
        args["switch_fn"] = sw
    if obs is not None:
        args["step_observer"] = obs
    if not case.uic:
        b0, sw0, _ = case.build()
        x0 = case._dc_start(b0, sw0)
        if x0 is not None:
            args["initial_state"] = x0
    import pulsim
    tg_all, yg_all, eg_all = [], {c: [] for c in cols}, []
    snap = None
    with warnings.catch_warnings(), nd.builder_form(True), nd.plan_defaults(True):
        warnings.simplefilter("ignore")
        kw = {k: v for k, v in nd.profile_kwargs(eng).items() if k != "output_rtol"}
        for t_w, dt in _windows(t, float(t[-1])):
            a = dict(args, **kw, dt=float(dt))
            if snap is not None:
                a.pop("initial_state", None)
                a.pop("t_start", None)
                a["resume_from"] = snap
            g = pulsim.simulate(b, t_w, **a)
            snap = g.final_snapshot
            if not getattr(snap, "valid", False):
                raise RefereeError(f"window ending {t_w:.6g} keeps no snapshot")
            tg_all.append(np.asarray(g.times, dtype=float))
            for c in cols:
                yg_all[c].append(np.asarray(v1_reader.observable(g, c), dtype=float))
            if g.event_times is not None:
                eg_all.append(np.asarray(g.event_times, dtype=float))
    tg = np.concatenate(tg_all)
    eg = np.unique(np.concatenate(eg_all)) if eg_all else np.zeros(0)
    if eg.size != ev.size or (ev.size and float(np.abs(eg - ev).max()) > EVENT_TOL):
        n = min(eg.size, ev.size)
        return dict(unit=unit, comparable=False, why=f"reference events differ: {ev.size} vs {eg.size}, "
                    f"max |dt_e| {float(np.abs(eg[:n] - ev[:n]).max()) if n else float('nan'):.3g}")
    fl = og.get("output_floor_by_column")
    worst, where, per = 0.0, None, {}
    for c in cols:
        yg = np.concatenate(yg_all[c])
        allv = np.r_[ys[c], yg]
        rng = max(float(np.ptp(allv)), 1e-3 * float(np.abs(allv).max()))
        floor = _column_floor(res, c, fl)
        tol = max(rt * rng, ATOL, floor)
        r, at = lines_ratio(t, ys[c], tg, yg, tol, events=ev)
        per[c] = dict(ratio=r, floor=floor, tol=tol)
        if r > worst:
            worst, where = r, (c, at)
    return dict(unit=unit, comparable=True, ratio=worst, at=where, n_lines=int(np.sum(np.diff(t) > 0)),
                n_ref=int(tg.size), n_windows=len(tg_all), rtol=rt, per=per,
                n_output_floored=og.get("n_output_floored"), output_floor_max=og.get("output_floor_max"),
                wall_ref_s=round(time.perf_counter() - t0, 2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--units", required=True, help="file with one benchmark unit per line")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    for u in Path(a.units).read_text(encoding="utf-8").split():
        try:
            rec = check(u)
        except Exception as ex:  # noqa: BLE001 - a harness failure is reported, never a pass
            rec = dict(unit=u, comparable=False, why=f"{type(ex).__name__}: {ex}"[:300])
        with open(a.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
