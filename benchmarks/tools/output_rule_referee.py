"""The referee of the exact engine's output rule (group F, session 7).

The rule promises: the straight line between two consecutive stored samples is within
``max(output_rtol * range, atol, floor)`` of the exact flow at EVERY instant of the line,
for every stored column (``range``: the column's range over the run so far, at least 1e-3 of
its peak; ``floor``: the column's rounding floor when the engine reports one,
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
  jump; only reference instants within ``EVENT_TOL`` of an event are skipped, and those
  within ``SETTLE`` of the run after one are read apart: the rule settles the parasitic
  cluster at the event, the rule-off reference lets it decay -- ``ratio_incl_settle``);
* the denominator is the promise itself with the RUN's range, no added slack (the promise's
  range is "so far, through the end of the exact step holding the line"; the stored trace does
  not show the steps, so the so-far reading through the line's end, :func:`promise`, is
  reported beside it as ``ratio_range_so_far``: it can only be stricter);
* every STORED column is read (``x[j]``), not only the benchmark's declared observables.

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
# The rule's event projection removes the parasitic cluster (time constants below 1e-6 of the
# run's resolved scale) over 40 of its time constants: at most 4e-5 of the run after an event,
# where the rule-off reference still shows that decay (diode_half_wave_rl_load: 0.31 V over
# ~7 ns). Reference instants inside that window are read separately (``ratio_incl_settle``).
SETTLE = 4e-5
# Since session 9 the window is per projected event, from the engine's projection record
# (diagnostics['output_grid']['projections']): [t, t + SETTLE_TAUS * tau], tau the projected
# cluster's slowest time constant; an event with no record has no window. Past the record's cap
# (1000) the old run-span window is used and the row says so (``settle_fallback``).
SETTLE_TAUS = 40.0


class RefereeError(Exception):
    """The harness could not read the flow on every line (a failure, not a pass)."""


def _in_windows(tq, windows):
    """tq inside any [a, b] of ``windows`` (an (n, 2) array)."""
    w = np.asarray(windows, dtype=float).reshape(-1, 2)
    if not w.size:
        return np.zeros(np.shape(tq), bool)
    w = w[np.argsort(w[:, 0])]
    k = np.searchsorted(w[:, 0], tq, side="right") - 1
    return (k >= 0) & (tq <= np.maximum.accumulate(w[:, 1])[np.clip(k, 0, None)])


def lines_ratio(t, y, tg, yg, tol, *, events=(), settle=0.0, windows=None):
    """Worst ``|line(tg) - yg| / tol`` over the stored lines of (t, y), read at every
    reference instant ``tg`` strictly inside a line. Raises RefereeError when a line of
    non-zero length holds no reference instant, or on a NaN."""
    acc = _Lines()
    acc.add(t, y, tg, yg, tol, events=events, settle=settle, windows=windows)
    return acc.result()


class _Lines:
    """:func:`lines_ratio` added up over pieces of the trace, each piece the stored lines of one
    reference window with that window's instants (a stored line lies in one window): the worst
    (first at a tie), the missing lines and the reads of the pieces are those of one call on the
    whole (F-R-1: the referee no longer keeps every window's reference)."""

    def __init__(self):
        self.e = self.at = self.first = None
        self.nan, self.n_missing, self.n_read = False, 0, 0

    def result(self):
        if self.nan:
            raise RefereeError("NaN in the trace or the reference, or no tolerance")
        if self.n_missing:
            raise RefereeError(f"{self.n_missing} line(s) without a reference instant, first "
                               f"[{self.first[0]:.17g}, {self.first[1]:.17g}]")
        if not self.n_read:   # nothing read is not a pass
            raise RefereeError("no reference instant read on any line")
        return self.e, self.at

    def add(self, t, y, tg, yg, tol, *, events=(), settle=0.0, windows=None):
        t, y, tg, yg = (np.asarray(a, dtype=float) for a in (t, y, tg, yg))
        tol = np.broadcast_to(np.asarray(tol, dtype=float), t.shape)   # per stored sample (a line reads its end's)
        if np.isnan(y).any() or np.isnan(yg).any() or not np.isfinite(tol).all() or (tol <= 0).any():
            self.nan = True
            return
        if t.size < 2:
            return
        self._add(t, y, tg, yg, tol, events, settle, windows)

    def _add(self, t, y, tg, yg, tol, events, settle, windows):
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
            # the rule's event projection settles the parasitic cluster at once; the rule-off
            # reference lets it decay: instants inside that window are not the same flow
            if windows is not None:
                near |= _in_windows(tg, windows)
            elif settle > 0:
                d = tg - ev[np.clip(k - 1, 0, ev.size - 1)]
                near |= (d >= 0) & (d <= settle)
            inside &= ~near
        hit = np.zeros(t.size, bool)
        hit[j[inside]] = True
        # a line too short to hold an instant away from an event (a jump's ulp step) is not a line
        short = (t[keep + 1] - t[keep]) <= 2 * EVENT_TOL
        if windows is not None:   # a line that starts inside a settle window may hold no instant outside it
            short |= _in_windows(t[keep], windows)
        elif settle > 0 and ev.size:
            ka = np.clip(np.searchsorted(ev, t[keep], side="right") - 1, 0, ev.size - 1)
            short |= (t[keep] >= ev[ka]) & (t[keep] <= ev[ka] + settle)
        missing = keep[~hit[keep] & ~short]
        if missing.size:
            a = int(missing[0])
            if self.first is None:
                self.first = (t[a], t[a + 1])
            self.n_missing += int(missing.size)
        jj = j[inside]
        lin = y[jj] + (y[jj + 1] - y[jj]) * (tg[inside] - t[jj]) / (t[jj + 1] - t[jj])
        e = np.abs(lin - yg[inside]) / tol[jj + 1]
        self.n_read += int(e.size)
        if e.size:
            q = int(np.argmax(e))
            if self.e is None or e[q] > self.e:
                self.e, self.at = float(e[q]), float(tg[inside][q])


def promise(y, rt, floor):
    """The promise at each stored sample, exactly as the option states it:
    ``max(rt * range, ATOL, floor)``, ``range`` the column's range over the run SO FAR (through
    that sample; the engine's own goes on to the end of the step holding the line, so this
    reading is never looser than the engine's), at least 1e-3 of its peak so far. A
    diagnostic beside the verdict, which uses the run's range."""
    y = np.asarray(y, dtype=float)
    hi, lo = np.maximum.accumulate(y), np.minimum.accumulate(y)
    rng = np.maximum(hi - lo, 1e-3 * np.maximum(np.abs(hi), np.abs(lo)))
    return np.maximum(rt * rng, max(ATOL, floor))


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


def check(unit: str, config: str = "auto+next") -> dict:
    warnings.simplefilter("ignore")
    case = v1_reader.load(R.ROOT / unit)
    cols = dr.observables_of(case)
    kind, res, _, msg, info = R.run_cell(case, config)
    rt = R._rule_rtol(res) if kind == "ran" else None
    if not rt:
        return dict(unit=unit, comparable=False, why=f"no rule cell ({kind}: {msg})")
    return dict(unit=unit, **referee(case, res, info, cols, rt, config))


def referee(case, res, info, cols, rt, config: str = "auto+next") -> dict:
    """The promise on one rule run ``res`` of ``case`` (``info``: run_cell's, for the
    route): ``comparable`` with ``ratio`` (worst over ``cols``), or not with ``why``."""
    spec = R.CONFIGS[config]
    og = res.diagnostics["output_grid"]
    if og.get("n_output_capped") or og.get("n_output_unbounded") or og.get("output_budget_t") is not None:
        return dict(comparable=False, why=f"rule suspended (cap/unbounded/budget): "
                    f"{ {k: og.get(k) for k in ('n_output_capped', 'n_output_unbounded', 'output_budget_t')} }")
    t, ys, _ = dr.traces(res, cols)
    ev = np.asarray(res.event_times if res.event_times is not None else [], dtype=float)
    eng = info.get("route") or info.get("engine_used") or "dsed"
    eng = "dsed" if eng == "exact" else eng
    t0 = time.perf_counter()
    import pulsim
    fl = og.get("output_floor_by_column")
    settle = SETTLE * float(t[-1] - t[0])
    # the settle windows: one per projected event, from the engine's record
    pr = og.get("projections") or []
    win = [(q["t"], q["t"] + SETTLE_TAUS * q["tau"]) for q in pr]
    fallback = int(og.get("n_projections") or 0) > len(pr)
    if fallback:   # past the record's cap: the run-span window for the later events
        t_last = max((q["t"] for q in pr), default=-np.inf)
        win += [(e, e + settle) for e in ev if e > t_last]
    win = np.asarray(win, dtype=float).reshape(-1, 2)

    # a source corner is no event, but both runs store a jump there (its two limits a few ulps
    # apart, the rule run's corner landed ulps from the reference's): reference instants within
    # EVENT_TOL of a jump of either trace are not read (pwm_voltage_rc's PWM source column read
    # 1e5 x at 40 us: the reference's left limit against the rule's right one)
    # 1 mantissa: a few-ulp pair is a jump; a fine reference grid (dt <= 2 ps) or the rule's
    # 2 ps sample past an event is not (it hid every instant of rlc_step's first 28 ps)
    def _jumps(s):
        return s[:-1][np.diff(s) <= np.maximum(64 * np.finfo(float).eps * np.abs(s[:-1]), 1e-300)]

    # the promise per column (the rule run's own trace and floors); a floor that cannot be
    # mapped fails in the column's turn, as the whole-run reading did
    tols, floors, order = {}, {}, []
    for c in cols:
        try:
            floors[c] = _column_floor(res, c, fl)
        except Exception as ex:  # noqa: BLE001 - raised in its turn below
            order.append(ex)
            continue
        # the range of the rule run's own (projected) trace: the unprojected reference's
        # parasitic excursions would widen the tolerance (session 9 reviewer)
        allv = np.asarray(ys[c], dtype=float)
        rng = max(float(np.ptp(allv)), 1e-3 * float(np.abs(allv).max()))
        tols[c] = max(rt * rng, ATOL, floors[c])
        order += [(c, "r"), (c, "whole"), (c, "so_far")]
    acc = {k: _Lines() for k in order if isinstance(k, tuple)}
    sofar_tol = {c: promise(ys[c], rt, floors[c]) for c in tols}
    xs = np.asarray(dr.with_right_limits(res).states, dtype=float)
    x_tols = None          # set at the first window, which gives the reference's column count
    jumps_t = _jumps(t)

    def read(w):
        """One window's verdict parts; ``w``: (t0, t1, tg, {col: yg}, xg) of the window."""
        lo = 0 if w[0] is None else int(np.searchsorted(t, w[0], side="left"))
        hi = int(np.searchsorted(t, w[1], side="right"))
        ts, tg_w = t[lo:hi], w[2]
        evx = np.unique(np.r_[ev, jumps_t, jumps_tg])
        for c in tols:
            yc, yg = np.asarray(ys[c], dtype=float)[lo:hi], w[3][c]
            acc[(c, "r")].add(ts, yc, tg_w, yg, tols[c], events=evx, windows=win)
            acc[(c, "whole")].add(ts, yc, tg_w, yg, tols[c], events=evx)
            acc[(c, "so_far")].add(ts, yc, tg_w, yg, sofar_tol[c][lo:hi], events=evx, windows=win)
        for j, tol_j in enumerate(x_tols or ()):
            acc[f"x[{j}]"].add(ts, xs[lo:hi, j], tg_w, w[4][:, j], tol_j, events=evx, windows=win)

    # the reference: the same route, rule off, chained windows (one build: a closed loop's
    # controller lives in its Python objects and continues across windows). Streamed (F-R-1):
    # a window is read once the next one is known (a jump of the reference at the seam belongs
    # to both), then released; only the event times and jump instants are kept.
    eg_all, jumps_tg, pending, last = [], np.zeros(0), None, None
    snap, n_windows, n_ref, n_seams, t_prev = None, 0, 0, 0, None
    with warnings.catch_warnings(), nd.builder_form(spec["form"]), nd.plan_defaults(spec.get("plan", False)):
        warnings.simplefilter("ignore")
        # built under the cell's builder form (as run_cell builds the rule run): built outside
        # it, the reference was another circuit form with another column set, and no stored
        # column was read (cl_buck_pi: 13 columns against 11)
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
        kw = ({k: v for k, v in nd.profile_kwargs(eng).items() if k != "output_rtol"}
              if spec.get("profile") else dict(spec.get("kw") or {}))
        kw.pop("output_rtol", None)
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
            tg = np.asarray(g.times, dtype=float)
            xg = np.asarray(g.states, dtype=float)
            yg = {c: np.asarray(v1_reader.observable(g, c), dtype=float) for c in cols}
            if g.event_times is not None:
                eg_all.append(np.asarray(g.event_times, dtype=float))
            del g
            n_windows += 1
            # a seam of the window chain (the resumed window's first sample repeats the last
            # one, every value bit-equal) is no jump: read as one, it hid every instant of a
            # 2 ps line between two seams (ll03_rl_step's source corner at 100 ns, session 10)
            if last is None:
                seam = np.zeros(tg.size, bool)
                seam[1:] = ((np.diff(tg) == 0) & (xg[1:] == xg[:-1]).all(axis=1)
                            & np.all([yg[c][1:] == yg[c][:-1] for c in cols], axis=0))
            else:
                pt = np.r_[last[0], tg]
                px = np.vstack([last[1], xg])
                py = {c: np.r_[last[2][c], yg[c]] for c in cols}
                seam = ((np.diff(pt) == 0) & (px[1:] == px[:-1]).all(axis=1)
                        & np.all([py[c][1:] == py[c][:-1] for c in cols], axis=0))
            last = (tg[-1], xg[-1:].copy(), {c: yg[c][-1:].copy() for c in cols})
            n_seams += int(seam.sum())
            tg, xg = tg[~seam], xg[~seam]
            yg = {c: v[~seam] for c, v in yg.items()}
            n_ref += int(tg.size)
            # the jumps of the seam-free reference, its pair across the seam included
            jumps_tg = np.r_[jumps_tg, _jumps(np.r_[pending[2][-1:], tg] if pending is not None else tg)]
            if x_tols is None:
                # (a dt run of some circuits stores another column set: then only the
                # observables are read, said so)
                x_tols = []
                if xs.shape == (t.size, xg.shape[1]) and (fl is None or len(fl) == xs.shape[1]):
                    x_tols = [max(rt * max(float(np.ptp(xs[:, j])), 1e-3 * float(np.abs(xs[:, j]).max())), ATOL,
                                  float(fl[j]) if fl is not None else 0.0) for j in range(xs.shape[1])]
                    for j in range(len(x_tols)):
                        acc[f"x[{j}]"] = _Lines()
                        order.append(f"x[{j}]")
            if pending is not None:
                read(pending)
            pending = (t_prev, t_w, tg, yg, xg)
            t_prev = t_w
            del tg, xg, yg
        if pending is not None:
            read(pending)
        pending = None
    eg = np.unique(np.concatenate(eg_all)) if eg_all else np.zeros(0)
    if eg.size != ev.size or (ev.size and float(np.abs(eg - ev).max()) > EVENT_TOL):
        n = min(eg.size, ev.size)
        return dict(comparable=False, why=f"reference events differ: {ev.size} vs {eg.size}, "
                    f"max |dt_e| {float(np.abs(eg[:n] - ev[:n]).max()) if n else float('nan'):.3g}")
    # the verdicts in the whole-run reading's order (its first failure is raised)
    got = {}
    for k in order:
        if isinstance(k, Exception):
            raise k
        got[k] = acc[k].result()
    worst, where, per, worst_whole = 0.0, None, {}, 0.0
    for c in tols:
        r, at = got[(c, "r")]
        whole = got[(c, "whole")][0]
        worst_whole = max(worst_whole, whole)
        # also against the range SO FAR through the line's end sample: stricter than the
        # engine's own (which goes on to the end of the exact step holding the line, a step
        # the stored trace does not show), so above 1 early in a run without a violation
        per[c] = dict(ratio=r, floor=floors[c], tol=tols[c], ratio_incl_settle=whole,
                      ratio_range_so_far=got[(c, "so_far")][0])
        if r > worst:
            worst, where = r, (c, at)
    # every stored column, each against its own promise and floor
    worst_x, n_x = (0.0, len(x_tols)) if x_tols else (None, 0)
    for j in range(n_x):
        r, at = got[f"x[{j}]"]
        worst_x = max(worst_x, r)
        if r > worst:
            worst, where = r, (f"x[{j}]", at)
    return dict(comparable=True, ratio=worst, at=where, ratio_incl_settle=worst_whole, settle_s=settle,
                settle_windows=int(win.shape[0]), settle_fallback=fallback, ratio_stored_columns=worst_x, n_columns=n_x, n_lines=int(np.sum(np.diff(t) > 0)),
                n_ref=n_ref, n_windows=n_windows, n_seams=n_seams, rtol=rt, per=per,
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
