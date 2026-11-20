"""Time-weighted comparison metrics for the reference harness.

Every waveform statistic here is weighted by TIME, never by sample: the
variable-step engines cluster samples around events, and an unweighted
``y.mean()`` once reported 3.04 V where the time-weighted value was
6.33 V (and was filed as an engine defect). Integrals are trapezoidal
on the piecewise-linear waveform through the samples.
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np

Reference = Union[Callable[[np.ndarray], np.ndarray], tuple]


def first_kept(t, exclude_t0: bool) -> int:
    """Index of the first sample a comparison keeps. ``exclude_t0`` drops the
    initial sample(s) (on pwl / trbdf2 the stored initial state of an algebraic
    node, not the t = 0+ solution: design.md, Context) -- unless that leaves fewer
    than two samples: a short trace (the exact engine stores a DC circuit as its
    two ends under the output rule) is then compared whole, not called
    undersampled."""
    t = np.asarray(t, dtype=float)
    if not exclude_t0:
        return 0
    k = int(np.searchsorted(t, t[0], side="right"))
    return k if t.size - k >= 2 else 0


def _window(t, y, exclude_t0: bool):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.shape != y.shape or t.size < 2 or not t[-1] > t[0]:
        raise ValueError(f"a trace needs >= 2 samples spanning a time window to be compared, "
                         f"got {t.shape} / {y.shape}")
    k = first_kept(t, exclude_t0)
    return t[k:], y[k:]


def _ref_window(t, ref, exclude_t0: bool):
    """A short trace kept whole (first_kept) is compared from the reference's own
    first kept sample on: the reference's initial state is not t = 0+ either."""
    if not exclude_t0 or callable(ref) or first_kept(t, True) > 0:
        return ref
    tr, yr = (np.asarray(a, dtype=float) for a in ref)
    k = first_kept(tr, True)
    return tr[k:], yr[k:]


def time_mean(t, y) -> float:
    """Time-weighted mean over the trace's own span."""
    t = np.asarray(t, dtype=float)
    return float(np.trapezoid(np.asarray(y, dtype=float), t) / (t[-1] - t[0]))


def _on_grid(t, y, ref):
    """Return (grid, y, ref) on a common grid.

    ``ref`` is a callable (analytic, evaluated on the trace's own grid) or a
    ``(t_ref, y_ref)`` trace (union of both grids, linear interpolation).
    """
    if callable(ref):
        return t, y, np.asarray(ref(t), dtype=float)
    tr, yr = (np.asarray(a, dtype=float) for a in ref)
    lo, hi = max(t[0], tr[0]), min(t[-1], tr[-1])
    g = np.union1d(t, tr)
    g = g[(g >= lo) & (g <= hi)]
    return g, np.interp(g, t, y), np.interp(g, tr, yr)


def _inside(g, exclude):
    """Mask of grid points strictly inside one of the ``(lo, hi)`` intervals."""
    iv = np.asarray(exclude, dtype=float).reshape(-1, 2)
    iv = iv[np.argsort(iv[:, 0])]
    reach = np.maximum.accumulate(iv[:, 1])        # furthest end among intervals starting earlier
    k = np.searchsorted(iv[:, 0], g, side="left") - 1  # last interval with lo < g
    return (k >= 0) & (g < reach[np.maximum(k, 0)])


def err_rms(t, y, ref: Reference, *, abs_scale: float = 0.0, exclude_t0: bool = False,
            exclude=()) -> float:
    """``sqrt(∫(y − ref)² dt / window) / span``, span = max(ref) − min(ref)
    floored at ``abs_scale`` (for a flat reference).

    ``exclude``: ``(lo, hi)`` time intervals left out of the integral and the
    window (the edge DIAGNOSTIC of the differential runner, never its gate)."""
    ref = _ref_window(t, ref, exclude_t0)
    t, y = _window(t, y, exclude_t0)
    g, a, r = _on_grid(t, y, ref)
    span = max(float(r.max() - r.min()), abs_scale)
    if span <= 0.0:
        raise ValueError("flat reference: give abs_scale > 0")
    return _rms(g, (a - r) ** 2, exclude) / span


def _rms(g, d2, exclude):
    """sqrt of the time mean of ``d2`` on ``g``, ``exclude`` intervals left out."""
    if not len(exclude):
        return float(np.sqrt(np.trapezoid(d2, g) / (g[-1] - g[0])))
    seg = np.diff(g) * 0.5 * (d2[1:] + d2[:-1])
    keep = ~_inside(0.5 * (g[1:] + g[:-1]), exclude)
    window = float(np.diff(g)[keep].sum())
    if window <= 0.0:
        raise ValueError("every interval excluded")
    return float(np.sqrt(seg[keep].sum() / window))


def err_rms_symmetric(a: tuple, b: tuple, *, abs_scale: float = 0.0, exclude=()) -> float:
    """:func:`err_rms` with neither trace the reference, for a DIAGNOSTIC:
    ``err_rms_symmetric(a, b) == err_rms_symmetric(b, a)`` bit for bit.

    ``a``, ``b`` are ``(t, y)`` traces. Both lose their initial sample(s) (the
    stored initial state of pwl / trbdf2, not the t = 0+ solution: kept on one
    side only, it was interpolated into the other's first step) and the span is
    the larger of the two traces' spans on the common window (floored at
    ``abs_scale``). ``err_rms`` keeps the reference's first sample and divides by
    the reference's span: on an ideal boost whose ``V(out)`` holds 24 V (6 mV of
    ripple) two converged traces read 0.030 one way and 0.0023 the other."""
    ta, ya = _window(*a, exclude_t0=True)
    tb, yb = _window(*b, exclude_t0=True)
    g, va, vb = _on_grid(ta, ya, (tb, yb))
    if len(g) < 2 or g[-1] <= g[0]:
        raise ValueError("the two traces share no time window")
    span = max(float(np.ptp(va)), float(np.ptp(vb)), abs_scale)
    if span <= 0.0:
        raise ValueError("flat traces: give abs_scale > 0")
    return _rms(g, (va - vb) ** 2, exclude) / span


def rule_ratio(t, y, ref: Reference, *, rtol: float, atol: float, exclude=()) -> float:
    """What the exact engine's output rule promises, checked: between events the
    linear interpolation of the stored samples is within ``max(rtol * range, atol)``
    of the solution (``simulate(output_rtol=)``). Returns the worst
    ``|interp(m) - ref(m)| / (rtol * range + atol)`` over the trace's interval
    midpoints ``m`` (``range``: the reference's range over the run, at least 1e-3 of
    its peak; a sum, not a max, of the two tolerances, so the reference's own error
    near atol is absorbed). Intervals of zero length or overlapping an ``exclude``
    interval (a recorded switch-state change: the rule promises nothing across an
    event) are skipped. <= 1 is within the promise; nan when no interval is left."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.diff(t) > 0
    lo, hi = t[:-1][keep], t[1:][keep]
    m = 0.5 * (lo + hi)
    ym = 0.5 * (y[:-1][keep] + y[1:][keep])
    if len(exclude):
        iv = np.asarray(exclude, dtype=float).reshape(-1, 2)
        bad = np.zeros(m.shape, bool)
        for a, b in iv:
            bad |= (lo < b) & (hi > a)
        m, ym = m[~bad], ym[~bad]
    if callable(ref):
        rm = np.asarray(ref(m), dtype=float)
        rr = np.asarray(ref(t), dtype=float)
    else:
        tr, yr = (np.asarray(a, dtype=float) for a in ref)
        inside = (m >= tr[0]) & (m <= tr[-1])
        m, ym = m[inside], ym[inside]
        rm, rr = np.interp(m, tr, yr), yr
    if not m.size:
        return float("nan")
    rng = max(float(np.ptp(rr)), 1e-3 * float(np.abs(rr).max()))
    return float(np.abs(ym - rm).max() / (rtol * rng + atol))


def exact_flow(td, yd, events=()):
    """A callable reading a DENSE trace of the exact flow (the exact engine run at a
    much smaller ``output_rtol``, the same events) at any instant: the quadratic
    through the three dense samples nearest it inside its own inter-event segment
    (``events``; a jump's right-limit row, at the event or one ulp after it, opens
    the next segment), linear when the segment holds two. The rule's
    promise is about the exact trajectory: a reference with its own chord error
    (trbdf2 at rtol 1e-8, read linearly between its steps) put 48 of the 54
    cells past 1 that sit at 0.96..0.9997 against this flow (group F, session 4)."""
    td = np.asarray(td, dtype=float)
    yd = np.asarray(yd, dtype=float)
    ev = np.sort(np.asarray(events, dtype=float))
    seg = np.searchsorted(ev, td, side="left")   # ev[k-1] < t <= ev[k]: segment k
    dup = np.zeros(td.size, bool)               # a jump's right limit (pulsim.with_right_limits
    dup[1:] = ((td[1:] == td[:-1])               # puts it at nextafter(t)) opens the next segment
               | ((td[1:] == np.nextafter(td[:-1], np.inf)) & np.isin(td[:-1], ev)))
    seg[dup & np.isin(td, ev)] += 1
    last = np.ones(td.size, bool)
    last[:-1] = ~dup[1:]                         # an event sample with no right-limit row

    def f(m):
        m = np.atleast_1d(np.asarray(m, dtype=float))
        out = np.empty_like(m)
        for q, x in enumerate(m):
            k = int(np.searchsorted(ev, x, side="left"))
            j = int(np.searchsorted(td, x))
            near = [i for i in range(max(j - 4, 0), min(j + 4, td.size))
                    if seg[i] == k or (k > 0 and td[i] == ev[k - 1] and last[i])]
            near = sorted(sorted(near, key=lambda i: abs(td[i] - x))[:3])
            ts, ys = td[near], yd[near]
            if len(near) < 3:
                out[q] = np.interp(x, ts, ys)
                continue
            l0 = (x - ts[1]) * (x - ts[2]) / ((ts[0] - ts[1]) * (ts[0] - ts[2]))
            l1 = (x - ts[0]) * (x - ts[2]) / ((ts[1] - ts[0]) * (ts[1] - ts[2]))
            l2 = (x - ts[0]) * (x - ts[1]) / ((ts[2] - ts[0]) * (ts[2] - ts[1]))
            out[q] = l0 * ys[0] + l1 * ys[1] + l2 * ys[2]
        return out
    f.samples = (td, yd)
    return f


def err_max(t, y, ref: Reference, *, abs_scale: float = 0.0, exclude_t0: bool = False,
            exclude=()) -> float:
    """``max|y − ref| / span`` at the same points as :func:`err_rms` (grid
    points strictly inside an ``exclude`` interval are skipped)."""
    ref = _ref_window(t, ref, exclude_t0)
    t, y = _window(t, y, exclude_t0)
    g, a, r = _on_grid(t, y, ref)
    span = max(float(r.max() - r.min()), abs_scale)
    if span <= 0.0:
        raise ValueError("flat reference: give abs_scale > 0")
    d = np.abs(a - r)
    if len(exclude):
        d = d[~_inside(g, exclude)]
    return float(d.max() / span)



def exact_dev(y, cand) -> np.ndarray:
    """|y_i − nearest candidate truth value at t_i| (``cand``: one row per
    sample, NaN-padded; see ``exact_truth.Solver.at_float``). A non-finite
    sample is an infinite deviation."""
    y = np.asarray(y, dtype=float)
    d = np.abs(np.asarray(cand, dtype=float) - y[:, None])
    e = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
    return np.where(np.isfinite(y), e, np.inf)


def err_exact(t, y, cand, *, span: float, gap=None, exclude_t0: bool = True) -> float:
    """The ``exact`` measure: ``sqrt((Σ w_i e_i² + Σ c_i) / Σ w_i) / span``
    over the trace's OWN samples, e_i = :func:`exact_dev`, w_i the trapezoid
    time weight of sample i (half of each adjacent interval). No
    interpolation of the trace, so a perfect trace scores the truth's
    evaluation error on any grid (no floor); what a trace does between its
    samples is not graded.

    ``gap`` (from ``at_float(..., gaps=True)``): a sample that matches a
    truth branch across an event (gap g_i > 0, the event displaced by g_i)
    is charged the L2 of that displacement, c_i = (jump at t_i)² g_i, so the
    snap window hides no timing error (two samples inside one displaced
    interval are both charged: conservative). Gaps within 4 ulp of t_i are
    not charged: a float time cannot place an event closer."""
    t = np.asarray(t, dtype=float)
    cand = np.asarray(cand, dtype=float)
    if np.any(np.diff(t) < 0):
        raise ValueError("times must be non-decreasing")
    e = exact_dev(y, cand)
    c = np.zeros(t.size)
    if gap is not None:
        rows = np.arange(t.size)
        d = np.abs(cand - np.asarray(y, dtype=float)[:, None])
        k = np.nanargmin(np.where(np.isnan(d), np.inf, d), axis=1)
        g = np.where(np.isnan(gap), np.inf, gap)
        g = np.where(g <= 4 * np.spacing(np.abs(t))[:, None], 0.0, g)   # below the float time's own resolution
        # the segment holding t_i (several at an exact event time: the one nearest the match)
        near = np.abs(cand - cand[rows, k][:, None])
        kc = np.argmin(np.where(g == g.min(axis=1, keepdims=True), np.nan_to_num(near, nan=np.inf), np.inf), axis=1)
        jump = cand[rows, kc] - cand[rows, k]
        c = np.where((k != kc) & np.isfinite(e), np.nan_to_num(jump * jump * g[rows, k], posinf=np.inf), 0.0)
    if exclude_t0:
        k0 = int(np.searchsorted(t, t[0], side="right"))
        t, e, c = t[k0:], e[k0:], c[k0:]
    if t.size < 2 or span <= 0.0:
        raise ValueError("need >= 2 graded samples and span > 0")
    if not np.all(np.isfinite(e)):
        return float("inf")
    h = np.diff(t)
    w = np.zeros(t.size)
    w[:-1] += h / 2
    w[1:] += h / 2
    return float(np.sqrt((np.sum(w * e * e) + np.sum(c)) / w.sum()) / span)
