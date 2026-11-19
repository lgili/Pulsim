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


def _window(t, y, exclude_t0: bool):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.shape != y.shape or t.size < 2:
        raise ValueError(f"need matching traces of >= 2 samples, got {t.shape} / {y.shape}")
    if exclude_t0:
        # The first stored sample of an algebraic node on pwl / trbdf2 is the
        # initial state, not the t = 0+ solution (design.md, Context).
        k = int(np.searchsorted(t, t[0], side="right"))
        t, y = t[k:], y[k:]
    return t, y


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
    t, y = _window(t, y, exclude_t0)
    g, a, r = _on_grid(t, y, ref)
    span = max(float(r.max() - r.min()), abs_scale)
    if span <= 0.0:
        raise ValueError("flat reference: give abs_scale > 0")
    d2 = (a - r) ** 2
    if not len(exclude):
        return float(np.sqrt(np.trapezoid(d2, g) / (g[-1] - g[0])) / span)
    seg = np.diff(g) * 0.5 * (d2[1:] + d2[:-1])
    keep = ~_inside(0.5 * (g[1:] + g[:-1]), exclude)
    window = float(np.diff(g)[keep].sum())
    if window <= 0.0:
        raise ValueError("every interval excluded")
    return float(np.sqrt(seg[keep].sum() / window) / span)


def err_max(t, y, ref: Reference, *, abs_scale: float = 0.0, exclude_t0: bool = False,
            exclude=()) -> float:
    """``max|y − ref| / span`` at the same points as :func:`err_rms` (grid
    points strictly inside an ``exclude`` interval are skipped)."""
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
    if gap is not None and np.all(np.isfinite(e)):
        rows = np.arange(t.size)
        d = np.abs(cand - np.asarray(y, dtype=float)[:, None])
        k = np.nanargmin(np.where(np.isnan(d), np.inf, d), axis=1)
        g = np.where(np.isnan(gap), np.inf, gap)
        g = np.where(g <= 4 * np.spacing(np.abs(t))[:, None], 0.0, g)   # below the float time's own resolution
        kc = np.argmin(g, axis=1)                   # the segment holding t_i
        jump = cand[rows, kc] - cand[rows, k]
        c = np.where(k != kc, np.nan_to_num(jump * jump * g[rows, k], posinf=np.inf), 0.0)
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
