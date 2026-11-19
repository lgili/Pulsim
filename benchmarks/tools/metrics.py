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
