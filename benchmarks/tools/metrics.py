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


def err_rms(t, y, ref: Reference, *, abs_scale: float = 0.0, exclude_t0: bool = False) -> float:
    """``sqrt(∫(y − ref)² dt / window) / span``, span = max(ref) − min(ref)
    floored at ``abs_scale`` (for a flat reference)."""
    t, y = _window(t, y, exclude_t0)
    g, a, r = _on_grid(t, y, ref)
    span = max(float(r.max() - r.min()), abs_scale)
    if span <= 0.0:
        raise ValueError("flat reference: give abs_scale > 0")
    return float(np.sqrt(np.trapezoid((a - r) ** 2, g) / (g[-1] - g[0])) / span)


def err_max(t, y, ref: Reference, *, abs_scale: float = 0.0, exclude_t0: bool = False) -> float:
    """``max|y − ref| / span`` at the same points as :func:`err_rms`."""
    t, y = _window(t, y, exclude_t0)
    g, a, r = _on_grid(t, y, ref)
    span = max(float(r.max() - r.min()), abs_scale)
    if span <= 0.0:
        raise ValueError("flat reference: give abs_scale > 0")
    return float(np.abs(a - r).max() / span)
