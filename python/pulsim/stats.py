"""Time-weighted statistics for simulated waveforms.

A transient result is a list of ``(t_k, y_k)`` samples. With a fixed
``dt`` the samples are evenly spaced and ``y.mean()`` happens to equal
the time average. Without one (``engine='auto'`` with no ``dt``,
``engine='trbdf2'``, ``engine='dsed'``) the grid is VARIABLE-STEP: the
step controller packs samples around switching edges and diode events
-- tens of femtosecond-sized steps after a turn-off -- and stretches
them across the quiet stretches in between. A plain ``y.mean()``
weights every SAMPLE equally, so the few nanoseconds around an event
count as much as microseconds of steady conduction. Measured on an
L-D-RC rectifier whose true output average is 6.33 V, the sample mean
reads 3.04 V on ``engine='trbdf2'`` and 5.01 V on the default engine.

The helpers here weight each sample by the time it covers. They treat
the samples as a piecewise-linear waveform (straight lines between
consecutive samples, which is how every plot draws them) and integrate
THAT exactly:

* :func:`time_average` -- ``(1/T) * integral of y dt``
* :func:`time_rms` -- ``sqrt((1/T) * integral of y**2 dt)``
* :func:`time_window` -- the samples inside ``[t_from, t_to]``, with the
  two edges interpolated onto the window boundaries
* :func:`steady_state_stats` -- the above (plus min / max / ripple) for a
  node or branch of a :class:`~pulsim.SimulationResult`

On a uniform grid :func:`time_average` differs from ``y.mean()`` only
by the half-weight given to the two end samples, so scripts that pass
``dt`` see the same figures to ~4 significant digits.

Duplicate timestamps -- the engines record an event instant twice,
once on each side of the discontinuity -- are handled: the zero-width
interval between them contributes nothing, and the jump is kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

__all__ = [
    "TimeStats",
    "time_window",
    "time_average",
    "time_rms",
    "steady_state_stats",
]


def _as_trace(t, y) -> "tuple[np.ndarray, np.ndarray]":
    tt = np.asarray(t, dtype=float)
    yy = np.asarray(y, dtype=float)
    if tt.ndim != 1:
        raise ValueError(f"t must be 1-D, got shape {tt.shape}")
    if yy.shape != tt.shape:
        raise ValueError(
            f"t and y must have the same shape, got t{tt.shape} and "
            f"y{yy.shape}")
    if tt.size < 2:
        raise ValueError(
            f"need at least two samples to integrate over time, got "
            f"{tt.size}")
    if not np.all(np.isfinite(tt)):
        raise ValueError("t contains NaN or inf")
    if np.any(np.diff(tt) < 0.0):
        k = int(np.argmax(np.diff(tt) < 0.0))
        raise ValueError(
            f"t must be non-decreasing (duplicates are fine), but "
            f"t[{k + 1}] = {tt[k + 1]!r} < t[{k}] = {tt[k]!r}")
    return tt, yy


def _lerp(t0: float, y0: float, t1: float, y1: float, t: float) -> float:
    # Callers guarantee t0 < t1 strictly.
    return float(y0 + (y1 - y0) * ((t - t0) / (t1 - t0)))


def time_window(
    t,
    y,
    t_from: Optional[float] = None,
    t_to: Optional[float] = None,
) -> "tuple[np.ndarray, np.ndarray]":
    """Return the samples of ``y(t)`` inside ``[t_from, t_to]``.

    The window edges rarely fall on a sample, so the two end points
    are INTERPOLATED onto ``t_from`` and ``t_to`` (linearly, between
    the samples that straddle them). The returned arrays therefore
    start exactly at ``t_from`` and end exactly at ``t_to``, and
    integrating them covers the window and nothing else.

    Parameters
    ----------
    t, y
        Sample times (non-decreasing; repeated instants allowed) and
        values, same length.
    t_from, t_to
        Window bounds. ``None`` means the first / last sample time.

    Raises
    ------
    ValueError
        If the arrays are malformed, if the window has zero or
        negative width, or if it reaches outside ``[t[0], t[-1]]``
        (extrapolating a waveform is not averaging it).
    """
    tt, yy = _as_trace(t, y)
    a = float(tt[0]) if t_from is None else float(t_from)
    b = float(tt[-1]) if t_to is None else float(t_to)
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError(f"window bounds must be finite, got [{a}, {b}]")
    if not b > a:
        raise ValueError(
            f"empty time window: t_from={a!r} must be < t_to={b!r}")
    if a < tt[0] or b > tt[-1]:
        raise ValueError(
            f"window [{a!r}, {b!r}] reaches outside the simulated span "
            f"[{float(tt[0])!r}, {float(tt[-1])!r}]")
    # First sample strictly after `a`: [i-1, i] straddles `a` with
    # t[i-1] <= a < t[i], so the interpolation never divides by zero,
    # even on a repeated timestamp.
    i = int(np.searchsorted(tt, a, side="right"))
    # First sample at or after `b`: t[j-1] < b <= t[j].
    j = int(np.searchsorted(tt, b, side="left"))
    ya = _lerp(tt[i - 1], yy[i - 1], tt[i], yy[i], a)
    yb = _lerp(tt[j - 1], yy[j - 1], tt[j], yy[j], b)
    tw = np.concatenate(([a], tt[i:j], [b]))
    yw = np.concatenate(([ya], yy[i:j], [yb]))
    return tw, yw


def _check_finite(y: np.ndarray, what: str) -> None:
    if not np.all(np.isfinite(y)):
        raise ValueError(f"{what}: the waveform has NaN or inf inside "
                         f"the window")


def time_average(
    t,
    y,
    t_from: Optional[float] = None,
    t_to: Optional[float] = None,
) -> float:
    """Time-weighted average of ``y(t)`` over ``[t_from, t_to]``.

    ``(1/T) * integral of y dt`` for the piecewise-linear waveform
    through the samples. Use this, not ``y.mean()``, on any result
    produced without an explicit ``dt``: a variable-step grid clusters
    samples around events and ``y.mean()`` over-weights them (see the
    module docstring for a measured -52 % case).

    Parameters
    ----------
    t, y
        Sample times and values (e.g. ``result.times`` and
        ``result.v("out")``).
    t_from, t_to
        Averaging window, default the whole record. For a periodic
        steady state, average over a whole number of periods.

    Raises
    ------
    ValueError
        On malformed input, an empty window, a window outside the
        record, or non-finite values inside the window.
    """
    tw, yw = time_window(t, y, t_from, t_to)
    _check_finite(yw, "time_average")
    span = float(tw[-1] - tw[0])
    return float(np.sum(0.5 * (yw[1:] + yw[:-1]) * np.diff(tw)) / span)


def time_rms(
    t,
    y,
    t_from: Optional[float] = None,
    t_to: Optional[float] = None,
) -> float:
    """Time-weighted RMS of ``y(t)`` over ``[t_from, t_to]``.

    ``sqrt((1/T) * integral of y**2 dt)``, integrated EXACTLY for the
    piecewise-linear waveform through the samples (per interval,
    ``h * (y0**2 + y0*y1 + y1**2) / 3``) -- the same waveform
    :func:`time_average` integrates, so a ramp or a triangle comes
    out exact rather than with the trapezoid rule's overshoot on
    ``y**2``.

    Same arguments and errors as :func:`time_average`.
    """
    tw, yw = time_window(t, y, t_from, t_to)
    _check_finite(yw, "time_rms")
    y0, y1 = yw[:-1], yw[1:]
    span = float(tw[-1] - tw[0])
    ms = np.sum((y0 * y0 + y0 * y1 + y1 * y1) * np.diff(tw)) / (3.0 * span)
    return float(np.sqrt(max(float(ms), 0.0)))


@dataclass(frozen=True)
class TimeStats:
    """Time-weighted statistics of one waveform over one window.

    ``mean`` and ``rms`` are time-weighted (see :func:`time_average`,
    :func:`time_rms`). ``min``, ``max`` and ``ripple_pp``
    (``max - min``) are taken over the samples in the window -- they
    need no weighting, but on a variable-step grid they DO include
    every event-landing sample, so a sub-nanosecond numerical
    excursion there shows up in them.
    """

    mean: float
    rms: float
    min: float
    max: float
    ripple_pp: float
    t_from: float
    t_to: float


def _resolve_signal(result, what, kind: Optional[str]) -> np.ndarray:
    if not isinstance(what, str):
        return np.asarray(what, dtype=float)
    if kind == "v":
        return np.asarray(result.v(what), dtype=float)
    if kind == "i":
        return np.asarray(result.i(what), dtype=float)
    if kind is not None:
        raise ValueError(f"kind must be 'v', 'i' or None, got {kind!r}")
    found = {}
    errors = []
    for k, getter in (("v", result.v), ("i", result.i)):
        try:
            found[k] = np.asarray(getter(what), dtype=float)
        except KeyError as exc:  # NameNotFoundError subclasses KeyError
            errors.append(str(exc))
    if len(found) == 2:
        raise ValueError(
            f"{what!r} names both a node and a branch; pass kind='v' "
            f"(node voltage) or kind='i' (branch current)")
    if not found:
        raise KeyError(
            f"{what!r} is neither a node nor a branch of this result "
            f"({'; '.join(errors)})")
    return next(iter(found.values()))


def steady_state_stats(
    result,
    what: Union[str, "np.ndarray"],
    *,
    t_from: Optional[float] = None,
    t_to: Optional[float] = None,
    last: Optional[float] = None,
    kind: Optional[str] = None,
) -> TimeStats:
    """Time-weighted mean / RMS / min / max / ripple of a result trace.

    The steady-state figures a converter script prints, computed
    correctly on any grid::

        res = pulsim.simulate(b, t_end=20e-3)          # variable step
        s = pulsim.steady_state_stats(res, "out", last=10 / f_sw)
        print(s.mean, s.ripple_pp)

    This does NOT find the steady state (see
    :func:`pulsim.steady_state` for periodic shooting); it summarises
    a window you name. Choose one that starts after the start-up
    transient and spans a whole number of switching (or line) periods
    -- a fractional period biases ``mean`` by up to half a ripple.

    Parameters
    ----------
    result
        A :class:`~pulsim.SimulationResult` from :func:`pulsim.simulate`.
    what
        A node name (voltage, via ``result.v``), a branch name (current,
        via ``result.i``), or an array sampled on ``result.times``.
        Names are looked up as a node and as a branch; a name that is
        both raises, asking for ``kind``.
    t_from, t_to
        Window bounds in seconds. ``t_to`` defaults to the end of the
        record.
    last
        Alternative to ``t_from``: average over the final ``last``
        seconds (``t_from = t_to - last``).
    kind
        ``'v'`` or ``'i'`` to skip the node/branch lookup.

    Raises
    ------
    ValueError
        If neither or both of ``t_from`` / ``last`` are given (a
        steady-state figure over the whole record, start-up included,
        is the mistake this helper exists to prevent), if the window is
        empty or outside the record, or if an array ``what`` is not
        sampled on ``result.times``.
    KeyError
        If a name is neither a node nor a branch.
    """
    t = np.asarray(result.times, dtype=float)
    if t.size < 2:
        raise ValueError(
            f"the result has {t.size} sample(s); need at least two")
    if (t_from is None) == (last is None):
        raise ValueError(
            "steady_state_stats needs exactly one of t_from= or last= "
            "to say where the steady state starts; averaging from t=0 "
            "would include the start-up transient")
    b = float(t[-1]) if t_to is None else float(t_to)
    if last is not None:
        if not float(last) > 0.0:
            raise ValueError(f"last must be > 0, got {last!r}")
        a = b - float(last)
    else:
        a = float(t_from)  # type: ignore[arg-type]
    y = _resolve_signal(result, what, kind)
    if y.shape != t.shape:
        raise ValueError(
            f"the trace has shape {y.shape} but result.times has "
            f"{t.shape}; for a signal on its own time base use "
            f"pulsim.time_average(t, y, ...) directly")
    tw, yw = time_window(t, y, a, b)
    _check_finite(yw, "steady_state_stats")
    lo, hi = float(np.min(yw)), float(np.max(yw))
    return TimeStats(
        mean=time_average(tw, yw),
        rms=time_rms(tw, yw),
        min=lo,
        max=hi,
        ripple_pp=hi - lo,
        t_from=a,
        t_to=b,
    )
