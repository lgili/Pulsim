"""Datasheet switching-energy tables E(v, i, Tj) — audit C.1.

Pulsim already annotated switching loss PSIM-style: one energy
curve ``E(I)`` measured at a reference bus voltage, scaled
linearly by the actual blocking voltage. That model has no
temperature axis at all, and switching energy is strongly
temperature-dependent.

Measured against a 600 V / 100 A Si IGBT's own datasheet, reading
the 25 °C / 300 V curve and asking for a 600 V bus at a 125 °C
junction:

    I (A)    what it gives    datasheet    error
      25        2.60 mJ        4.40 mJ    -40.9 %
     100       11.00 mJ       18.00 mJ    -38.9 %
     200       24.00 mJ       38.60 mJ    -37.8 %

Roughly **40 % of the switching loss missing**, and it separates
cleanly:

    voltage       linear 300 -> 600 V scaling      -5.2 %
    temperature   25 °C table read at 125 °C      -35.6 %

So the linear voltage scaling was a decent approximation all
along; what was missing was the junction temperature. This module
adds it as a real axis rather than a correction factor, which is
what PLECS does and what a datasheet's own tables are shaped
like.

A table is read with trilinear interpolation and **linear
extrapolation** past its edges — the same choice the existing 1-D
curve makes, and for the same reason: clamping would silently
report a converter running past its datasheet as if it were at
the last tabulated point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def _ascending(name: str, axis) -> np.ndarray:
    a = np.asarray(axis, dtype=float).ravel()
    if a.size == 0:
        raise ValueError(f"LossTable: {name} axis is empty")
    if a.size > 1 and not np.all(np.diff(a) > 0):
        raise ValueError(
            f"LossTable: {name} axis must be strictly ascending, got "
            f"{a.tolist()}. Interpolation and the extrapolation "
            "slopes both read it in order, so an unsorted axis "
            "silently returns energies from the wrong cells.")
    return a


def _interp_axis(q: np.ndarray, axis: np.ndarray):
    """Index + weight for linear interpolation, extrapolating
    linearly past both ends.

    Returns ``(lo, hi, w)`` with the value being
    ``(1 - w) * f[lo] + w * f[hi]``; ``w`` is allowed outside
    [0, 1], which is exactly what makes it extrapolate.
    """
    n = axis.size
    if n == 1:
        z = np.zeros_like(q, dtype=int)
        return z, z, np.zeros_like(q, dtype=float)
    lo = np.clip(np.searchsorted(axis, q, side="right") - 1, 0, n - 2)
    hi = lo + 1
    w = (q - axis[lo]) / (axis[hi] - axis[lo])
    return lo, hi, w


@dataclass(frozen=True)
class LossTable:
    """Switching energy on a rectilinear ``(v, i, Tj)`` grid.

    Parameters
    ----------
    v_axis, i_axis, tj_axis
        Strictly ascending grids: blocking voltage [V], switched
        current [A], junction temperature [°C]. Any of them may be
        a single point, which collapses that dimension (useful
        when a datasheet only characterises one voltage).
    energy
        Array of shape ``(len(v_axis), len(i_axis), len(tj_axis))``
        in joules.
    """

    v_axis: np.ndarray
    i_axis: np.ndarray
    tj_axis: np.ndarray
    energy: np.ndarray

    def __post_init__(self) -> None:
        v = _ascending("v", self.v_axis)
        i = _ascending("i", self.i_axis)
        tj = _ascending("tj", self.tj_axis)
        e = np.asarray(self.energy, dtype=float)
        want = (v.size, i.size, tj.size)
        if e.shape != want:
            raise ValueError(
                f"LossTable: energy has shape {e.shape}, but the axes "
                f"say it should be {want} (v x i x Tj). Datasheet "
                "tables are usually transcribed current-major, so "
                "check the axis order before the numbers.")
        if np.any(e < 0):
            raise ValueError(
                "LossTable: switching energy must be >= 0; got a "
                f"minimum of {float(e.min()):g}")
        if not np.all(np.isfinite(e)):
            raise ValueError("LossTable: energy has non-finite entries")
        object.__setattr__(self, "v_axis", v)
        object.__setattr__(self, "i_axis", i)
        object.__setattr__(self, "tj_axis", tj)
        object.__setattr__(self, "energy", e)

    # -- reading ----------------------------------------------------
    def __call__(self, v, i, tj) -> np.ndarray:
        """Energy [J] at each ``(v, i, tj)``, trilinearly.

        Broadcasting follows numpy's rules, so scalars mix freely
        with arrays. The result is clamped to ``>= 0``: linear
        extrapolation below the first tabulated current can run
        negative, and a negative switching energy is never a
        physical answer.
        """
        vq, iq, tq = np.broadcast_arrays(
            np.asarray(v, dtype=float),
            np.asarray(i, dtype=float),
            np.asarray(tj, dtype=float))
        shape = vq.shape
        vq, iq, tq = vq.ravel(), iq.ravel(), tq.ravel()

        v0, v1, wv = _interp_axis(vq, self.v_axis)
        i0, i1, wi = _interp_axis(iq, self.i_axis)
        t0, t1, wt = _interp_axis(tq, self.tj_axis)

        e = self.energy
        out = np.zeros_like(vq)
        for av, ww_v in ((v0, 1.0 - wv), (v1, wv)):
            for ai, ww_i in ((i0, 1.0 - wi), (i1, wi)):
                for at, ww_t in ((t0, 1.0 - wt), (t1, wt)):
                    out += e[av, ai, at] * ww_v * ww_i * ww_t
        return np.maximum(out, 0.0).reshape(shape)

    # -- construction ----------------------------------------------
    @classmethod
    def from_curves(cls, curves: Mapping[Any, Sequence],
                     *, i_axis=None) -> "LossTable":
        """Build from the shape a datasheet actually publishes:
        ``E(I)`` curves, one per ``(V, Tj)`` pair.

        ``curves`` maps ``(V, Tj)`` to a sequence of ``(I, E)``
        points. The current axis is the union of every curve's
        currents unless ``i_axis`` is given, and each curve is
        interpolated onto it — so the curves need not share
        sample points, which they rarely do.

        Every ``(V, Tj)`` combination in the cross product must be
        present. A missing corner is refused rather than filled,
        because filling it would invent a datasheet number.
        """
        if not curves:
            raise ValueError("LossTable.from_curves: no curves given")
        vs = sorted({float(k[0]) for k in curves})
        ts = sorted({float(k[1]) for k in curves})
        missing = [(v, t) for v in vs for t in ts
                   if not any(abs(float(k[0]) - v) < 1e-12
                              and abs(float(k[1]) - t) < 1e-12
                              for k in curves)]
        if missing:
            raise ValueError(
                "LossTable.from_curves: the grid is incomplete — no "
                f"curve for (V, Tj) = {missing}. Every combination of "
                f"V in {vs} and Tj in {ts} needs one. Filling a "
                "missing corner would invent a datasheet number; give "
                "the curve, or drop that V or Tj from the set.")
        if i_axis is None:
            allc: list[float] = []
            for pts in curves.values():
                allc.extend(float(p[0]) for p in pts)
            i_ax = np.unique(np.asarray(allc, dtype=float))
        else:
            i_ax = _ascending("i", i_axis)

        e = np.zeros((len(vs), len(i_ax), len(ts)), dtype=float)
        for key, pts in curves.items():
            v, t = float(key[0]), float(key[1])
            arr = np.asarray(sorted((float(a), float(b))
                                     for a, b in pts), dtype=float)
            if arr.size == 0:
                raise ValueError(
                    f"LossTable.from_curves: curve for (V={v}, "
                    f"Tj={t}) is empty")
            iv = vs.index(v)
            it = ts.index(t)
            if arr.shape[0] == 1:
                e[iv, :, it] = arr[0, 1]
            else:
                e[iv, :, it] = np.interp(i_ax, arr[:, 0], arr[:, 1])
                # Match __call__'s linear extrapolation rather than
                # np.interp's clamping, so the table is consistent
                # with how it will be read.
                below = i_ax < arr[0, 0]
                if below.any():
                    s = (arr[1, 1] - arr[0, 1]) / (arr[1, 0] - arr[0, 0])
                    e[iv, below, it] = arr[0, 1] + s * (i_ax[below]
                                                         - arr[0, 0])
                above = i_ax > arr[-1, 0]
                if above.any():
                    s = ((arr[-1, 1] - arr[-2, 1])
                         / (arr[-1, 0] - arr[-2, 0]))
                    e[iv, above, it] = arr[-1, 1] + s * (i_ax[above]
                                                          - arr[-1, 0])
        return cls(np.asarray(vs, dtype=float), i_ax,
                    np.asarray(ts, dtype=float),
                    np.maximum(e, 0.0))

    def __repr__(self) -> str:  # pragma: no cover — debug aid
        return (f"LossTable(v={self.v_axis.tolist()}, "
                f"i={len(self.i_axis)} pts, "
                f"Tj={self.tj_axis.tolist()})")


def as_loss_table(obj) -> LossTable:
    """Accept a `LossTable`, or a mapping in `from_curves` form."""
    if isinstance(obj, LossTable):
        return obj
    if isinstance(obj, Mapping):
        return LossTable.from_curves(obj)
    raise TypeError(
        "expected a LossTable or a {(V, Tj): [(I, E), ...]} mapping, "
        f"got {type(obj).__name__}")


def resolve_tj(tj, times: np.ndarray) -> np.ndarray:
    """Junction temperature as an array over `times`.

    Accepts a scalar (a fixed junction temperature), an array of
    the same length as `times` (from a thermal simulation), or a
    callable ``t -> Tj``. Anything else is refused by name — a
    silently broadcast wrong shape would put the whole run at one
    temperature and read plausibly.
    """
    if callable(tj):
        fn = tj
        return np.asarray([float(fn(float(t))) for t in times],
                          dtype=float)
    arr = np.asarray(tj, dtype=float)
    if arr.ndim == 0:
        return np.full(times.shape, float(arr))
    if arr.shape != times.shape:
        raise ValueError(
            f"Tj has shape {arr.shape} but the result has "
            f"{times.shape} time points. Give a scalar for a fixed "
            "junction temperature, an array on the result's own time "
            "grid, or a callable t -> Tj.")
    return arr
