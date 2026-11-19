"""Kernel probes and recorders on ``simulate(..., recorders=[...])``.

add-probes-and-recorders. A :class:`Recorder` names probes (expressions over
``v(node)``, ``v(a,b)``, ``i(device)`` with the operators and functions of
the circuit IR's parameter expressions) and what to keep: every accepted
sample, every n-th, or none; per declared window and/or per period the
time-weighted integral, average, RMS, minimum and maximum with their
times; and both limits of every probe at each event where the engine
stored a right limit.

FEEDING MODE (today): a kernel-side pass over the stored result arrays,
on every engine (``pulsim._pulsim._record_pass``). The arithmetic is the
kernel recorder's (``core/include/pulsim/record/recorder.hpp``): windows
clipped exactly, each step the straight line from its start (the right
limit where the engine stored one: ``engine='exact'``) to its end, exact
integrals of the line and of its square. pwl and trbdf2 store no right
limit, so their event rows are not reported (the record is continuous
there); the in-loop feeding of the unified stepper will replace this
pass without changing the numbers' definition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

__all__ = ["Recorder", "Recording"]

_FIELDS = ("t0", "t1", "integral", "avg", "rms", "min", "t_min", "max",
           "t_max")


@dataclass(frozen=True)
class Recorder:
    """What one recorder evaluates and keeps.

    Parameters
    ----------
    name
        Key of the recording in ``result.recorded``.
    probes
        Probe expressions, e.g. ``"v(dp) - v(dn)"``, ``"v(out)*i(L1)"``.
    keep
        ``'every'`` (each accepted sample), ``'decimate'`` (samples
        0, n, 2n, ... with ``every=n``; statistics still use every
        sample) or ``'none'`` (statistics and events only).
    windows
        ``[(t0, t1), ...]``: statistics over each window, clipped exactly.
    period, offset
        Statistics per period ``[offset + kT, offset + (k+1)T]``; only
        complete periods are reported.
    """

    name: str
    probes: Sequence[str]
    keep: str = "every"
    every: int = 1
    windows: Sequence[tuple] = ()
    period: Optional[float] = None
    offset: float = 0.0


class Recording:
    """One recorder's output: named arrays.

    ``rec["v(out)"]`` / ``rec.values[...]``: kept samples at ``rec.times``.
    ``rec.event_t`` with ``rec.event_left[p]`` / ``rec.event_right[p]``:
    both limits at each event (exact engine). ``rec.windows[p]`` and
    ``rec.periods[p]``: dicts of arrays ``t0 t1 integral avg rms min t_min
    max t_max``, one entry per window / complete period.
    ``rec.feeding`` names the feeding mode.
    """

    feeding = "stored-arrays"

    def __init__(self, spec: Recorder, out: dict):
        names = list(spec.probes)
        self.spec = spec
        self.probes = names
        self.n_samples = int(out["n_samples"])
        self.times = np.asarray(out["times"])
        self.values = {p: out["values"][:, j] for j, p in enumerate(names)}
        self.event_t = np.asarray(out["event_t"])
        self.event_left = {p: out["event_left"][:, j]
                           for j, p in enumerate(names)}
        self.event_right = {p: out["event_right"][:, j]
                            for j, p in enumerate(names)}
        w, pr = out["windows"], out["periods"]
        self.windows = {p: {f: w[f][:, j] for f in _FIELDS}
                        for j, p in enumerate(names)}
        ok = pr["complete"][:, 0] > 0 if pr["complete"].size else \
            np.zeros(0, dtype=bool)
        self.periods = {p: {f: pr[f][ok, j] for f in _FIELDS}
                        for j, p in enumerate(names)}

    def __getitem__(self, probe: str) -> np.ndarray:
        return self.values[probe]

    def __repr__(self) -> str:
        return (f"Recording({self.spec.name!r}, probes={self.probes}, "
                f"kept={self.times.size}, events={self.event_t.size}, "
                f"windows={len(self.spec.windows)}, feeding={self.feeding!r})")


def _normalise(recorders) -> "list[Recorder]":
    if isinstance(recorders, (Recorder, dict)):
        recorders = [recorders]
    out = []
    for r in recorders:
        if isinstance(r, dict):
            r = Recorder(**r)
        if not isinstance(r, Recorder):
            raise TypeError(
                f"recorders: expected pulsim.Recorder or dict, got "
                f"{type(r).__name__}")
        out.append(r)
    names = [r.name for r in out]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"recorders: duplicate names {sorted(dup)}")
    return out


def _refuse(kw: dict, recs: "list[Recorder]", t_start: float,
            t_end: float) -> None:
    from . import _pulsim
    who = ", ".join(repr(r.name) for r in recs)
    se = kw.get("store_every")
    if se is not None and int(se) > 1:
        raise ValueError(
            f"simulate(recorders=[{who}], store_every={se}): the recorders "
            "are fed from the stored arrays (stored-array mode) and need "
            "every accepted step; use Recorder(keep='decimate', "
            f"every={se}) to keep fewer rows")
    if kw.get("resume_from") is not None:
        raise ValueError(
            f"simulate(recorders=[{who}], resume_from=...): recorder state "
            "is not part of the snapshot, so a window or period spanning "
            "the resume would be cut; record the whole run")
    for r in recs:
        if r.keep not in ("every", "decimate", "none"):
            raise ValueError(
                f"recorder {r.name!r}: keep must be 'every', 'decimate' or "
                f"'none', got {r.keep!r}")
        if int(r.every) < 1:
            raise ValueError(f"recorder {r.name!r}: every must be >= 1")
        if not r.probes:
            raise ValueError(f"recorder {r.name!r}: no probes")
        for p in r.probes:
            try:
                atoms = _pulsim._probe_atoms(p)
            except ValueError as e:
                raise ValueError(f"recorder {r.name!r}: {e}") from None
            for a in atoms:
                if a.startswith("p("):
                    raise ValueError(
                        f"recorder {r.name!r}, probe {p!r}: {a} has no "
                        "per-sample trace on a stored result; write it as "
                        "v(a,b)*i(device)")
        for k, (a, b) in enumerate(r.windows):
            if not (t_start <= a < b <= t_end):
                raise ValueError(
                    f"recorder {r.name!r}: windows[{k}] = ({a!r}, {b!r}) "
                    f"must satisfy t_start <= t0 < t1 <= t_end "
                    f"([{t_start!r}, {t_end!r}])")
        if r.period is not None and not (np.isfinite(r.period)
                                         and r.period > 0):
            raise ValueError(
                f"recorder {r.name!r}: period must be > 0, got {r.period!r}")
        if not np.isfinite(r.offset):
            raise ValueError(
                f"recorder {r.name!r}: offset must be finite, got "
                f"{r.offset!r}")


def _reading(res, atom: str) -> np.ndarray:
    kind, arg = atom[0], atom[2:-1]
    if kind == "v":
        parts = arg.split(",")
        if len(parts) == 1:
            return np.asarray(res.v(parts[0]), dtype=float)
        if len(parts) == 2:
            return (np.asarray(res.v(parts[0]), dtype=float)
                    - np.asarray(res.v(parts[1]), dtype=float))
        raise ValueError(f"reading {atom!r}: v() takes one or two nodes")
    if kind == "i":
        return np.asarray(res.i(arg), dtype=float)
    raise ValueError(f"reading {atom!r} is not supported")


def record_result(res, recs: "list[Recorder]") -> dict:
    """Run the kernel recorders over a finished result."""
    from . import _pulsim
    from ._result_views import with_right_limits

    atoms: list[str] = []
    for r in recs:
        for p in r.probes:
            for a in _pulsim._probe_atoms(p):
                if a not in atoms:
                    atoms.append(a)
    t = np.asarray(res.times, dtype=float)
    idx = getattr(res, "right_limit_index", None)
    idx = np.zeros(0, dtype=np.int64) if idx is None else \
        np.asarray(idx, dtype=np.int64)
    if idx.size:
        # the exact engine's right limits: evaluate every reading on the
        # trace with the right-limit rows inserted, then split the rows
        if np.any(np.diff(idx) <= 0):
            raise ValueError("result.right_limit_index is not increasing")
        if getattr(res, "right_limit_states", None) is None:
            raise ValueError("result has right_limit_index but no "
                             "right_limit_states: the right limits are "
                             "missing")
        wr = with_right_limits(res)
        pos = idx + 1 + np.arange(idx.size)
        keep = np.ones(t.size + idx.size, dtype=bool)
        keep[pos] = False
        cols = [_reading(wr, a) for a in atoms]
        full = np.column_stack(cols) if cols else np.zeros((keep.size, 0))
        left, right = full[keep], full[pos]
    else:
        cols = [_reading(res, a) for a in atoms]
        left = np.column_stack(cols) if cols else np.zeros((t.size, 0))
        right = np.zeros((0, len(atoms)))
    out = {}
    for r in recs:
        d = _pulsim._record_pass(
            list(r.probes), r.keep, int(r.every),
            [(float(a), float(b)) for a, b in r.windows],
            float(r.period or 0.0), float(r.offset), atoms, t,
            np.ascontiguousarray(left), [int(k) for k in idx],
            np.ascontiguousarray(right))
        rec = Recording(r, d)
        bad = [k for k in range(len(r.windows))
               if not d["windows"]["complete"][k, 0]]
        if bad:
            raise ValueError(
                f"recorder {r.name!r}: windows {bad} are not covered by the "
                f"record [{t[0]!r}, {t[-1]!r}]")
        out[r.name] = rec
    return out


def simulate_with_recorders(simulate: Callable[..., Any], kw: dict,
                            recorders) -> Any:
    recs = _normalise(recorders)
    t_end = kw.get("t_end")
    t_start = float(kw.get("t_start") or 0.0)
    if t_end is not None:
        _refuse(kw, recs, t_start, float(t_end))
    res = simulate(**kw)
    if t_end is None:  # a LoadedCircuit supplied the run window
        _refuse(kw, recs, float(res.times[0]), float(res.times[-1]))
    res.recorded = record_result(res, recs)
    return res
