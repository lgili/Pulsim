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
        sample), ``'none'`` (statistics and events only) or ``'times'``
        (one row at each of ``times``, the t_eval policy: linear on the
        step that holds it, the right limit at an event instant;
        ``Recording.interpolated`` flags the rows that are not samples).
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
    times: Sequence[float] = ()


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
        self.interpolated = np.asarray(out["interpolated"], dtype=bool)
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


# Kinds whose i() is exact on a stored sample (a state column, or the
# device law under the stored switch state), and whose current runs from
# the add_* call's first terminal to its second THROUGH the device, so
# p() = v(from, to) * i() is the power it absorbs. Verified by
# test_recorders.py (a source's p() is minus its loads').
_P_KINDS = frozenset({
    "resistor", "inductor", "diode", "switch", "shockley_diode",
    "voltage_source", "pwm_voltage_source", "sine_voltage_source",
    "pulse_voltage_source"})
# i() on a stored result is not exact (or not available) for these kinds:
# refused before the run, with the reason.
_I_REFUSED = {
    "capacitor": "its current is the derivative of the stored voltage, not "
                 "the companion law (add-probes-and-recorders section 6); "
                 "put a 1 mOhm series resistor and read i() of it",
    "nonlinear_capacitor": "no per-sample current law on a stored result",
    "mosfet_level1": "no per-sample current law on a stored result",
    "igbt_level1": "no per-sample current law on a stored result",
    "nonlinear_diode": "no per-sample current law on a stored result",
    "lauritzen_diode": "no per-sample current law on a stored result",
    "vcvs": "no per-sample current law on a stored result",
    "saturable_inductor": "no per-sample current law on a stored result",
}


def _descriptor(builder, name: str):
    """The components() entry of device ``name``, or None."""
    try:
        alias = builder._resolve_alias(name) if hasattr(
            builder, "_resolve_alias") else None
        if alias is not None and alias[0] == "branch":
            name = alias[1]
        b_id = int(builder.branch_index_of(name))
        for d in builder.components():
            if int(d.get("branch_id", -1)) == b_id:
                return d
    except Exception:  # noqa: BLE001 - unknown name: res.i names it
        return None
    return None


def _check_device_atoms(builder, r: "Recorder", p: str, atoms) -> None:
    if builder is None or not hasattr(builder, "components"):
        return
    for a in atoms:
        if a[0] not in "ip":
            continue
        d = _descriptor(builder, a[2:-1])
        if d is None:
            continue
        kind = d.get("kind", "?")
        if kind in _I_REFUSED:
            raise ValueError(
                f"recorder {r.name!r}, probe {p!r}: {a} of a {kind}: "
                f"{_I_REFUSED[kind]}")
        if a[0] == "p" and kind not in _P_KINDS:
            raise ValueError(
                f"recorder {r.name!r}, probe {p!r}: {a} of a {kind} is not "
                "a two-terminal power this pass knows the sign of; write "
                "it as v(a,b)*i(device)")


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
        if r.keep not in ("every", "decimate", "none", "times"):
            raise ValueError(
                f"recorder {r.name!r}: keep must be 'every', 'decimate', "
                f"'none' or 'times', got {r.keep!r}")
        ts = np.asarray(r.times, dtype=float).reshape(-1)
        if (r.keep == "times") != (ts.size > 0):
            raise ValueError(
                f"recorder {r.name!r}: keep='times' and a non-empty times= "
                "go together")
        if ts.size and not (np.all(np.isfinite(ts)) and np.all(np.diff(ts) > 0)
                            and t_start <= ts[0] and ts[-1] <= t_end):
            raise ValueError(
                f"recorder {r.name!r}: times must be finite, strictly "
                f"increasing and inside [{t_start!r}, {t_end!r}]")
        if int(r.every) < 1:
            raise ValueError(f"recorder {r.name!r}: every must be >= 1")
        if not r.probes:
            raise ValueError(f"recorder {r.name!r}: no probes")
        for p in r.probes:
            try:
                atoms = _pulsim._probe_atoms(p)
            except ValueError as e:
                raise ValueError(f"recorder {r.name!r}: {e}") from None
            _check_device_atoms(
                getattr(kw.get("builder"), "builder", kw.get("builder")),
                r, p, atoms)
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
    if kind == "p":
        builder = getattr(res, "_builder", None)
        d = None if builder is None else _descriptor(builder, arg)
        if d is None or d.get("kind") not in _P_KINDS:
            raise ValueError(
                f"reading {atom!r}: power of a "
                f"{None if d is None else d.get('kind')!r} is not available; "
                "write it as v(a,b)*i(device)")
        from ._result_views import node_voltage_trace, states_as_array
        st = states_as_array(res)
        f, to = (int(n) for n in d["nodes"][:2])
        v = node_voltage_trace(st, f) - node_voltage_trace(st, to)
        return v * np.asarray(res.i(arg), dtype=float)
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
            np.ascontiguousarray(right),
            [float(x) for x in np.asarray(r.times, dtype=float).reshape(-1)])
        if int(d["n_times_unreached"]):
            raise ValueError(
                f"recorder {r.name!r}: {int(d['n_times_unreached'])} of times "
                f"lie outside the record [{t[0]!r}, {t[-1]!r}]")
        rec = Recording(r, d)
        bad = [k for k in range(len(r.windows))
               if not d["windows"]["complete"][k, 0]]
        if bad:
            raise ValueError(
                f"recorder {r.name!r}: windows {bad} are not covered by the "
                f"record [{t[0]!r}, {t[-1]!r}]")
        out[r.name] = rec
    return out


def _run_at_user_frame(simulate: Callable[..., Any], kw: dict) -> Any:
    """``simulate(**kw)`` with the engine's warnings attributed to the
    caller of ``simulate(..., recorders=...)``: the two extra frames of
    this route would otherwise move every ``stacklevel`` into pulsim.
    Warnings raised from user code (a switch_fn) keep their location."""
    import os
    import sys
    import warnings

    user = sys._getframe(3)   # this, simulate_with_recorders, simulate, user
    pkg = os.path.dirname(os.path.abspath(__file__))
    caught: list = []

    def emit():
        for w in caught:
            if os.path.abspath(w.filename).startswith(pkg + os.sep):
                warnings.warn_explicit(
                    w.message, w.category, user.f_code.co_filename,
                    user.f_lineno, module=user.f_globals.get("__name__"),
                    registry=user.f_globals.setdefault("__warningregistry__", {}))
            else:
                warnings.warn_explicit(w.message, w.category, w.filename,
                                       w.lineno)

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = simulate(**kw)
    except BaseException:
        # the run's own error wins over a warning the filters turn into one
        try:
            emit()
        except Exception:  # noqa: BLE001
            pass
        raise
    emit()
    return res


def simulate_with_recorders(simulate: Callable[..., Any], kw: dict,
                            recorders) -> Any:
    recs = _normalise(recorders)
    t_end = kw.get("t_end")
    t_start = float(kw.get("t_start") or 0.0)
    if t_end is not None:
        _refuse(kw, recs, t_start, float(t_end))
    res = _run_at_user_frame(simulate, kw)
    if t_end is None:  # a LoadedCircuit supplied the run window
        _refuse(kw, recs, float(res.times[0]), float(res.times[-1]))
    res.recorded = record_result(res, recs)
    return res


# ---- measures on the result (add-probes-and-recorders D3, section 4) ----

def _window(res, t0, t1):
    t = np.asarray(res.times, dtype=float)
    if t.size < 2:
        raise ValueError("result: fewer than two samples, nothing to measure")
    return (float(t[0]) if t0 is None else float(t0),
            float(t[-1]) if t1 is None else float(t1))


def _stat(res, probe: str, field: str, t0, t1) -> float:
    a, b = _window(res, t0, t1)
    rec = Recorder("_measure", [probe], keep="none", windows=[(a, b)])
    _refuse({}, [rec], *_window(res, None, None))
    return float(record_result(res, [rec])["_measure"].windows[probe][field][0])


def result_avg(self, probe: str, t0=None, t1=None) -> float:
    """Time average of ``probe`` over [t0, t1] (default: the whole run):
    the integral of the engines' dense output (linear between samples,
    from the right limit where one is stored) divided by ``t1 - t0``,
    windows clipped exactly. Correct on any grid, unlike ``.mean()``."""
    return _stat(self, probe, "avg", t0, t1)


def result_rms(self, probe: str, t0=None, t1=None) -> float:
    """RMS of ``probe`` over [t0, t1]: exact for the linear dense output
    (``h(a^2 + ab + b^2)/3`` per step)."""
    return _stat(self, probe, "rms", t0, t1)


def result_integral(self, probe: str, t0=None, t1=None) -> float:
    """Integral of ``probe`` over [t0, t1] (see :meth:`avg`)."""
    return _stat(self, probe, "integral", t0, t1)


def result_minmax(self, probe: str, t0=None, t1=None) -> "tuple[float, float]":
    """(min, max) of ``probe`` over [t0, t1], both limits of an event
    inside the window included."""
    return (_stat(self, probe, "min", t0, t1), _stat(self, probe, "max", t0, t1))


def result_at(self, probe: str, t):
    """``probe`` at time(s) ``t``: linear on the step that holds it, the
    right limit at an event instant where the engine stored one. A scalar
    for a scalar ``t``; an array for an increasing 1-D ``t``."""
    ts = np.asarray(t, dtype=float)
    rec = Recorder("_at", [probe], keep="times", times=tuple(ts.reshape(-1)))
    _refuse({}, [rec], *_window(self, None, None))
    v = record_result(self, [rec])["_at"].values[probe]
    return float(v[0]) if ts.ndim == 0 else v
