"""Kernel probes and recorders on ``simulate(..., recorders=[...])``.

add-probes-and-recorders. A :class:`Recorder` names probes (expressions over
``v(node)``, ``v(a,b)``, ``i(device)`` with the operators and functions of
the circuit IR's parameter expressions) and what to keep: every accepted
sample, every n-th, or none; per declared window and/or per period the
time-weighted integral, average, RMS, minimum and maximum with their
times; and both limits of every probe at each event where the engine
stored a right limit.

FEEDING MODES (``Recording.feeding``, ``result.recorder_feed``):

* ``'in-loop'`` — pwl and trbdf2 through the kernel Simulation object:
  the recorders are fed from the engine's step hooks while it runs
  (``core/include/pulsim/record/loop_feed.hpp``), so they see every
  accepted step whatever ``store_every`` keeps. Readings must be functions
  of the state vector alone: ``v()``, ``i()`` of a state column or a
  resistor, ``p()`` of those.
* ``'stored-arrays'`` — a kernel-side pass over the stored result
  (``pulsim._pulsim._record_pass``): the exact engine (its stored right
  limits are exact), routes that bypass the object, other readings. With
  ``store_every > 1`` such a run is refused (the arrays would alias).

The arithmetic is the kernel recorder's (``record/recorder.hpp``): windows
clipped exactly, each step the straight line from its start (the right
limit where the engine has one) to its end, exact integrals of the line
and of its square. Event rows carry both limits and a resolution:
``'exact'`` (exact engine), ``'settled'`` (trbdf2's settled post-event
state at the instant), ``'within_dt'`` (a bracket: left at ``event_t``,
right at ``event_t_right``, the end of the first step under the new mode).
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

from . import _pulsim  # type: ignore[attr-defined]
from ._refusal import UnsupportedError, unsupported

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
    stream
        A :class:`pulsim.NativeLiveStream` (task 8.3, the RingSink): every
        kept row is pushed into its ring as ``(t, one value per probe)``
        (attached with ``state_size = len(probes)``, the probe texts as
        channel names; the stream's own ``decimate`` applies). In the loop
        (pwl, trbdf2) the rows arrive while the engine runs; on the
        stored-array pass (the exact engine, the routes without the
        Simulation object) after it. ``keep='none'`` pushes nothing. Its
        stop / pause act on the run only through ``should_continue``. Each
        run re-attaches it (a new ring); a run whose loop feed had to fall
        back to the stored pass keeps the rows the loop pushed.
    """

    name: str
    probes: Sequence[str]
    keep: str = "every"
    every: int = 1
    windows: Sequence[tuple] = ()
    period: Optional[float] = None
    offset: float = 0.0
    times: Sequence[float] = ()
    stream: Any = None


def _ring_of(r: "Recorder"):
    """Attach r.stream for r's probes and return its native ring (or None)."""
    if r.stream is None:
        return None
    r.stream.attach(len(r.probes), names=list(r.probes))
    return r.stream.native_ring


_RESOLUTION = ("exact", "settled", "within_dt")


class Recording:
    """One recorder's output: named arrays.

    ``rec["v(out)"]`` / ``rec.values[...]``: kept samples at ``rec.times``.
    ``rec.event_t`` with ``rec.event_left[p]`` / ``rec.event_right[p]``:
    both limits at each event, ``rec.event_resolution`` ('exact' /
    'settled' / 'within_dt') and ``rec.event_t_right`` say how the right
    value was obtained. ``rec.windows[p]`` and
    ``rec.periods[p]``: dicts of arrays ``t0 t1 integral avg rms min t_min
    max t_max``, one entry per window / complete period.
    ``rec.feeding`` names the feeding mode: ``'in-loop'`` (fed from the
    engine's step hooks while it ran: pwl and trbdf2 through the kernel
    Simulation object) or ``'stored-arrays'`` (a pass over the stored
    result: the exact engine, routes without the object, readings that
    are not functions of the state alone).
    """

    def __init__(self, spec: Recorder, out: dict,
                 feeding: str = "stored-arrays"):
        names = list(spec.probes)
        self.spec = spec
        self.feeding = feeding
        self.probes = names
        self.n_samples = int(out["n_samples"])
        self.times = np.asarray(out["times"])
        self.interpolated = np.asarray(out["interpolated"], dtype=bool)
        self.values = {p: out["values"][:, j] for j, p in enumerate(names)}
        self.event_t = np.asarray(out["event_t"])
        # where / how each event row's right value was taken: the exact
        # engine's right limit ('exact'), trbdf2's settled post-event state
        # at the instant ('settled'), or pwl's bracket ('within_dt': the
        # right value is the end of the first step under the new mode, at
        # event_t_right, up to one dt after event_t)
        self.event_t_right = np.asarray(out.get("event_t_right",
                                                out["event_t"]))
        # one take, not a Python loop over the event rows
        self.event_resolution = np.asarray(_RESOLUTION, dtype=object)[
            np.asarray(out.get("event_resolution",
                               np.zeros(self.event_t.size, np.uint8)), dtype=np.intp)]
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
# Readings of these kinds exist only IN THE LOOP (item 1, DECISIONS 20):
# the committed companion current of a capacitor (KCL-consistent), a Newton
# law at the committed state, a VCVS's current column. A stored result has
# no such reading (a capacitor's would be the derivative of its voltage),
# so the stored-array pass refuses them by name.
_LOOP_ONLY = frozenset({
    "capacitor", "nonlinear_capacitor", "mosfet_level1", "igbt_level1",
    "nonlinear_diode", "lauritzen_diode", "vcvs", "saturable_inductor",
    "excess_loss"})
_LOOP_ONLY_WHY = (
    "is read at the committed step only when the recorders are fed in the "
    "loop (pwl / trbdf2 through the Simulation object); a stored result has "
    "no such reading")


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


def _check_device_atoms(builder, r: "Recorder", p: str, atoms,
                        engine=None) -> None:
    if builder is None or not hasattr(builder, "components"):
        return
    for a in atoms:
        if a[0] not in "ip":
            continue
        d = _descriptor(builder, a[2:-1])
        if d is None:
            continue
        kind = d.get("kind", "?")
        if kind in _LOOP_ONLY and engine in ("exact", "dsed"):
            raise unsupported(
                str(engine), "recorders",
                f"recorder {r.name!r}, probe {p!r}: {a} of a {kind} "
                f"{_LOOP_ONLY_WHY}; engine={engine!r} is fed from its "
                "stored arrays", device=a[2:-1], kind=kind)
        if a[0] == "p" and kind not in _P_KINDS and kind not in _LOOP_ONLY:
            raise ValueError(
                f"recorder {r.name!r}, probe {p!r}: {a} of a {kind} is not "
                "a two-terminal power this pass knows the sign of; write "
                "it as v(a,b)*i(device)")


def _refuse(kw: dict, recs: "list[Recorder]", t_start: float,
            t_end: float) -> None:
    from . import _pulsim
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
                r, p, atoms, kw.get("engine"))
        for k, (a, b) in enumerate(r.windows):
            # a window may end after t_end: it is reported open (NaN) and a
            # run resumed from final_snapshot continues it (task 0b.7)
            if not (t_start <= a < b and a < t_end):
                raise ValueError(
                    f"recorder {r.name!r}: windows[{k}] = ({a!r}, {b!r}) "
                    f"must satisfy t_start <= t0 < t1 and t0 < t_end "
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
    builder = getattr(res, "_builder", None)
    d = None if builder is None or kind not in "ip" else _descriptor(builder, arg)
    if d is not None and d.get("kind") in _LOOP_ONLY:
        raise unsupported("stored-arrays", "recorders",
                          f"reading {atom!r} of a {d.get('kind')} {_LOOP_ONLY_WHY}",
                          device=arg, kind=str(d.get("kind")))
    if kind == "i":
        return np.asarray(res.i(arg), dtype=float)
    if kind == "p":
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


def record_result(res, recs: "list[Recorder]", stream: bool = True) -> dict:
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
            [float(x) for x in np.asarray(r.times, dtype=float).reshape(-1)],
            _ring_of(r) if stream else None)
        out[r.name] = _checked(r, d, t[0], t[-1], "stored-arrays")
    return out


def _checked(r: "Recorder", d: dict, t0, t1, feeding: str) -> "Recording":
    if int(d["n_times_unreached"]):
        raise ValueError(
            f"recorder {r.name!r}: {int(d['n_times_unreached'])} of times "
            f"lie outside the record [{t0!r}, {t1!r}]")
    done = d["windows"]["complete"]
    bad = [k for k, (_, b) in enumerate(r.windows)
           if not done[k, 0] and b <= t1]
    for k, (_, b) in enumerate(r.windows):
        if not done[k, 0] and b > t1:
            # open: it ends after this record (continued by a resume); its
            # statistics so far are not a window's statistics
            for f in _FIELDS[2:]:
                d["windows"][f][k, :] = np.nan
    if bad:
        raise ValueError(
            f"recorder {r.name!r}: windows {bad} are not covered by the "
            f"record [{t0!r}, {t1!r}]")
    return Recording(r, d, feeding)


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


# ---- in-loop feeding (task 0b.6) ----------------------------------------

# Set by simulate_with_recorders for the inner simulate() call; the pwl and
# trbdf2 routes through the kernel Simulation object attach the feed.
_LOOP_PLAN: "ContextVar[Optional[_LoopPlan]]" = ContextVar(
    "pulsim_recorder_loop_plan", default=None)


class _LoopPlan:
    def __init__(self, recs, caller, kw):
        self.recs, self.caller, self.feed = recs, caller, None
        self.t_start = float(kw.get("t_start") or 0.0)
        self.t_end = kw.get("t_end")
        self.dt = kw.get("dt")
        self.why_not = "this run's route does not go through the kernel " \
                       "Simulation object"
        self.resume = kw.get("resume_from")
        self.sim = None       # the Simulation the feed is attached to
        self.n_steps0 = 0     # the feed's steps before this run (a resume)
        self.resumed = None   # what a resume did with the recorder state


def _atoms_of(recs) -> "list[str]":
    from . import _pulsim
    atoms: list[str] = []
    for r in recs:
        for p in r.probes:
            for a in _pulsim._probe_atoms(p):
                if a not in atoms:
                    atoms.append(a)
    return atoms


def _node_terms(builder, caller, name: str):
    alias = getattr(caller, "_resolve_alias", None)
    alias = alias(name) if alias is not None else None
    if alias is not None and alias[0] == "node":
        name = alias[1]
    idx = int(builder.node_id_of(name))
    return ([], []) if idx < 0 else ([idx], [1.0])


def _branch(builder, caller, name: str):
    alias = getattr(caller, "_resolve_alias", None)
    alias = alias(name) if alias is not None else None
    if alias is not None and alias[0] == "branch":
        name = alias[1]
    return int(builder.branch_index_of(name))


def _column(fn, b_id: int, graph):
    """``fn(b_id, graph)`` or None when the branch is another kind."""
    if fn is None:
        return None
    try:
        return int(fn(b_id, graph))
    except Exception:  # noqa: BLE001 - another kind
        return None


def _state_reading(builder, caller, atom: str):
    """``(cols, coef, div)`` or a product ``((..), (..))`` of the reading
    as a function of the state vector alone, or None. The same column /
    law as ``res.v`` / ``res.i`` read from a stored sample."""
    kind, arg = atom[0], atom[2:-1]
    try:
        if kind == "v":
            parts = arg.split(",")
            if len(parts) == 1:
                c, w = _node_terms(builder, caller, parts[0])
                return (c, w, 1.0)
            ca, wa = _node_terms(builder, caller, parts[0])
            cb, wb = _node_terms(builder, caller, parts[1])
            return (ca + cb, wa + [-x for x in wb], 1.0)
        b_id = _branch(builder, caller, arg)
        d = None
        for x in builder.components():
            if int(x.get("branch_id", -1)) == b_id:
                d = x
                break
        if d is None:
            return None
        i_lin = None
        # res.i's fast path, in its order: a state column of the branch
        for acc in ("branch_var_id_for_inductor", "branch_var_id_for_source",
                    "branch_var_id_for_switch"):
            k = _column(getattr(builder.pool, acc, None), b_id, builder.graph)
            if k is not None and k >= 0:
                i_lin = ([k], [1.0], 1.0)
                break
        f, to = (int(n) for n in d["nodes"][:2])
        v_lin = ([n for n in (f, to) if n >= 0],
                 [w for n, w in ((f, 1.0), (to, -1.0)) if n >= 0], 1.0)
        if i_lin is None and d.get("kind") == "resistor":
            i_lin = (v_lin[0], v_lin[1], float(d["params"]["R_ohms"]))
        if i_lin is None:
            if kind == "p" and d.get("kind") not in _P_KINDS | _LOOP_ONLY:
                return None
            # a device law under the committed mode (the kernel resolves it
            # from the pool's kind and refuses a kind it has no law for)
            val = float(d.get("params", {}).get("I", 0.0)) \
                if d.get("kind") == "current_source" else 0.0
            return ("dev", b_id, kind == "p", val, v_lin)
        if kind == "i":
            return i_lin
        if kind == "p" and d.get("kind") in _P_KINDS | _LOOP_ONLY:
            return (v_lin, i_lin)
    except Exception:  # noqa: BLE001 - unknown name: the stored pass names it
        return None
    return None


def _spec_signature(recs) -> str:
    """What a snapshot's recorder state must have been written by."""
    return repr([(r.name, list(r.probes), r.keep, int(r.every),
                  [(float(a), float(b)) for a, b in r.windows],
                  float(r.period or 0.0), float(r.offset),
                  [float(x) for x in np.asarray(r.times, dtype=float).reshape(-1)])
                 for r in recs])


def _resume_state(plan):
    """The snapshot's recorder state for these recorders, or None (and why)."""
    snap = plan.resume
    if snap is None:
        return None
    st = list(getattr(snap, "recorder_state", None) or [])
    if not st:
        plan.resumed = ("started at the resume instant t = "
                        f"{float(snap.t)!r}: the snapshot carries no recorder "
                        "state (an older pickle, or a run without in-loop "
                        "recorders); a window or period before it is not in "
                        "this record")
        return None
    if snap.recorder_spec != _spec_signature(plan.recs):
        raise unsupported(
            "any", "resume_from",
            "simulate(recorders=..., resume_from=...): the snapshot's recorder "
            f"state was written by other recorders ({snap.recorder_spec}); "
            "declare the same recorders to continue them")
    return st


def attach_loop_feed(sim, engine: str) -> None:
    """Called by simulate()'s Simulation routes (pwl, trbdf2) before the
    run: when recorders are declared and every reading is a function of the
    state, feed them from the run's step hooks."""
    plan = _LOOP_PLAN.get()
    if plan is None:
        return
    from . import _pulsim
    builder = sim.builder
    atoms = _atoms_of(plan.recs)
    specs = []
    for a in atoms:
        r = _state_reading(builder, plan.caller, a)
        if r is None:
            plan.why_not = (f"reading {a} is not a function of the state "
                            "vector alone (a switched or Newton device's law)")
            return
        if r[0] == "dev":
            _, b_id, power, val, (cv, wv, dv) = r
            specs.append((cv, wv, dv, False, [], [], 1.0, b_id, power, val))
        elif len(r) == 2:
            (ca, wa, da), (cb, wb, db) = r
            specs.append((ca, wa, da, True, cb, wb, db, -1, False, 0.0))
        else:
            specs.append((r[0], r[1], r[2], False, [], [], 1.0, -1, False, 0.0))
    feed = _pulsim._LoopFeed(atoms, specs, [
        (list(r.probes), r.keep, int(r.every),
         [(float(a), float(b)) for a, b in r.windows],
         float(r.period or 0.0), float(r.offset),
         [float(x) for x in np.asarray(r.times, dtype=float).reshape(-1)])
        for r in plan.recs])
    st = _resume_state(plan)
    if st is not None:
        feed.restore(st)        # before reserve(): restore resizes the windows
        plan.resumed = ("continued from the snapshot's recorder state at t = "
                        f"{float(plan.resume.t)!r}")
    plan.n_steps0 = int(feed.n_steps)
    if plan.t_end is not None:
        n = 0
        if engine == "pwl" and plan.dt:
            n = int(round((float(plan.t_end) - plan.t_start) / float(plan.dt))) + 1
        feed.reserve(n, float(plan.t_end))
    try:
        sim.set_loop_feed(feed)   # binds the circuit: a kind with no law raises
    except ValueError as e:
        plan.why_not = str(e)
        return
    plan.sim = sim
    for k, r in enumerate(plan.recs):
        ring = _ring_of(r)
        if ring is not None:
            feed.stream_to(k, ring)
    feed.arm(engine == "pwl")
    plan.feed = feed


def simulate_with_recorders(simulate: Callable[..., Any], kw: dict,
                            recorders) -> Any:
    recs = _normalise(recorders)
    t_end = kw.get("t_end")
    t_start = float(kw.get("t_start") or 0.0)
    if t_end is not None:
        _refuse(kw, recs, t_start, float(t_end))
    caller = getattr(kw.get("builder"), "builder", kw.get("builder"))
    plan = _LoopPlan(recs, caller, kw)
    token = _LOOP_PLAN.set(plan)
    try:
        res = _run_at_user_frame(simulate, kw)
    finally:
        _LOOP_PLAN.reset(token)
        if plan.sim is not None:
            # a later run on that object (after a throw too) feeds nothing
            plan.sim.clear_loop_feed()
    t = np.asarray(res.times, dtype=float)
    if t_end is None:  # a LoadedCircuit supplied the run window
        _refuse(kw, recs, float(t[0]), float(t[-1]))
    feed = plan.feed
    if feed is not None and feed.n_dropped > 0:
        # an event report the feed could not place: never a silent in-loop
        # record with a missing limit (the stored pass, or the refusal)
        plan.why_not = (f"{int(feed.n_dropped)} event report(s) of the engine "
                        "could not be placed in the record")
        feed = None
    if feed is not None and feed.n_steps > plan.n_steps0:   # fed by THIS run
        snap = getattr(res, "final_snapshot", None)
        if snap is not None:
            # task 0b.7: the record continues across simulate(resume_from=)
            try:
                # before finish(): no flush
                feed.save_to(snap, _spec_signature(recs))
            except RuntimeError as e:   # an event report pending at the end
                plan.resumed = f"final_snapshot without recorder state: {e}"
        feed.finish()
        res.recorder_feed = {
            "mode": "in-loop", "n_steps": int(feed.n_steps),
            "n_events": int(feed.n_events), "n_jumps": int(feed.n_jumps),
            "n_dropped": int(feed.n_dropped)}
        if plan.resumed is not None:
            res.recorder_feed["resume"] = plan.resumed
        res.recorded = {r.name: _checked(r, feed.result(k), t[0], t[-1],
                                         "in-loop")
                        for k, r in enumerate(recs)}
        return res
    if plan.resume is not None and list(
            getattr(plan.resume, "recorder_state", None) or []):
        raise unsupported(
            "any", "resume_from",
            "simulate(recorders=..., resume_from=...): the snapshot carries "
            "recorder state, but this run could not be fed in the loop "
            f"({plan.why_not}), so the record cannot be continued")
    se = kw.get("store_every")
    if se is not None and int(se) > 1:
        who = ", ".join(repr(r.name) for r in recs)
        why = plan.why_not if feed is None else \
            "the engine's step hook did not fire (the static or legacy path)"
        raise ValueError(
            f"simulate(recorders=[{who}], store_every={se}): the recorders "
            f"could not be fed in the loop ({why}), and the stored arrays "
            "of a decimated run would alias; record with store_every=1 and "
            f"Recorder(keep='decimate', every={se}) to keep fewer rows")
    why = plan.why_not if feed is None else "the step hook did not fire"
    # rows the loop already pushed stay the stream's (no restart, no repeat)
    pushed = plan.feed is not None and plan.feed.n_steps > plan.n_steps0
    try:
        res.recorded = record_result(res, recs, stream=not pushed)
    except UnsupportedError as e:
        if _LOOP_ONLY_WHY not in str(e):
            raise
        raise unsupported(e.engine, e.feature, f"{e}; this run was not fed in "
                          f"the loop: {why}", device=e.device, kind=e.kind) from None
    res.recorder_feed = {"mode": "stored-arrays", "why": why}
    if plan.resume is not None:
        res.recorder_feed["resume"] = (
            f"started at the resume instant t = {float(plan.resume.t)!r}: the "
            "snapshot carries no recorder state")
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


# ---- external devices' readings (add-external-device-abi 3.4) -------------

def attach_xdev_trace(sim):
    """Called by simulate()'s Simulation routes after the recorders: trace
    the external devices added with ``record=True``. Returns the trace or
    None (no such device: nothing is attached)."""
    if not _pulsim._XdevTrace.wanted(sim.builder):
        return None
    tr = _pulsim._XdevTrace()
    sim.set_xdev_trace(tr)
    return tr


def result_device_output(self, name: str, key: str):
    """``key`` of the external device ``name`` at every stored sample:
    ``'i(<pin>)'`` (the current INTO the device at that pin, its law's
    ``f`` plus the committed ``dq/dt``), ``'power'`` (the sum of pin voltage
    times pin current) or one of the model's named outputs (evaluated with
    the state committed at that step). The device must be added with
    ``record=True`` and the run must go through pwl or trbdf2."""
    from ._refusal import unsupported
    tr = getattr(self, "_xdev_trace", None)
    if tr is None:
        raise unsupported(
            "", "device_output",
            f"device_output({name!r}, {key!r}): this result carries no "
            "external-device trace (add the device with record=True and run "
            "it on engine='pwl' or 'trbdf2')", device=name, kind="external_device")
    cols = list(tr.columns(name))
    if key not in cols:
        raise KeyError(f"device_output({name!r}, {key!r}): no such reading; "
                       f"the device has {cols}")
    c = cols.index(key)
    t = np.asarray(self.times, dtype=float)
    tt = np.asarray(tr.times, dtype=float)
    vals = np.asarray(tr.values(name))[:, c] if tt.size else np.empty(0)
    idx = np.searchsorted(tt, t)
    hit = np.zeros(t.size, dtype=bool)
    inside = idx < tt.size
    hit[inside] = tt[idx[inside]] == t[inside]
    out = np.empty(t.size)
    out[hit] = vals[idx[hit]]
    miss = np.flatnonzero(~hit)
    if miss.size and miss[0] == 0:   # the run's start: no step ends there
        x0 = np.asarray(self.states, dtype=float)[0]
        out[0] = tr.start_row(name, x0, float(t[0]))[c]
        miss = miss[1:]
    if miss.size:
        raise unsupported(
            "", "device_output",
            f"device_output({name!r}, {key!r}): {miss.size} stored sample(s) "
            f"(first at t = {t[miss[0]]!r}) are not accepted steps of the run",
            device=name, kind="external_device")
    return out
