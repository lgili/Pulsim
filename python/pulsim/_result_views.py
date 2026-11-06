"""Shared helpers for walking a `SimulationResult` post-hoc.

These small utilities exist because both :mod:`pulsim.losses` and
:mod:`pulsim.thermal` need to reconstruct per-device traces (node
voltages, switch-mask history) from the kernel's
:class:`SimulationResult` and the user's :class:`CircuitBuilder`.
Keeping the implementations here lets both modules share one
source of truth instead of drifting in parallel.

The helpers are intentionally minimal and dependency-light — no
pulsim internals, just NumPy on the result trace + the builder's
public Python surface.

These are package-private (single leading underscore on the module
name); third-party callers should use :func:`device_loss_summary`
or :func:`device_thermal_summary` instead.

Switch state, post-hoc vs. recorded
-----------------------------------
A *controlled* switch's on/off state is set by the user's ``switch_fn``
at simulate-time. Re-evaluating ``switch_fn(t)`` afterwards is only
faithful when ``switch_fn`` is **stateless** (open-loop PWM): a
closed-loop controller (PI / PFC / FOC) carries internal state that
evolved across the run, so calling it now returns the mask it *would*
emit today (converged duty), not the mask in effect at time ``t``.
Pairing that stale mask with the saved (historical) terminal voltage
produces an unphysical ``v² · g_on`` conduction spike during the
controller's settling window. :func:`resolve_switch_closed_trace`
handles this: it prefers an exact mask recorded at simulate-time
(:class:`SwitchMaskRecorder`) and otherwise re-evaluates ``switch_fn``
with a voltage-consistency guard that drops samples claiming ON while
the device is clearly blocking.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np


def states_as_array(result) -> np.ndarray:
    """Coerce ``SimulationResult.states`` (numpy array OR a list of
    state vectors) to a contiguous ``(N, state_size)`` float array.

    .. note::
       For a kernel ``SimulationResult`` this returns the READ-ONLY
       zero-copy view itself (v2.0) — deliberately, so walking a
       long run costs nothing. Callers that need to modify the data
       must ``.copy()`` first; the public per-signal accessors
       (``res.v`` / ``res.i``) already hand back owned copies."""
    s = result.states
    if hasattr(s, "shape"):
        return np.asarray(s, dtype=float)
    return np.asarray([list(v) for v in s], dtype=float)


def times_as_array(result) -> np.ndarray:
    """Coerce ``SimulationResult.times`` to a float ndarray."""
    return np.asarray(result.times, dtype=float)


def node_voltage_trace(states: np.ndarray, node_id: int) -> np.ndarray:
    """Return the voltage trace for ``node_id``. Ground (``node_id <
    0``) is treated as the reference and reported as a zero trace."""
    if node_id < 0:
        return np.zeros(states.shape[0], dtype=float)
    return states[:, node_id]


# ---------------------------------------------------------------
# The kernel's per-sample record of the STAMPED mask
# ---------------------------------------------------------------

def recorded_switch_states(result, builder=None):
    """``(states, reason)``: the ``(n_samples, n_switches)`` bool array
    of the switch / diode mask the solver stamped for each stored
    sample (``SimulationResult.switch_states``), or ``(None, reason)``
    when the result carries no usable record — ``reason`` says why,
    for the fallback's warning.

    The record is validated against the result (one row per sample)
    and, when ``builder`` is given, against the circuit (one column per
    switch-kind branch). An unsolved row (a sample 0 seeded from
    ``initial_state``) is returned with the mask of the first solve
    after it (:func:`_borrow_for_unsolved`)."""
    try:
        rec = getattr(result, "switch_states", None)
    except Exception as exc:  # noqa: BLE001 — a broken property
        return None, f"reading result.switch_states failed ({exc!r})"
    if rec is None:
        if hasattr(result, "states_reduced"):
            return None, ("result.switch_states is absent — this "
                          "engine='dsed' run went through the DSED "
                          "adapter path that records no mask (a "
                          "Python-mask switch_fn, or no full MNA "
                          "reconstruction)")
        return None, ("result.switch_states is absent — a result not "
                      "produced by a kernel engine, or one whose "
                      "engine path does not record it")
    arr = np.asarray(rec, dtype=bool)
    n = int(np.asarray(result.times).shape[0])
    if arr.ndim != 2 or arr.shape[0] != n:
        return None, (f"result.switch_states has shape {arr.shape}, "
                      f"not ({n}, n_switches)")
    if builder is not None:
        try:
            n_sw = int(builder.graph.num_switches)
        except Exception:  # noqa: BLE001 — a duck-typed builder
            n_sw = arr.shape[1]
        if arr.shape[1] != n_sw:
            return None, (f"result.switch_states has {arr.shape[1]} "
                          f"columns but the circuit has {n_sw} "
                          f"switch-kind branches")
    return _borrow_for_unsolved(arr, recorded_unsolved(result, n)), ""


def _borrow_for_unsolved(arr: np.ndarray, unsolved: np.ndarray) -> np.ndarray:
    """Read an UNSOLVED row (``switch_states_unsolved``: a sample 0
    seeded from ``initial_state``, whose recorded bits are the mask the
    solver carried in, not the one its state was solved under) with the
    mask of the first solve after it.

    Measured on an RL half-wave seeded mid-conduction: the carried row
    had the diode OFF while the seeded state carried 1.259 A through
    it, a 1.26 A KCL residual at that one sample (every other sample
    closes to 2e-12); the first solve has it ON, and reading row 0
    with it closes KCL there too. Continuity is the better guess for a
    single seeded sample: its state came out of a solve that the next
    step continues. The flag itself is kept, so the commutation-rule
    check still exempts the row."""
    if not unsolved.any():
        return arr
    out = arr.copy()
    solved = np.nonzero(~unsolved)[0]
    for i in np.nonzero(unsolved)[0]:
        j = solved[np.searchsorted(solved, i)] if (
            solved.size and solved[-1] > i) else None
        if j is not None:
            out[i] = arr[j]
    return out


def record_advice(result) -> str:
    """How to get a result that carries the stamped-mask record."""
    if hasattr(result, "states_reduced"):
        return ("run engine='pwl' or engine='auto' (both record the "
                "stamped mask for every sample), or the native DSED "
                "path, which records it when the circuit is built with "
                "CircuitBuilder and switch_fn returns SwitchStateMask.")
    return ("use a result returned by pulsim.simulate() with "
            "engine='pwl', 'auto' or 'trbdf2', which record the stamped "
            "mask for every sample.")


def recorded_unsolved(result, n: int) -> np.ndarray:
    """Per-sample bool: the sample was not produced by a solve under
    its recorded mask (``SimulationResult.switch_states_unsolved``)."""
    u = getattr(result, "switch_states_unsolved", None)
    if u is None:
        return np.zeros(n, dtype=bool)
    u = np.asarray(u, dtype=bool)
    return u if u.shape == (n,) else np.zeros(n, dtype=bool)


def switch_bits_by_branch(builder) -> "dict[int, int]":
    """``{branch_id: SwitchStateMask bit}`` for every switch-kind
    branch (controlled switches, PWL diodes, MOSFET body diodes — they
    interleave in builder order). ``switch_index_of(name)`` is the
    authority; an unnamed branch falls back to its position among the
    switch / diode components."""
    out: "dict[int, int]" = {}
    pos = 0
    for c in sorted(builder.components(),
                    key=lambda d: int(d.get("branch_id", -1))):
        if c.get("kind") not in ("switch", "diode"):
            continue
        bid = int(c["branch_id"])
        bit = pos
        name = c.get("name")
        if name:
            try:
                bit = int(builder.switch_index_of(name))
            except Exception:  # noqa: BLE001 — alias / unnamed
                bit = pos
        out[bid] = bit
        pos += 1
    return out


def evaluate_switch_mask_trace(
    switch_fn: Callable[[float], Any],
    times: np.ndarray,
    switch_idx: int,
) -> np.ndarray:
    """Sample ``switch_fn(t).get(switch_idx)`` at every result time
    and return a boolean ndarray of length ``len(times)``.

    .. warning::
       This re-evaluates ``switch_fn`` *post-hoc*. It is faithful only
       for **stateless** switch_fns (open-loop / fixed PWM). A
       closed-loop controller's state has converged by the time this
       runs, so the returned mask is its *final* state, not its state
       at ``t``. Prefer :func:`resolve_switch_closed_trace`, which
       handles the desync."""
    out = np.zeros(times.size, dtype=bool)
    for k, t in enumerate(times):
        mask = switch_fn(float(t))
        out[k] = bool(mask.get(int(switch_idx)))
    return out


class SwitchMaskRecorder:
    """Wrap a ``switch_fn`` to record the *actual* mask it returns at
    every simulate-time call, so post-hoc loss / thermal summaries can
    use the true historical switch state instead of re-evaluating a
    (possibly stateful, closed-loop) ``switch_fn`` whose controller has
    since converged.

    Pass the *same* recorder instance to both :func:`simulate` and the
    summary::

        sf  = pulsim.SwitchMaskRecorder(my_switch_fn)
        res = pulsim.simulate(b, ..., switch_fn=sf, step_observer=obs)
        summ = pulsim.device_thermal_summary(b, res, switch_fn=sf, ...)

    It is a drop-in callable: ``sf(t)`` delegates to the wrapped
    ``switch_fn`` and records ``(t, mask)``. The summaries detect the
    recorder via :meth:`recorded_mask_trace` and use the recorded
    history verbatim (no voltage-consistency guard needed).
    """

    def __init__(self, switch_fn: Callable[[float], Any]) -> None:
        self._fn = switch_fn
        self._times: list[float] = []
        self._masks: list[Any] = []

    def __call__(self, t: float) -> Any:
        mask = self._fn(float(t))
        self._times.append(float(t))
        self._masks.append(mask)
        return mask

    @property
    def n_records(self) -> int:
        return len(self._times)

    def recorded_mask_trace(self, times: np.ndarray,
                            switch_idx: int) -> np.ndarray:
        """Map each query time to the last mask recorded at-or-before it.

        Falls back to a live re-evaluation if nothing was recorded
        (e.g. the recorder was never passed to ``simulate``)."""
        if not self._times:
            return evaluate_switch_mask_trace(self._fn, times, switch_idx)
        rt = np.asarray(self._times, dtype=float)
        order = np.argsort(rt, kind="stable")
        rt_sorted = rt[order]
        q = np.asarray(times, dtype=float)
        # last recorded sample at-or-before each query time
        idx = np.searchsorted(rt_sorted, q, side="right") - 1
        idx = np.clip(idx, 0, rt_sorted.size - 1)
        out = np.zeros(q.size, dtype=bool)
        for k in range(q.size):
            out[k] = bool(self._masks[order[idx[k]]].get(int(switch_idx)))
        return out


def _voltage_consistency_guard(closed: np.ndarray, v_branch: np.ndarray,
                               *, name: Any = None,
                               frac: float = 0.05) -> np.ndarray:
    """Drop ``closed`` samples that claim a switch is ON while its saved
    terminal voltage sits in the blocking regime.

    A conducting switch drops only ``i / g_on`` — orders of magnitude
    below the voltage it blocks when off. So a sample tagged ON whose
    ``|v|`` exceeds ``frac`` of the device's own typical blocking level
    is physically impossible; it is the closed-loop post-hoc mask
    desync (stale ON mask paired with the historical off-state
    voltage). Those samples are forced OFF so ``P = v²·g`` no longer
    explodes. A no-op when the mask already agrees with the voltages
    (stateless / open-loop ``switch_fn``), so correct results are
    untouched."""
    closed = np.asarray(closed, dtype=bool)
    av = np.abs(np.asarray(v_branch, dtype=float))
    off = ~closed
    if not closed.any() or not off.any():
        return closed
    v_block = float(np.percentile(av[off], 95.0))
    if not np.isfinite(v_block) or v_block <= 0.0:
        return closed
    thresh = frac * v_block
    bad = closed & (av > thresh)
    n_bad = int(bad.sum())
    if n_bad:
        closed = closed.copy()
        closed[bad] = False
        warnings.warn(
            f"switch {name!r}: {n_bad} sample(s) reported ON by switch_fn "
            f"while the saved terminal voltage was in the blocking regime "
            f"(|v| > {thresh:.3g} V ≈ {frac:.0%} of the {v_block:.3g} V "
            f"blocking level). This is the closed-loop post-hoc switch-mask "
            f"desync — switch_fn was re-evaluated with its converged "
            f"controller state, not its state at time t. Those samples were "
            f"treated as OFF to avoid an unphysical conduction spike. For "
            f"exact masks, wrap switch_fn in pulsim.SwitchMaskRecorder and "
            f"pass it to both simulate() and this summary.",
            stacklevel=3,
        )
    return closed


def resolve_switch_closed_trace(result: Any,
                                switch_fn: Callable[[float], Any],
                                times: np.ndarray,
                                switch_idx: int,
                                v_branch: np.ndarray,
                                *, name: Any = None) -> np.ndarray:
    """Best available ON/OFF trace for a controlled switch.

    Preference order:

    1. The kernel's own record of the mask it STAMPED for each sample
       (``result.switch_states``) — exact on every engine, for
       stateless and closed-loop switch_fns alike, with no switch_fn
       call at all.
    2. A :class:`SwitchMaskRecorder` passed as ``switch_fn`` (exact
       simulate-time history) — or a ``result._switch_masks`` 2-D bool
       array attached by the caller.
    3. Re-evaluate ``switch_fn(t)`` (faithful for stateless switch_fns)
       then apply :func:`_voltage_consistency_guard`.
    """
    kernel, _ = recorded_switch_states(result)
    if kernel is not None and 0 <= switch_idx < kernel.shape[1]:
        return np.asarray(kernel[:, switch_idx], dtype=bool)
    if switch_fn is None:
        raise ValueError(
            f"switch {name!r}: the result carries no record of the "
            f"stamped switch mask and no switch_fn was given to "
            f"re-evaluate.")
    rec = getattr(switch_fn, "recorded_mask_trace", None)
    q = mask_sample_times(result, times)
    if callable(rec):
        return np.asarray(rec(q, switch_idx), dtype=bool)
    masks = getattr(result, "_switch_masks", None)
    if masks is not None:
        arr = np.asarray(masks)
        if (arr.ndim == 2 and arr.shape[0] == np.asarray(times).size
                and 0 <= switch_idx < arr.shape[1]):
            return arr[:, switch_idx].astype(bool)
    warnings.warn(
        f"switch {name!r}: the result carries no record of the stamped "
        f"switch mask ({recorded_switch_states(result)[1]}), so "
        f"switch_fn was re-evaluated after the run. That is exact only "
        f"for a stateless switch_fn; {record_advice(result)}",
        RuntimeWarning, stacklevel=3)
    closed = evaluate_switch_mask_trace(switch_fn, q, switch_idx)
    return _voltage_consistency_guard(closed, v_branch, name=name)


def mask_sample_times(result: Any, times: np.ndarray) -> np.ndarray:
    """The time at which the engine evaluated ``switch_fn`` for the
    step that PRODUCED each stored sample.

    * Fixed-step engine (``engine='pwl'``): the step's end, i.e. the
      sample time itself (``run_event_iteration(t, ...)`` calls
      ``switch_fn(t)``).
    * ``engine='trbdf2'`` / ``'auto'``: the step MIDPOINT
      (``trbdf2_transient.hpp``: ``mask_step = switch_fn(t + h/2)``).
      A step that lands on a gate edge is therefore stored with the
      PRE-edge mask; re-evaluating ``switch_fn`` at the sample time
      returned the post-edge state, which made a 50 kHz buck's switch
      read 0.455 duty instead of 0.500 and set off the closed-loop
      desync warning on a stateless PWM.
    """
    t = np.asarray(times, dtype=float)
    if getattr(result, "engine_used", None) != "trbdf2" or t.size < 2:
        return t
    q = t.copy()
    q[1:] = 0.5 * (t[:-1] + t[1:])
    return q


# ---------------------------------------------------------------
# v2.0 Phase 3 — irregular time grids
#
# `engine='auto'` (variable-step TR-BDF2) returns samples on the
# ACCEPTED grid: dense clusters at every gate edge and diode
# commutation, long strides through smooth stretches. Pointwise
# accessors (v(), i(), plot) consume `times` as data and are
# already correct there. Two families of analysis are NOT:
#
#   * anything that infers a sample rate — an FFT's `d=dt`, a
#     window index computed as t/dt. Feeding a clustered grid to
#     rfftfreq stretches the frequency axis by the ratio of the
#     first spacing to the mean (measured 6.6e6x on a buck, whose
#     first accepted step is an event landing) and smears every
#     harmonic. The fix is to RESAMPLE onto a uniform grid first —
#     which is exactly what a real scope's ADC does, and a no-op
#     on an already-uniform grid.
#   * anything that averages SAMPLES rather than integrating over
#     TIME — an unweighted mean over-weights the clusters, which
#     sit at commutation instants where ripple peaks. Measured
#     +4.6% on a buck diode's RMS at rtol=1e-6, growing tighter
#     with tolerance (denser clusters) rather than converging.
# ---------------------------------------------------------------

def grid_is_uniform(times: np.ndarray, rtol: float = 1e-6) -> bool:
    """True when `times` is a uniform grid to within `rtol`."""
    t = np.asarray(times, dtype=float)
    if t.size < 3:
        return True
    d = np.diff(t)
    if not np.all(np.isfinite(d)) or np.any(d <= 0):
        return False
    return bool(np.max(np.abs(d - d.mean())) <= rtol * d.mean())


def resample_uniform(times: np.ndarray, y: np.ndarray,
                     n_out: "int | None" = None):
    """Linear-interpolate `(times, y)` onto a uniform grid.

    Returns `(t_u, y_u, dt)`. The default sample count preserves
    the input's finest resolution without exploding: the span
    divided by the MEDIAN spacing (the median ignores both the
    event clusters and the long smooth strides), clamped to
    [len(times), 4 * len(times)].
    """
    t = np.asarray(times, dtype=float)
    y = np.asarray(y, dtype=float)
    if n_out is None:
        d = np.diff(t)
        med = float(np.median(d)) if d.size else 0.0
        span = float(t[-1] - t[0])
        est = int(span / med) + 1 if med > 0 else t.size
        n_out = int(np.clip(est, t.size, 4 * t.size))
    t_u = np.linspace(t[0], t[-1], n_out)
    return t_u, np.interp(t_u, t, y), float(t_u[1] - t_u[0])


def time_weighted_rms(y: np.ndarray, times: np.ndarray) -> float:
    """RMS over TIME, not over samples: sqrt(∫y² dt / T).

    Identical to the sample RMS on a uniform grid (to trapezoid
    error) and the only correct form on an irregular one.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(times, dtype=float)
    if y.size == 0:
        return 0.0
    if t.size != y.size or t.size < 2:
        return float(np.sqrt(np.mean(y ** 2)))
    span = float(t[-1] - t[0])
    if not (span > 0):
        return float(np.sqrt(np.mean(y ** 2)))
    return float(np.sqrt(np.trapezoid(y ** 2, t) / span))
