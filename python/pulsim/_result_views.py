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
    state vectors) to a contiguous ``(N, state_size)`` float array."""
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

    1. A :class:`SwitchMaskRecorder` passed as ``switch_fn`` (exact
       simulate-time history) — or a ``result._switch_masks`` 2-D bool
       array attached by the caller.
    2. Re-evaluate ``switch_fn(t)`` (faithful for stateless switch_fns)
       then apply :func:`_voltage_consistency_guard`.
    """
    rec = getattr(switch_fn, "recorded_mask_trace", None)
    if callable(rec):
        return np.asarray(rec(times, switch_idx), dtype=bool)
    masks = getattr(result, "_switch_masks", None)
    if masks is not None:
        arr = np.asarray(masks)
        if (arr.ndim == 2 and arr.shape[0] == np.asarray(times).size
                and 0 <= switch_idx < arr.shape[1]):
            return arr[:, switch_idx].astype(bool)
    closed = evaluate_switch_mask_trace(switch_fn, times, switch_idx)
    return _voltage_consistency_guard(closed, v_branch, name=name)
