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
"""
from __future__ import annotations

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
    and return a boolean ndarray of length ``len(times)``."""
    out = np.zeros(times.size, dtype=bool)
    for k, t in enumerate(times):
        mask = switch_fn(float(t))
        out[k] = bool(mask.get(int(switch_idx)))
    return out
