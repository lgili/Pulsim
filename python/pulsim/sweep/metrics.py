"""Metric extractors for Pulsim sweep results.

A metric is a callable `(SimulationResult, Circuit) → dict[str, float]`
that maps a single simulation outcome onto one or more named scalar
metrics. The sweep harness calls each metric on every sample and
collects the result into the `SweepResult.metrics` table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np


__all__ = [
    "Metric",
    "steady_state",
    "peak",
    "rms",
    "settling_time",
    "custom",
]


@dataclass
class Metric:
    """A named metric extractor with a `name` (used as the column key
    in the resulting metric table) and a `fn` callable that consumes a
    `(simulation_result, circuit)` pair and returns a scalar."""
    name: str
    fn: Callable[[Any, Any], float]

    def __call__(self, result: Any, circuit: Any) -> float:
        return float(self.fn(result, circuit))


def _node_index(circuit: Any, node_name: str) -> int:
    idx = circuit.get_node(node_name)
    if idx < 0:
        raise ValueError(
            f"metric: node {node_name!r} not found in circuit "
            "(check the spelling against `circuit.add_node(...)` calls)")
    return idx


def _channel_trace(result: Any, circuit: Any, channel: str) -> np.ndarray:
    """Extract a 1-D trace for a named circuit node from a Pulsim
    SimulationResult. Skips empty results gracefully."""
    if not getattr(result, "success", False):
        raise RuntimeError(
            f"metric: simulation failed (failure_reason={result.failure_reason!r})")
    states = result.states
    if not states:
        raise RuntimeError("metric: simulation returned no state samples")
    idx = _node_index(circuit, channel)
    return np.asarray([s[idx] for s in states], dtype=float)


def steady_state(channel: str, *, t_window: tuple[float, float] | None = None) -> Metric:
    """Mean of the channel over the time window `t_window = (t0, t1)`.
    When `t_window` is None, takes the last 10 % of the run as the
    "steady-state" window — a sensible default for converters that
    settle within their full run.
    """
    def _fn(result: Any, circuit: Any) -> float:
        trace = _channel_trace(result, circuit, channel)
        time = np.asarray(result.time, dtype=float)
        if t_window is None:
            t0 = float(time[-1] * 0.9)
            t1 = float(time[-1])
        else:
            t0, t1 = t_window
        mask = (time >= t0) & (time <= t1)
        if not mask.any():
            return float("nan")
        return float(np.mean(trace[mask]))

    label = (f"steady_state[{channel}]" if t_window is None
             else f"steady_state[{channel}, {t_window[0]:.3e}–{t_window[1]:.3e}]")
    return Metric(name=label, fn=_fn)


def peak(channel: str) -> Metric:
    """Peak (maximum) of the channel over the entire run."""
    return Metric(
        name=f"peak[{channel}]",
        fn=lambda r, c: float(np.max(_channel_trace(r, c, channel))),
    )


def rms(channel: str) -> Metric:
    """RMS of the channel over the entire run."""
    def _fn(r: Any, c: Any) -> float:
        trace = _channel_trace(r, c, channel)
        return float(math.sqrt(np.mean(trace ** 2)))
    return Metric(name=f"rms[{channel}]", fn=_fn)


def settling_time(channel: str, *, target: float, tolerance: float = 0.02) -> Metric:
    """Time after which the channel stays inside `±tolerance·target`.
    Returns `+inf` if it never settles.
    """
    def _fn(r: Any, c: Any) -> float:
        trace = _channel_trace(r, c, channel)
        time = np.asarray(r.time, dtype=float)
        bound = abs(tolerance * target)
        # Walk backwards: find the last t at which |trace - target| > bound.
        deviations = np.abs(trace - target)
        out_of_band = deviations > bound
        if not out_of_band.any():
            return float(time[0])     # already settled at t=0
        last_violation_idx = int(np.where(out_of_band)[0].max())
        if last_violation_idx >= len(time) - 1:
            return float("inf")        # still violating at end
        return float(time[last_violation_idx + 1])

    return Metric(
        name=f"settling_time[{channel}, target={target}, tol={tolerance}]",
        fn=_fn)


def custom(name: str, fn: Callable[[Any, Any], float]) -> Metric:
    """Wrap a user-supplied callable into a Metric."""
    return Metric(name=name, fn=fn)
