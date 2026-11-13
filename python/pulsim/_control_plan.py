"""One composition of controllers for every engine (fix-control-tick-semantics).

`simulate()` used to compose closed loops, observers and gate functions
twice, once per engine, and the copies diverged (a `|` on two masks that
has no operator on the variable-step path; `controller_period` read on
one engine only). `compose_control` runs once, above the engine
dispatch, and both engines consume its plan.

Tick rule: source j with period T_j fires at t_start + k*T_j for every
integer k >= 0 with that instant < t_end, i.e. ceil(span / T_j) times,
sources in registration order (closed loops in list order, then the
user observer) when several are due at one instant.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

TickFn = Callable[[float, object], None]


@dataclass
class ControlPlan:
    """What the engines need: the merged gate function, the tick sources
    (callback, period) and the observers that run every step."""

    switch_fn: Optional[Callable] = None
    ticks: List[Tuple[TickFn, float]] = field(default_factory=list)
    every_step: List[TickFn] = field(default_factory=list)
    # Every observer, for the post-run walk that looks for motor bundles.
    inner_observers: List[TickFn] = field(default_factory=list)


def compose_control(builder, closed_loops, step_observer, controller_period,
                    switch_fn) -> ControlPlan:
    from ._pulsim import make_combined_switch_fn  # type: ignore[import-not-found]

    loops = list(closed_loops or [])
    plan = ControlPlan(switch_fn=switch_fn)
    gates = [loop.switch_fn for loop in loops if loop.switch_fn is not None]
    if gates:
        if switch_fn is not None:
            gates.append(switch_fn)
        plan.switch_fn = (gates[0] if len(gates) == 1 else
                          make_combined_switch_fn(
                              builder.graph.num_switches, gates))
    for loop in loops:
        obs = loop.step_observer
        period = float(getattr(loop, "period", 0.0) or 0.0)
        if period > 0.0:
            # `.tick` is the loop's unthrottled update; the schedule,
            # not the loop's own elapsed-time throttle, decides.
            plan.ticks.append((getattr(obs, "tick", obs), period))
        else:
            plan.every_step.append(obs)
        plan.inner_observers.append(obs)
    if step_observer is not None:
        period = controller_period
        if period is None:
            period = getattr(step_observer, "period", None)
        if period is not None:
            plan.ticks.append(
                (getattr(step_observer, "tick", step_observer),
                 float(period)))
        else:
            plan.every_step.append(step_observer)
        plan.inner_observers.append(step_observer)
    elif controller_period is not None:
        raise ValueError(
            "simulate(controller_period=...) was given without a "
            "step_observer: there is nothing to call at that period. "
            "Pass step_observer=, or drop controller_period.")
    for _fn, period in plan.ticks:
        if not (period > 0.0 and math.isfinite(period)):
            raise ValueError(
                f"simulate(): controller period {period!r} must be a "
                "positive finite number of seconds")
    return plan


class FixedStepTicks:
    """The fixed engine's scheduler: tick k of source j is delivered at
    the first grid point t_n >= t_start + k*T_j (to 1e-9*dt), counted
    with an integer k, so no tick is lost and nothing drifts. Called at
    the start of every step with (t_n, x), before the step's mask; the
    kernel also calls it at t_end, where no tick is due (instants are
    < t_end)."""

    def __init__(self, ticks, t_start: float, t_end: float, dt: float):
        self._fns = [fn for fn, _ in ticks]
        self._periods = [float(p) for _, p in ticks]
        self._t0 = float(t_start)
        self._dt = float(dt)
        self._t_end = float(t_end)
        self._k = [0] * len(ticks)
        self.counts = [0] * len(ticks)
        self.jitter_max = 0.0
        for period in self._periods:
            if period < self._dt * (1.0 - 1e-9):
                raise ValueError(
                    f"simulate(): controller period {period:.6g} s is "
                    f"shorter than dt = {self._dt:.6g} s, so two ticks "
                    "would fall in one step. Pass dt <= the period.")
        off = [p for p in self._periods
               if abs(p / self._dt - round(p / self._dt)) > 1e-9 * (p / self._dt)]
        if off:
            warnings.warn(
                f"simulate(): controller period(s) {off} are not a "
                f"multiple of dt = {self._dt:.6g} s; each tick is "
                "delivered at the first step at or after its instant, "
                f"late by less than dt (result.control_tick_jitter_max "
                "records the largest lateness). Pick dt = T/n to remove it.",
                stacklevel=4)

    def __call__(self, t: float, x) -> None:
        tol = 1e-9 * self._dt
        for j, fn in enumerate(self._fns):
            due = self._t0 + self._k[j] * self._periods[j]
            if t >= due - tol and due < self._t_end - tol:
                fn(t, x)
                self._k[j] += 1
                self.counts[j] += 1
                self.jitter_max = max(self.jitter_max, t - due)
