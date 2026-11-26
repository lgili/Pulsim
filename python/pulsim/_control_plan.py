"""One composition of controllers for every engine (fix-control-tick-semantics).

`simulate()` used to compose closed loops, observers and gate functions
twice, once per engine, and the copies diverged (a `|` on two masks that
has no operator on the variable-step path; `controller_period` read on
one engine only). `compose_control` runs once, above the engine
dispatch, and both engines consume its plan.

Tick rule: source j with period T_j fires at t_start + k*T_j for every
integer k >= 0 with that instant < t_end, i.e. ceil(span / T_j) times,
sources in registration order (closed loops in list order, then the
user observer) when several are due at one instant; on pwl the
closed loops that are graphs (kernel helpers) run before the Python
tick sources.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

# Module level, not in compose_control: simulate() pays every import on
# every call (task 7.2 of add-kernel-simulation-object).
from ._pulsim import make_combined_switch_fn  # type: ignore[import-not-found]
from ._control_state import start_run
from .c_block import _c_graph_observer, static_state_hold
from ._plan import emit as _emit
from .errors import options_error

TickFn = Callable[[float, object], None]


@dataclass
class ControlPlan:
    """What the engines need: the merged gate function, the tick sources
    (callback, period) and the observers that run every step."""

    switch_fn: Optional[Callable] = None
    ticks: List[Tuple[TickFn, float]] = field(default_factory=list)
    # Per entry of `ticks` on a variable-step engine: the tick offset
    # (empty = all 0).
    tick_offsets: List[float] = field(default_factory=list)
    every_step: List[TickFn] = field(default_factory=list)
    # Every observer, for the post-run walk that looks for motor bundles.
    inner_observers: List[TickFn] = field(default_factory=list)
    # A control graph with rates (add-compiled-control-graph): its tick
    # callbacks for the variable-step engine, and its driven sources.
    rate_ticks: List[Tuple[TickFn, float]] = field(default_factory=list)
    rate_observer: Optional[TickFn] = None
    rate_offsets: List[float] = field(default_factory=list)
    rate_delays: List[int] = field(default_factory=list)
    b_extra_fn: Optional[Callable] = None
    # Closed loops that are control graphs (label time): their observers
    # (which tick themselves) in loop order, and per entry of `ticks`
    # the graph observer it belongs to (None: a Python tick source).
    loop_graphs: List[TickFn] = field(default_factory=list)
    tick_graph: List[Optional[TickFn]] = field(default_factory=list)
    # The builder's C-block graph (add_c_block), last in every order.
    c_graph: Optional[TickFn] = None
    # The fixed engine's Python tick sources (callback, period), which
    # the kernel's event queue schedules (run_transient tick_fns=).
    _kernel_ticks: List[Tuple[TickFn, float]] = field(default_factory=list)
    # With kernel ticks: what runs before them at each step (the loop
    # graphs, then simulate() adds its MMC / on_cache drivers in front).
    tick_before: Optional[TickFn] = None
    # The run's control graphs, put in their starting state (G26).
    graphs: List[TickFn] = field(default_factory=list)
    # The static-state C libraries this run holds (G27; released with
    # the plan).
    static_hold: Any = None
    _ticks0: dict = field(default_factory=dict)
    # control_tick='state' on the fixed engine: the graphs' rate groups are
    # kernel queue sources (chain, group) after the Python ones, and the
    # (origin, next k) of each on the graph's own grid.
    control_tick: str = "label"
    _queue_groups: List[Tuple[Any, int]] = field(default_factory=list)

    def merge_b_extra(self, user_fn):
        """The b_extra the engines see: the caller's ``b_extra_fn``, then
        the graph's driven sources, then the C blocks' outputs (summed in
        that order, as before)."""
        parts = [f for f in (user_fn, self.b_extra_fn,
                             getattr(self.c_graph, "_b_extra_fn", None))
                 if f is not None]
        if len(parts) <= 1:
            return parts[0] if parts else None
        import numpy as np

        def total(t):
            acc = np.asarray(parts[0](t), dtype=float)
            for f in parts[1:]:
                acc = acc + np.asarray(f(t))
            return acc
        from . import _carry_jumps
        return _carry_jumps(total, parts)

    def for_fixed_step(self, t_start: float, t_end: float, dt: float):
        """The fixed engine's observer: the closed-loop graphs (label
        time), the Python tick sources, the every-step observers, then the
        C-block graph — in that order. The Python tick sources are
        scheduled by the kernel's event queue (add-unified-event-queue
        group 5): with any, the graphs go to `fixed_tick_kwargs()`'s
        `tick_before` (run before the ticks) and the observer returned
        here holds the rest (run after them). All graphs: one C++ observer
        (no Python call per step)."""
        _check_fixed_periods(
            [p for (_f, p), g in zip(self.ticks, self.tick_graph)
             if g is not None], float(dt))
        graphs = list(self.loop_graphs)
        if self.c_graph is not None:
            graphs.append(self.c_graph)
        self._ticks0 = {}
        for obs in graphs + list(self.every_step):
            cxx = getattr(obs, "_cxx_chain", None)
            if cxx is not None and _helper(obs):
                cxx.set_solver_dt(float(dt))
                cxx.set_horizon(float(t_end))
                # a reused helper graph keeps counting: report this run's
                self._ticks0[id(cxx)] = int(sum(cxx.tick_counts()[1:]))
        self._kernel_ticks = [(f, p) for (f, p), g
                              in zip(self.ticks, self.tick_graph) if g is None]
        _check_fixed_periods([p for _f, p in self._kernel_ticks], float(dt))
        post = list(self.every_step) \
            + ([self.c_graph] if self.c_graph is not None else [])
        self._queue_groups = []
        queue = self.control_tick == "state"
        for obs in self.graphs:
            cxx = obs._cxx_chain
            cxx.set_queue_driven(queue)
            if queue:
                groups = range(1, len(cxx.tick_counts()))
                _check_graph_periods(cxx, groups, float(dt))
                self._queue_groups += [(cxx, gi) for gi in groups]
        if self._queue_groups:
            # the graphs run after the queue's ticks, in one pass each
            post = list(self.loop_graphs) + post
            self.tick_before = None
            return self._compose(post)
        if self._kernel_ticks:
            self.tick_before = self._compose(list(self.loop_graphs))
            return self._compose(post)
        self.tick_before = None
        parts = list(self.loop_graphs) + post
        if len(parts) == 1 and getattr(parts[0], "_cxx_chain", None) is not None \
                and not parts[0]._cxx_chain.label_time:
            return parts[0]          # one user chain: the chain fast path
        return self._compose(parts)

    def _compose(self, parts):
        """One observer running `parts` in order (None when empty)."""
        if not parts:
            return None
        if all(getattr(o, "_cxx_chain", None) is not None for o in parts):
            from ._pulsim import compose_chain_observers  # type: ignore[import-not-found]
            tick = compose_chain_observers(
                [(o._cxx_chain, o._chain_dt) for o in parts])
            tick._n_python_blocks = sum(  # type: ignore[attr-defined]
                getattr(o, "_n_python_blocks", 0) for o in parts)
            return tick
        if len(parts) == 1:
            return parts[0]

        def _composed_observer(t: float, x, _obs=tuple(parts)) -> None:
            for obs in _obs:
                obs(t, x)

        # T2.2: the post-run _auto_attach_motor_traces walks these.
        _composed_observer._inner_observers = self.inner_observers  # type: ignore[attr-defined]
        return _composed_observer

    def fixed_tick_kwargs(self, dt: float = 0.0, t_start: float = 0.0,
                          resume_from=None) -> dict:
        """run_transient's tick sources: empty without Python ticks or
        queue-driven graphs. A resume keeps the Python sources' grids (the
        snapshot's origins); a graph group ticks on the graph's own grid,
        one step late: the pwl observer call labelled t_n carries
        x(t_{n-1}), so the group's tick at tau is taken at the call whose
        t_{n-1} >= tau (state time)."""
        if not self._kernel_ticks and not self._queue_groups:
            return {}
        n_py = len(self._kernel_ticks)
        kw = {"tick_periods": [float(p) for _f, p in self._kernel_ticks],
              "tick_fns": [f for f, _p in self._kernel_ticks]}
        if self.tick_before is not None:
            kw["tick_before"] = self.tick_before
        n = n_py + len(self._queue_groups)
        from . import _resume_tick_next_k, _resume_tick_origins
        origins = _resume_tick_origins(resume_from, n)
        ks = _resume_tick_next_k(resume_from, len(origins))
        if not self._queue_groups:
            if origins:   # a resume keeps the run's tick grid
                kw["tick_origins"], kw["tick_next_ks"] = origins, ks
            return kw
        nan, unknown = float("nan"), 2 ** 64 - 1
        kw["tick_origins"] = origins[:n_py] if origins else [nan] * n_py
        kw["tick_next_ks"] = ks[:n_py] if ks else [unknown] * n_py
        # A group's instant tau is queued at tau + dt, so the last one the
        # run can serve (tau = t_end - dt, at the call labelled t_end) sits
        # AT t_end: the queue's run end is closed for them (release finding
        # F-4-7: the half-open end dropped it, n steps -> n - 1 fires).
        kw["tick_closed_ends"] = [False] * n_py + [True] * len(self._queue_groups)
        for cxx, gi in self._queue_groups:
            # the graph continues its grid unless time goes back (it then
            # rearms at t_start: rearm_if_restarted)
            cont = bool(cxx.n_calls) and float(t_start) >= float(cxx.last_call_t)
            t0 = float(cxx.grid_t0) if cont else float(t_start)
            kw["tick_periods"].append(float(cxx.group_period(gi)))
            kw["tick_fns"].append(cxx.make_queue_tick(gi))
            kw["tick_origins"].append(t0 + float(cxx.group_offset(gi)) + float(dt))
            kw["tick_next_ks"].append(int(cxx.group_next_k(gi)) if cont else 0)
        return kw

    def fixed_tick_counts(self, res):
        """Per tick source in registration order (graphs read their own
        counters, the Python sources the kernel's) and the largest
        lateness, after a fixed-step run."""
        counts, jitter, j = [], 0.0, 0
        kcounts = list(getattr(res, "kernel_tick_counts", ()))
        for g in self.tick_graph:
            if g is None:
                counts.append(int(kcounts[j]))
                j += 1
            else:
                cxx = g._cxx_chain
                counts.append(int(sum(cxx.tick_counts()[1:]))
                              - self._ticks0.get(id(cxx), 0))
                jitter = max(jitter, float(cxx.tick_jitter_max))
        for obs in self.every_step:      # a helper graph passed as step_observer
            cxx = getattr(obs, "_cxx_chain", None)
            if cxx is not None and _helper(obs):
                counts.append(int(sum(cxx.tick_counts()[1:]))
                              - self._ticks0.get(id(cxx), 0))
                jitter = max(jitter, float(cxx.tick_jitter_max))
        if self._kernel_ticks:
            jitter = max(jitter, float(res.kernel_tick_jitter_max))
        return counts, jitter

    def for_variable_step(self, engine: str = "trbdf2") -> "ControlPlan":
        """The plan for an engine that lands steps on tick instants: the
        graph runs on its group ticks instead of every step, each group
        at t_start + offset + k*period (the kernel's event queue).
        Refused: a compute_delay time (the kernel lands no second event
        per tick yet)."""
        if not self.rate_ticks and self.c_graph is None:
            return self
        from ._refusal import unsupported
        if any(d == 1 for d in self.rate_delays):
            raise unsupported(
                engine, "compute_delay",
                f"simulate(engine={engine!r}): a compute_delay time needs a "
                "second event per tick, which this engine does not schedule "
                "yet; use engine='pwl', 0 or 'one_sample'.")
        c_ticks = list(getattr(self.c_graph, "_rate_ticks", []) or [])
        c_offsets = list(getattr(self.c_graph, "_rate_offsets", []) or [])
        rate_offsets = list(self.rate_offsets) or [0.0] * len(self.rate_ticks)
        offsets = ([0.0] * len(self.ticks) + rate_offsets
                   + (c_offsets or [0.0] * len(c_ticks)))
        ticks = list(self.ticks) + list(self.rate_ticks) + c_ticks
        assert len(offsets) == len(ticks), (len(offsets), len(ticks))
        return ControlPlan(
            switch_fn=self.switch_fn,
            ticks=ticks,
            tick_offsets=[float(o) for o in offsets],
            every_step=[o for o in self.every_step
                        if o is not self.rate_observer],
            inner_observers=self.inner_observers,
            b_extra_fn=self.b_extra_fn,
            graphs=self.graphs,
            static_hold=self.static_hold)


def compose_control(builder, closed_loops, step_observer, controller_period,
                    switch_fn, resume: bool = False,
                    snapshot=None, control_tick: str = "label") -> ControlPlan:
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
        if getattr(obs, "_rate_ticks", None):
            # A control-graph loop (bind_pi_to_switch): it ticks itself
            # on pwl; trbdf2 calls its C++ tick at every period.
            plan.loop_graphs.append(obs)
            plan.ticks.append((obs._rate_ticks[0][0], period))
            plan.tick_graph.append(obs)
        elif period > 0.0:
            # `.tick` is the loop's unthrottled update; the schedule,
            # not the loop's own elapsed-time throttle, decides.
            plan.ticks.append((getattr(obs, "tick", obs), period))
            plan.tick_graph.append(None)
        else:
            plan.every_step.append(obs)
        plan.inner_observers.append(obs)
    if step_observer is not None and controller_period is not None and \
            getattr(step_observer, "_rate_ticks", None):
        raise ValueError(
            "simulate(controller_period=...) with a chain that has "
            "Rate groups: the groups already set when the graph ticks. "
            "Drop controller_period.")
    if step_observer is not None:
        period = controller_period
        if period is None:
            period = getattr(step_observer, "period", None)
        if period is not None:
            # a Monitor stays itself: its flag tells the engines it reads only
            plan.ticks.append(
                (step_observer if _monitor(step_observer)
                 else getattr(step_observer, "tick", step_observer),
                 float(period)))
            plan.tick_graph.append(None)
        else:
            plan.every_step.append(step_observer)
        plan.inner_observers.append(step_observer)
    elif controller_period is not None:
        raise ValueError(
            "simulate(controller_period=...) was given without a "
            "step_observer: there is nothing to call at that period. "
            "Pass step_observer=, or drop controller_period.")
    if step_observer is not None:
        plan.rate_ticks = list(getattr(step_observer, "_rate_ticks", []))
        if plan.rate_ticks:
            plan.rate_observer = step_observer
            plan.rate_offsets = list(step_observer._rate_offsets)
            plan.rate_delays = list(step_observer._rate_delays)
        plan.b_extra_fn = getattr(step_observer, "_b_extra_fn", None)
    plan.static_hold = static_state_hold(           # G27: one run at a time
        builder, plan.inner_observers)
    plan.c_graph = _c_graph_observer(builder, fresh=not resume)
    # every graph from its declared initial state, or the snapshot's
    plan.graphs = start_run(plan, snapshot if resume else None)
    plan.control_tick = control_tick
    # control_tick='state' (G30): the label-time graphs (closed-loop
    # helpers, C blocks) tick at the state time for this run
    for obs in plan.graphs:
        g = getattr(obs, "_graph", None)
        if g is not None and g.label_time:
            obs._cxx_chain.set_label_time(control_tick == "label")
    for _fn, period in plan.ticks:
        if not (period > 0.0 and math.isfinite(period)):
            raise ValueError(
                f"simulate(): controller period {period!r} must be a "
                "positive finite number of seconds")
    return plan


def _monitor(obs) -> bool:
    """A declared read-only tick (pulsim.control.Monitor)."""
    return bool(getattr(obs, "_pulsim_monitor", False))


def monitor_flags(ticks) -> list:
    """Per tick source: True for a Monitor (engines neither land on it nor
    re-sample the gate after it)."""
    return [_monitor(f) for f, _p in ticks]


def _helper(obs) -> bool:
    """A label-time graph (helpers, C blocks), whatever mode this run's
    control_tick put its chain in."""
    return bool(getattr(getattr(obs, "_graph", None), "label_time", False))


def _check_graph_periods(cxx, groups, dt: float) -> None:
    """The graph's own check (BlockChain::check_periods), before the run: a
    queue-driven graph meets it only at its first in-run call, where the
    kernel re-wraps the ControlGraphBuildError as SimulationAborted (release
    finding F-3-5); the chain path checks in set_solver_dt, before the run."""
    for gi in groups:
        period = float(cxx.group_period(gi))
        if period < dt * (1.0 - 1e-9):
            from ._pulsim import ControlGraphBuildError  # type: ignore[import-not-found]
            raise ControlGraphBuildError(
                f"control graph: a Rate period ({period:.6g} s) is shorter than "
                f"the fixed solver step ({dt:.6g} s); a group ticks at most once "
                "per step. Use a period >= dt, or a smaller dt.")


def _check_fixed_periods(periods, dt: float, stacklevel: int = 6) -> None:
    """A period below dt is refused (two ticks in one step); a period
    that is not a multiple of dt warns (ticks late by less than dt)."""
    for period in periods:
        if period < dt * (1.0 - 1e-9):
            raise options_error(
                "dt",
                f"simulate(): controller period {period:.6g} s is "
                f"shorter than dt = {dt:.6g} s, so two ticks "
                "would fall in one step. Pass dt <= the period.")
    off = [p for p in periods
           if abs(p / dt - round(p / dt)) > 1e-9 * (p / dt)]
    if off:
        _emit("controller_period",
            f"simulate(): controller period(s) {off} are not a "
            f"multiple of dt = {dt:.6g} s; each tick is "
            "delivered at the first step at or after its instant, "
            f"late by less than dt (result.control_tick_jitter_max "
            "records the largest lateness). Pick dt = T/n to remove it.",
            stacklevel=stacklevel)
