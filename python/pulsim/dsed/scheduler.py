"""PED scheduler — the outer loop of the path-based event-driven engine.

Gate 1 prototype: pure-Python, no path-based LU integration yet
(that's Gate 4 with BDF2). The scheduler:

  while t < t_end:
    1. compute next *gate* edge t_gate (analytical, from switch_fn)
    2. propose dt from PI controller, capped by (t_gate - t)
    3. step integrator (DOPRI5)
    4. if step rejected: shrink dt, retry
    5. POST-STEP predicate scan over (t, t+dt) for diode/ZCD events
       (uses cubic Hermite from FSAL's pre-/post-step derivatives;
       Illinois on the resulting g(τ))
    6. if a predicate fired strictly inside the step: backtrack to
       t_event, project state (e.g. clamp i_L=0 for ZCD),
       fire event, reset FSAL+PI
    7. elif step landed on t_gate: fire gate event, reset FSAL+PI
    8. record state

Gate 1 + 2 exercised only the gate-edge fast path (no diodes).
Gate 3 (this iteration) lights up the predicate scan + projection
hooks for buck DCM and body-diode commutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from . import rk45_dormand_prince as rk45
from .event_predictor import EventPredictor, brent_fallback, illinois
from .step_controller import PIController


# Closeness threshold for "event fired exactly at the step endpoint".
# Tighter than the Illinois tolerance (1e-10) to avoid spurious backtracks
# when the predicate root lands within numerical noise of t + h_use.
_BACKTRACK_TOL = 1e-15


# Projection callback signature: (t_event, x_pre, predicate_type) -> x_post
# Default = identity (no projection).
StateProjection = Callable[[float, np.ndarray, str], np.ndarray]


def _hermite_interp(
    x0: np.ndarray,
    x1: np.ndarray,
    k1: np.ndarray,
    k7: np.ndarray,
    h: float,
    theta: float,
) -> np.ndarray:
    """Cubic Hermite over (t, t+h) using FSAL's pre-/post-step derivatives.

    DOPRI5's FSAL gives us k1 = f(t, x) before the step and k7 = f(t+h, x1)
    after the step at no extra cost. The Hermite cubic through
    ``(t, x0, k1)`` and ``(t+h, x1, k7)`` is order-4 accurate inside [t, t+h],
    well within DOPRI5's dense-output budget.
    """
    if theta <= 0.0:
        return x0.copy()
    if theta >= 1.0:
        return x1.copy()
    theta_sq = theta * theta
    theta_cu = theta_sq * theta
    h00 = 2.0 * theta_cu - 3.0 * theta_sq + 1.0
    h10 = theta_cu - 2.0 * theta_sq + theta
    h01 = -2.0 * theta_cu + 3.0 * theta_sq
    h11 = theta_cu - theta_sq
    return h00 * x0 + h10 * h * k1 + h01 * x1 + h11 * h * k7


@dataclass
class EventRecord:
    """One row in the simulation's event log."""
    t: float
    name: str
    predicate_type: str
    old_mask: Any
    new_mask: Any


@dataclass
class PEDResult:
    """Output of a PED simulation."""
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    states: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    event_log: List[EventRecord] = field(default_factory=list)
    n_accept: int = 0
    n_reject: int = 0
    n_events: int = 0
    cpu_time: float = 0.0


class PEDSimulator:
    """Path-Based Event-Driven simulator (Gate 1 prototype).

    The ``system`` argument must expose:

    - ``rhs(t, x) -> dx/dt`` — RHS for the current mask
    - ``set_mask(mask) -> None`` — swap to a new mask
    - ``current_mask`` property — read current mask for event resolver

    The ``switch_fn(t) -> mask`` callable defines the gate edges.
    The event predictor learns the next gate edge from
    ``switch_fn`` via ``switch_fn.next_edge_after(t_now)``
    if available (otherwise scanning is used).
    """

    def __init__(
        self,
        system: Any,
        switch_fn: Callable[[float], Any],
        predictor: Optional[EventPredictor] = None,
        controller: Optional[PIController] = None,
        dt_init: float = 1e-9,
        dt_max: float = 1e-5,
        store_every: int = 1,
        state_projection: Optional[StateProjection] = None,
    ) -> None:
        self.system = system
        self.switch_fn = switch_fn
        self.predictor = predictor or EventPredictor()
        self.controller = controller or PIController()
        self.dt_init = dt_init
        self.dt_max = dt_max
        self.store_every = store_every
        self.state_projection = state_projection

    def simulate(self, x0: np.ndarray, t_end: float) -> PEDResult:
        """Run the simulation from t=0 to t=t_end."""
        import time

        t0 = time.perf_counter()

        x = np.asarray(x0, dtype=float).copy()
        t = 0.0
        h = self.dt_init
        mask = self.switch_fn(t)
        self.system.set_mask(mask)

        rk_state = rk45.RK45State()
        result = PEDResult()
        times_list = [t]
        states_list = [x.copy()]
        step_idx = 0

        # Hard cap to avoid runaway in tests
        max_steps = 10_000_000

        def f(tau: float, xx: np.ndarray) -> np.ndarray:
            return self.system.rhs(tau, xx)

        while t < t_end:
            if result.n_accept + result.n_reject > max_steps:
                raise RuntimeError(
                    f"PED simulator: exceeded {max_steps} steps "
                    f"at t={t:.6e}/{t_end:.6e}"
                )

            # 1. Compute next gate-edge time (analytical fast path)
            t_gate = self._next_gate_edge(t, t_end)
            t_target = min(t_gate, t_end)

            # 2. Propose step capped at next gate
            h_use = min(h, self.dt_max, t_target - t)
            if h_use <= 0:
                # We're sitting exactly on a gate edge — fire it.
                if t_gate < t_end:
                    self._fire_event(
                        t_gate, "gate_edge", "gate", x, rk_state, result,
                    )
                    t = t_gate
                continue

            # 3. Snapshot k1 BEFORE step (needed for Hermite backtrack
            # when a predicate fires inside the step).
            need_predicate_scan = bool(self.predictor.predicates)
            k1_pre: Optional[np.ndarray] = None
            if need_predicate_scan:
                if rk_state.fsal_valid and rk_state.k1 is not None:
                    k1_pre = rk_state.k1.copy()
                else:
                    k1_pre = self.system.rhs(t, x)

            # 4. Take step (DOPRI5 with FSAL)
            x_new, err = rk45.step(f, t, x, h_use, rk_state)

            # 5. PI accept/reject
            accepted, h_next = self.controller.accept(err, x, x_new, h_use)
            if not accepted:
                rk_state.invalidate()
                h = h_next
                result.n_reject += 1
                continue

            # 6. Post-step predicate scan for diode/ZCD events.
            # rk_state.k1 now holds k7 of the just-taken step (FSAL).
            t_pred: Optional[float] = None
            p_name: Optional[str] = None
            p_type: Optional[str] = None
            x_pred: Optional[np.ndarray] = None
            if need_predicate_scan:
                assert k1_pre is not None
                k7 = rk_state.k1
                assert k7 is not None
                t_pred, p_name, p_type, x_pred = self._locate_event_in_step(
                    t, x, t + h_use, x_new, k1_pre, k7, h_use,
                )

            if (
                t_pred is not None
                and p_name is not None
                and p_type is not None
                and x_pred is not None
                and t_pred < t + h_use - _BACKTRACK_TOL
            ):
                # Predicate fired strictly inside the step — backtrack.
                t = t_pred
                x = x_pred
                self._fire_event(t, p_name, p_type, x, rk_state, result)
                # After event: PI is reset, FSAL invalid. Pick a small h
                # to restart the new mask's dynamics safely.
                h = max(self.dt_init, min(h_next, h_use))
                result.n_accept += 1
            else:
                # Commit full step
                t = t + h_use
                x = x_new
                h = h_next
                result.n_accept += 1

                # 7. Did we land on a gate edge?
                if t_gate < t_end and abs(t - t_gate) < 1e-12:
                    self._fire_event(
                        t_gate, "gate_edge", "gate", x, rk_state, result,
                    )

            # 8. Record
            step_idx += 1
            if step_idx % self.store_every == 0:
                times_list.append(t)
                states_list.append(x.copy())

        # Final state
        if times_list[-1] != t:
            times_list.append(t)
            states_list.append(x.copy())

        result.times = np.asarray(times_list)
        result.states = np.asarray(states_list)
        result.cpu_time = time.perf_counter() - t0
        return result

    def _next_gate_edge(self, t_now: float, t_end: float) -> float:
        """Next *gate* edge time from switch_fn (does not include
        predicate-driven events like ZCD)."""
        next_edge_fn = getattr(self.switch_fn, "next_edge_after", None)
        if next_edge_fn is None:
            return t_end
        return min(float(next_edge_fn(t_now)), t_end)

    def _locate_event_in_step(
        self,
        t0: float,
        x0: np.ndarray,
        t1: float,
        x1: np.ndarray,
        k1: np.ndarray,
        k7: np.ndarray,
        h: float,
    ) -> Tuple[Optional[float], Optional[str], Optional[str], Optional[np.ndarray]]:
        """Scan all armed predicates over (t0, t1). Return the earliest
        sign-change root via Illinois on the Hermite cubic.

        Returns ``(t_event, predicate_name, predicate_type, x_at_event)``
        for the firing event, or ``(None, None, None, None)`` if no
        predicate changed sign.

        Ties on simultaneous events are broken by predicate priority
        (lower wins — see EventPredictor._DEFAULT_PRIORITY).
        """
        best_t: Optional[float] = None
        best_p: Optional[Any] = None
        best_x: Optional[np.ndarray] = None

        for p in self.predictor.predicates:
            g0 = p.value(t0, x0)
            g1 = p.value(t1, x1)

            # Boundary cases — predicate already at zero at the start of
            # the step means we just fired it; skip to avoid re-firing.
            if g0 == 0.0:
                continue
            if g0 * g1 >= 0.0:
                continue  # no sign change in this step

            # Locate root via Illinois on the Hermite interpolant.
            def g_at(tau: float, _p=p) -> float:
                if tau <= t0:
                    return g0
                if tau >= t1:
                    return g1
                theta = (tau - t0) / h
                x_tau = _hermite_interp(x0, x1, k1, k7, h, theta)
                return _p.value(tau, x_tau)

            try:
                t_root = illinois(g_at, t0, t1, g0, g1)
            except RuntimeError:
                self.predictor.illinois_failures += 1
                t_root = brent_fallback(g_at, t0, t1)

            # Track best (earliest, then priority tiebreak)
            if best_t is None or t_root < best_t - _BACKTRACK_TOL:
                theta = (t_root - t0) / h
                x_root = _hermite_interp(x0, x1, k1, k7, h, theta)
                best_t, best_p, best_x = t_root, p, x_root
            elif (
                best_p is not None
                and abs(t_root - best_t) <= _BACKTRACK_TOL
                and p.priority < best_p.priority
            ):
                theta = (t_root - t0) / h
                x_root = _hermite_interp(x0, x1, k1, k7, h, theta)
                best_t, best_p, best_x = t_root, p, x_root

        if best_t is None or best_p is None or best_x is None:
            return None, None, None, None
        return best_t, best_p.name, best_p.predicate_type, best_x

    def _fire_event(
        self,
        t_event: float,
        name: str,
        ptype: str,
        x: np.ndarray,
        rk_state: rk45.RK45State,
        result: PEDResult,
    ) -> None:
        """Apply an event: notify switch_fn of ZCD, project state,
        swap mask, invalidate FSAL, reset PI.

        For diode/ZCD events the switch_fn may need to be notified
        (so subsequent ``switch_fn(t)`` calls return the post-ZCD mode
        for the remainder of this switching cycle). For Gate 3 this
        is signalled via the optional ``register_zcd_transition(t)``
        method on the switch_fn.

        State projection (e.g. clamping ``i_L = 0`` after ZCD) is
        delegated to the optional ``state_projection`` callback
        passed to ``__init__``. The default = identity.
        """
        old_mask = self.system.current_mask

        # Notify switch_fn of non-gate events for state-machine bookkeeping
        if ptype != "gate":
            register = getattr(self.switch_fn, "register_zcd_transition", None)
            if register is not None and ptype in {
                "current_zc", "diode_off", "diode_on",
            }:
                register(t_event)

        # State projection — clamp algebraic constraints at the event
        if self.state_projection is not None:
            x_proj = self.state_projection(t_event, x, ptype)
            x[:] = np.asarray(x_proj, dtype=float)

        new_mask = self.switch_fn(t_event + 1e-15)

        # Gate-edge that's spurious (same mask both sides) — silently drop
        if ptype == "gate" and new_mask == old_mask:
            return

        if new_mask != old_mask:
            self.system.set_mask(new_mask)

        rk_state.invalidate()       # f discontinues at event
        self.controller.reset()     # avoid PI wind-up across mode change

        result.event_log.append(
            EventRecord(
                t=t_event,
                name=name,
                predicate_type=ptype,
                old_mask=old_mask,
                new_mask=new_mask,
            )
        )
        result.n_events += 1
