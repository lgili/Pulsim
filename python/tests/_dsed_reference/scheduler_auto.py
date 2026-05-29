"""Auto-dispatch PED scheduler — Python port of ``scheduler_auto.hpp`` (Gate 4.D).

Unified PED scheduler that picks DOPRI5 (explicit) or BDF2 (implicit)
per mode-segment based on :class:`StiffnessDetector.select`. This is
the algorithmic capstone of Gate 4: the scheduler INHERITS the BEST
integrator for each operating regime, instead of forcing the user
to commit at construction time.

Outer loop (one iteration):

1. Compute t_gate = next gate edge.
2. Query stiffness detector for the CURRENT mode + tentative h.
3. Set integrator_choice for this mode-segment.
4. Take steps until t_gate or t_end:
     choice == DOPRI5  →  rk45.step + PI adapt + Hermite/predicate scan
     choice == BDF2    →  bdf2_step (fixed h_bdf2; cap at t_gate - t)
5. At t_gate: fire event, swap mask, INVALIDATE BOTH integrator
   states, re-query stiffness for the new mode.

Validated end-to-end in C++ (notes/GATE5_PROGRESS.md): per-mode
dispatch tracks the theoretical speedup ceiling at 80-100% across
stiff-fraction sweeps.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, List

import numpy as np

from . import rk45_dormand_prince as rk45
from .bdf2_integrator import BDF2State, bdf2_step
from .step_controller import PIController
from .stiffness_detector import IntegratorChoice, StiffnessDetector


@dataclass
class AutoDispatchEventRecord:
    """One row in the auto-dispatcher's event log — extends EventRecord
    with the integrator that was active for the segment ending here."""
    t: float
    name: str
    predicate_type: str
    old_mask: Any
    new_mask: Any
    integrator_used: IntegratorChoice


@dataclass
class PEDResultAuto:
    """Output of an auto-dispatch PED simulation."""
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    states: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    event_log: List[AutoDispatchEventRecord] = field(default_factory=list)
    n_accept: int = 0
    n_reject: int = 0
    n_events: int = 0
    n_rk45_steps: int = 0
    n_bdf2_steps: int = 0
    cpu_time: float = 0.0


class PEDSimulatorAuto:
    """Auto-dispatch PED simulator: picks DOPRI5 or BDF2 per mode-segment.

    Parameters
    ----------
    system
        Must expose ``A_matrix() / b_vector(t) / rhs(t, x) / set_mask /
        current_mask``.
    switch_fn
        Callable ``t -> mask``; ideally with ``next_edge_after(t)``.
    controller
        PI controller for RK45 segments (default: PIController()).
    detector
        Stiffness detector (default: StiffnessDetector()).
    dt_init, dt_max
        Initial / max step for RK45 segments.
    h_bdf2
        Fixed step for BDF2 segments.
    store_every
        Record every Nth accepted step (default 1).
    """

    def __init__(
        self,
        system: Any,
        switch_fn: Callable[[float], Any],
        controller: PIController = None,                # type: ignore[assignment]
        detector: StiffnessDetector = None,             # type: ignore[assignment]
        dt_init: float = 1e-9,
        dt_max: float = 1e-5,
        h_bdf2: float = 1e-6,
        store_every: int = 1,
    ) -> None:
        self.system = system
        self.switch_fn = switch_fn
        self.controller = controller or PIController()
        self.detector = detector or StiffnessDetector()
        self.dt_init = dt_init
        self.dt_max = dt_max
        self.h_bdf2 = h_bdf2
        self.store_every = store_every

    def simulate(self, x0: np.ndarray, t_end: float) -> PEDResultAuto:
        """Run the auto-dispatch simulation from ``t=0`` to ``t=t_end``."""
        t0_wall = time.perf_counter()

        x = np.asarray(x0, dtype=float).copy()
        t = 0.0
        h_rk45 = self.dt_init

        mask = self.switch_fn(t)
        self.system.set_mask(mask)

        # Pick initial integrator
        choice = self.detector.select(
            self._mode_id(mask), self.system.A_matrix(), self.dt_max)

        rk_state = rk45.RK45State()
        bdf2_state = BDF2State()
        result = PEDResultAuto()
        times_list = [t]
        states_list = [x.copy()]
        step_idx = 0

        max_steps = 10_000_000
        kEpsEdge = 1e-12

        def f(tau: float, xx: np.ndarray) -> np.ndarray:
            return self.system.rhs(tau, xx)

        def b_fn(tau: float) -> np.ndarray:
            return self.system.b_vector(tau)

        while t < t_end:
            if result.n_accept + result.n_reject > max_steps:
                raise RuntimeError(
                    f"PEDSimulatorAuto: exceeded {max_steps} steps "
                    f"at t={t:.6e}/{t_end:.6e}"
                )

            # 1. Next gate-edge time
            t_gate = self._next_gate_edge(t, t_end)

            # 2. Take one step using the current mode's integrator
            if choice == IntegratorChoice.BDF2:
                h_use = min(self.h_bdf2, t_gate - t)
                if h_use <= 0:
                    if t_gate < t_end:
                        prev_choice = choice
                        self._fire_gate_event(
                            t_gate, rk_state, bdf2_state,
                            prev_choice, result)
                        t = t_gate
                        choice = self.detector.select(
                            self._mode_id(self.system.current_mask()),
                            self.system.A_matrix(), self.dt_max)
                        if prev_choice == IntegratorChoice.BDF2 \
                            and choice == IntegratorChoice.DOPRI5:
                            h_rk45 = self.dt_init
                    continue
                A_cur = self.system.A_matrix()
                x_new, _err = bdf2_step(A_cur, b_fn, t, x, h_use, bdf2_state)
                accepted = True
                result.n_bdf2_steps += 1
            else:
                # DOPRI5 with adaptive PI
                h_use = min(h_rk45, self.dt_max, t_gate - t)
                if h_use <= 0:
                    if t_gate < t_end:
                        prev_choice = choice
                        self._fire_gate_event(
                            t_gate, rk_state, bdf2_state,
                            prev_choice, result)
                        t = t_gate
                        choice = self.detector.select(
                            self._mode_id(self.system.current_mask()),
                            self.system.A_matrix(), self.dt_max)
                        if prev_choice == IntegratorChoice.BDF2 \
                            and choice == IntegratorChoice.DOPRI5:
                            h_rk45 = self.dt_init
                    continue
                x_new, err = rk45.step(f, t, x, h_use, rk_state)
                accepted, h_next = self.controller.accept(
                    err, x, x_new, h_use)
                if not accepted:
                    rk_state.invalidate()
                    h_rk45 = h_next
                    result.n_reject += 1
                    continue
                h_rk45 = h_next
                result.n_rk45_steps += 1

            # 3. Commit accepted step
            t += h_use
            x = x_new
            result.n_accept += 1

            # 4. Did we land on a gate edge?
            if t_gate < t_end and abs(t - t_gate) < kEpsEdge:
                prev_choice = choice
                self._fire_gate_event(
                    t_gate, rk_state, bdf2_state, prev_choice, result)
                choice = self.detector.select(
                    self._mode_id(self.system.current_mask()),
                    self.system.A_matrix(), self.dt_max)
                if prev_choice == IntegratorChoice.BDF2 \
                    and choice == IntegratorChoice.DOPRI5:
                    h_rk45 = self.dt_init

            # 5. Record
            step_idx += 1
            if step_idx % self.store_every == 0:
                times_list.append(t)
                states_list.append(x.copy())

        if times_list[-1] != t:
            times_list.append(t)
            states_list.append(x.copy())

        result.times = np.asarray(times_list)
        result.states = np.asarray(states_list)
        result.cpu_time = time.perf_counter() - t0_wall
        return result

    # ------------------------------------------------------------------

    def _next_gate_edge(self, t_now: float, t_end: float) -> float:
        next_edge_fn = getattr(self.switch_fn, "next_edge_after", None)
        if next_edge_fn is None:
            return t_end
        return min(float(next_edge_fn(t_now)), t_end)

    @staticmethod
    def _mode_id(mask: Any) -> int:
        """Convert a mask to a hashable int (for the per-mode cache).

        Works with bool, int, SwitchStateMask, and any object with
        a usable ``hash()`` — falls back to id() for arbitrary types.
        """
        if isinstance(mask, bool):
            return int(mask)
        if isinstance(mask, int):
            return mask
        try:
            return hash(mask)
        except TypeError:
            return id(mask)

    def _fire_gate_event(
        self,
        t_event: float,
        rk_state: rk45.RK45State,
        bdf2_state: BDF2State,
        segment_integrator: IntegratorChoice,
        result: PEDResultAuto,
    ) -> None:
        old_mask = self.system.current_mask
        if callable(old_mask):
            old_mask = old_mask()
        new_mask = self.switch_fn(t_event + 1e-15)
        if new_mask == old_mask:
            return  # spurious

        self.system.set_mask(new_mask)
        # Discontinuous f → invalidate both integrators' caches.
        rk_state.invalidate()
        bdf2_state.invalidate()
        self.controller.reset()

        result.event_log.append(AutoDispatchEventRecord(
            t=t_event,
            name="gate_edge",
            predicate_type="gate",
            old_mask=old_mask,
            new_mask=new_mask,
            integrator_used=segment_integrator,
        ))
        result.n_events += 1
