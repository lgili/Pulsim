"""BDF2 PED scheduler — Python port of ``scheduler_bdf2.hpp`` (Gate 4.C).

Implicit BDF2 variant of the PED scheduler — companion to the
explicit :class:`PEDSimulator` (DOPRI5). Used on stiff masks where
the explicit method's stability constraint shrinks h below what
accuracy alone requires.

Design (mirrors the C++ template):

* Gate-edge fast path via ``switch_fn.next_edge_after(t)``
* No predicate scan / Hermite backtrack (body-diode events are
  RK45-handled — commutation prefers small dt + explicit dynamics)
* Fixed h (cap at gate-edge boundary)
* On mask change: invalidate BDF2 history (y_prev) + LU cache;
  next step uses Crank-Nicolson bootstrap to restart

The ``system`` argument must expose:

* ``A_matrix() -> ndarray``     — current mask's A
* ``b_vector(t) -> ndarray``    — current mask's b(t)
* ``set_mask(mask) -> None``    — swap mask
* ``current_mask()``            — read current mask

The ``switch_fn`` callable must satisfy ``mask = switch_fn(t)``
and ideally expose ``next_edge_after(t) -> float`` for the gate-edge
fast path.

Validated end-to-end in C++ (notes/GATE4_PROGRESS.md): 10.9× speedup
vs DOPRI5-at-stability on a 2-mode stiff RLC switched at 10 kHz.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from .bdf2_integrator import BDF2State, bdf2_step
from .scheduler import EventRecord, PEDResult


class PEDSimulatorBDF2:
    """BDF2-based PED simulator (Gate 4.C port).

    Parameters
    ----------
    system
        Must expose ``A_matrix() / b_vector(t) / set_mask / current_mask``.
    switch_fn
        Callable ``t -> mask``; ideally with ``next_edge_after(t)``.
    h_fixed
        Fixed BDF2 step (capped at gate-edge boundary).
    store_every
        Record every Nth accepted step in the result (default 1 = all).
    """

    def __init__(
        self,
        system: Any,
        switch_fn: Callable[[float], Any],
        h_fixed: float = 1e-6,
        store_every: int = 1,
    ) -> None:
        self.system = system
        self.switch_fn = switch_fn
        self.h_fixed = h_fixed
        self.store_every = store_every

    def simulate(self, x0: np.ndarray, t_end: float) -> PEDResult:
        """Run the BDF2 simulation from ``t=0`` to ``t=t_end``."""
        t0_wall = time.perf_counter()

        x = np.asarray(x0, dtype=float).copy()
        t = 0.0
        mask = self.switch_fn(t)
        self.system.set_mask(mask)

        bdf2_state = BDF2State()
        result = PEDResult()
        times_list = [t]
        states_list = [x.copy()]
        step_idx = 0

        max_steps = 10_000_000

        def b_fn(tau: float) -> np.ndarray:
            return self.system.b_vector(tau)

        while t < t_end:
            if result.n_accept > max_steps:
                raise RuntimeError(
                    f"PEDSimulatorBDF2: exceeded {max_steps} steps "
                    f"at t={t:.6e}/{t_end:.6e}"
                )

            # 1. Next gate-edge time (analytical fast path)
            t_gate = self._next_gate_edge(t, t_end)
            t_target = min(t_gate, t_end)

            # 2. Step size: capped at next gate edge
            h_use = min(self.h_fixed, t_target - t)
            if h_use <= 0:
                # Sitting exactly on a gate edge — fire it.
                if t_gate < t_end:
                    self._fire_gate_event(t_gate, bdf2_state, result)
                    t = t_gate
                continue

            # 3. Pull current A from the system (set by current mask)
            A_cur = self.system.A_matrix()

            # 4. Take BDF2 step
            x_new, _err = bdf2_step(A_cur, b_fn, t, x, h_use, bdf2_state)

            # 5. Commit
            t += h_use
            x = x_new
            result.n_accept += 1

            # 6. Did we land on a gate edge?
            if t_gate < t_end and abs(t - t_gate) < 1e-12:
                self._fire_gate_event(t_gate, bdf2_state, result)

            # 7. Record
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

    def _next_gate_edge(self, t_now: float, t_end: float) -> float:
        next_edge_fn = getattr(self.switch_fn, "next_edge_after", None)
        if next_edge_fn is None:
            return t_end
        return min(float(next_edge_fn(t_now)), t_end)

    def _fire_gate_event(
        self,
        t_event: float,
        bdf2_state: BDF2State,
        result: PEDResult,
    ) -> None:
        old_mask = self.system.current_mask
        if callable(old_mask):
            old_mask = old_mask()
        new_mask = self.switch_fn(t_event + 1e-15)
        if new_mask == old_mask:
            return  # spurious

        self.system.set_mask(new_mask)
        # A and b changed → both history (y_prev) and LU (J = I - (2h/3) A_old)
        # are invalidated; next step starts fresh via Crank-Nicolson bootstrap.
        bdf2_state.invalidate()

        result.event_log.append(EventRecord(
            t=t_event,
            name="gate_edge",
            predicate_type="gate",
            old_mask=old_mask,
            new_mask=new_mask,
        ))
        result.n_events += 1
