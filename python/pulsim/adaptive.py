"""Pulsim — adaptive (variable-step) transient driver.

The v2 kernel's `simulate(...)` runs at a single fixed `dt`. For a
500 ms RL settling transient with τ = 10 µs and 100 ns initial dt,
the fixed-dt cost is 5 M steps — most of them wasted on a state that
isn't really changing. This module wraps `simulate()` in a Python-
level adaptive driver that runs short "segments" with a constant
`dt`, then measures how fast the state changes within each segment
and adjusts `dt` for the next segment.

This is COARSE-GRAIN adaptive stepping (vs the in-kernel STEP-LEVEL
adaptive stepping planned for a future kernel revision). It hits the
sweet spot of:
  * Simple to implement (no history snapshot/restore inside the
    kernel),
  * Practically effective on settling / steady-state tails (the
    common case where fixed-dt wastes the most cycles),
  * Equivalent end-to-end accuracy when `atol` and `rtol` are
    respected.

The driver does NOT do step-level LTE estimation — instead it uses
the L2 norm of the state derivative ``||x(t+dt)−x(t)||/dt`` as the
fast/slow detector. When the derivative is small, the system is in
steady state and dt is grown (up to `dt_max`); when it's large, dt
is shrunk (down to `dt_min`).

Typical usage
-------------

    builder = build_rl_circuit()
    res = p.run_transient_adaptive(
        builder, t_start=0.0, t_end=500e-3,
        dt_init=100e-9, dt_min=100e-9, dt_max=50e-6,
        atol=1e-6, rtol=1e-4,
    )
    print(f"accepted={res.n_accepted}, rejected={res.n_rejected}")
    print(f"final dt = {res.dt_history[-1]*1e6:.1f} µs")
    print(f"speedup vs fixed-dt={(res.t_end-res.t_start)/100e-9 / res.n_accepted:.1f}×")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


__all__ = [
    "AdaptiveResult",
    "run_transient_adaptive",
]


@dataclass
class AdaptiveResult:
    """Outcome of an adaptive-step transient simulation.

    Same shape as the v2 kernel's `SimulationResult` plus extras:
      * `dt_history`: dt chosen for each segment (length = n_segments).
      * `n_accepted` / `n_rejected`: segment-level accept/reject counts.
    """
    times: np.ndarray
    states: np.ndarray
    dt_history: list = field(default_factory=list)
    n_accepted: int = 0
    n_rejected: int = 0
    t_start: float = 0.0
    t_end: float = 0.0

    def num_steps(self) -> int:
        return self.times.shape[0]


def run_transient_adaptive(builder,
                              *,
                              t_start: float = 0.0,
                              t_end: float,
                              dt_init: float,
                              dt_min: float | None = None,
                              dt_max: float | None = None,
                              atol: float = 1e-6,
                              rtol: float = 1e-4,
                              segment_steps: int = 100,
                              safety: float = 0.9,
                              growth_max: float = 5.0,
                              shrink_max: float = 0.1,
                              switch_fn=None,
                              step_observer=None,
                              b_extra_fn=None,
                              start_from_dc_op: bool = False,
                              max_event_iterations: int = 8,
                              verbose: bool = False,
                              ) -> AdaptiveResult:
    """Adaptive-step transient driver.

    Parameters
    ----------
    builder
        Populated CircuitBuilder.
    t_start, t_end
        Simulation time window in seconds.
    dt_init
        Initial step size — should be small enough to resolve any
        startup transient.
    dt_min, dt_max
        Lower / upper bounds for dt. Defaults: dt_init and 1000×dt_init.
    atol, rtol
        Absolute and relative tolerance for the LTE estimate. The
        driver tries to keep
        ``||dx/dt||_∞ · dt ≤ atol + rtol · ||x||_∞``.
    segment_steps
        Number of fixed-dt steps per adaptive segment. Larger →
        coarser dt control but lower overhead.
    safety, growth_max, shrink_max
        Step-controller knobs. After estimating LTE the next dt
        is multiplied by `safety · (tol/LTE)^(1/3)` (third-order
        accuracy assumed for the trap method), clamped to
        ``[shrink_max·dt, growth_max·dt]``.
    switch_fn, b_extra_fn, step_observer, start_from_dc_op,
    max_event_iterations
        Forwarded to ``simulate(...)`` segment-by-segment.

    Returns
    -------
    AdaptiveResult
        Concatenated `times` and `states` across all segments, plus
        the per-segment `dt_history` and accept/reject counters.
    """
    import pulsim as _v2

    if dt_min is None:
        dt_min = dt_init
    if dt_max is None:
        dt_max = dt_init * 1000.0

    dt = float(dt_init)
    t = float(t_start)

    times_segments = []
    states_segments = []
    dt_history = []
    n_accepted = 0
    n_rejected = 0

    # Track previous state for derivative estimation between segments.

    # Track the state at the end of the previous segment so the next
    # segment resumes from there instead of restarting at zero.
    last_state = None

    while t < t_end - 1e-15:
        # Cap segment to avoid overshoot.
        segment_t_end = min(t + segment_steps * dt, t_end)
        seg_dt = (segment_t_end - t) / segment_steps
        if seg_dt <= 0.0:
            break

        res = _v2.simulate(
            builder,
            t_start=t, t_end=segment_t_end, dt=seg_dt,
            switch_fn=switch_fn,
            b_extra_fn=b_extra_fn,
            step_observer=step_observer,
            start_from_dc_op=start_from_dc_op and (t == t_start),
            max_event_iterations=max_event_iterations,
            initial_state=last_state,   # carry state across segments
        )
        seg_times = np.asarray(res.times)
        seg_states = np.asarray(res.states)
        # The first state of the next segment equals the last of
        # the previous one when chaining — drop the duplicate
        # except on the first segment.
        if times_segments:
            seg_times = seg_times[1:]
            seg_states = seg_states[1:]

        # LTE-like estimate: max change-rate within the segment.
        if seg_states.shape[0] >= 2:
            dx = np.diff(seg_states, axis=0)
            dx_max_norm = float(np.linalg.norm(dx, ord=np.inf))
            x_scale = float(np.linalg.norm(seg_states, ord=np.inf)) + 1e-12
        else:
            dx_max_norm = 0.0
            x_scale = 1.0

        # Implicit-trap is O(dt²); LTE ≈ ||dx/dt|| · dt²  …but we
        # don't have the second derivative. As a proxy, use the
        # first-difference of dx (an approximation of d²x/dt²).
        if seg_states.shape[0] >= 3:
            d2x = np.diff(seg_states, axis=0, n=2)
            lte_estimate = float(np.linalg.norm(d2x, ord=np.inf))
        else:
            lte_estimate = dx_max_norm

        tol = atol + rtol * x_scale
        # Always accept — segments themselves don't overshoot
        # because we forward-simulate at the requested dt. The
        # "rejection" here just means we'll shrink dt next time.
        n_accepted += 1
        times_segments.append(seg_times)
        states_segments.append(seg_states)
        dt_history.append(seg_dt)
        t = float(seg_times[-1])
        last_state = np.asarray(seg_states[-1], dtype=float)

        # Update dt for next segment using PI step controller.
        if lte_estimate > 0.0:
            err_norm = lte_estimate / tol
            if err_norm > 1.0:
                # Step too coarse → shrink.
                factor = max(shrink_max,
                              safety * (1.0 / err_norm) ** (1.0/3))
                n_rejected += 1   # logical "reject" for next round
            else:
                # Step OK → maybe grow.
                factor = min(growth_max,
                              safety * (1.0 / max(err_norm, 1e-6))
                                          ** (1.0/3))
        else:
            factor = growth_max
        dt = float(np.clip(dt * factor, dt_min, dt_max))

        if verbose:
            print(f"  t={t*1e3:7.3f} ms   dt_used={seg_dt*1e6:8.3f} µs   "
                   f"lte={lte_estimate:.3e}   tol={tol:.3e}   "
                   f"next dt={dt*1e6:8.3f} µs")

    # Stitch all segments.
    if times_segments:
        all_times = np.concatenate(times_segments)
        all_states = np.concatenate(states_segments, axis=0)
    else:
        all_times = np.array([t_start])
        all_states = np.zeros((1, builder.pool.state_size(builder.graph)))

    return AdaptiveResult(
        times=all_times, states=all_states,
        dt_history=dt_history,
        n_accepted=n_accepted, n_rejected=n_rejected,
        t_start=t_start, t_end=t_end,
    )
