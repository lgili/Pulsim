"""Pulsim v2 — periodic steady-state shooting solver.

Restores v1's ``run_periodic_shooting`` capability. Finds the
steady-state cycle of a periodically driven nonlinear circuit
without waiting for the natural transient to settle.

Algorithm (single-shooting Newton)
----------------------------------
Given a system ``dx/dt = f(x, t)`` with period T (PWM period, line
period, etc.), the steady-state initial condition ``x*`` satisfies

    G(x*) = φ(T, x*) - x* = 0,

where ``φ(T, x)`` integrates one full period starting from ``x``.

Newton iteration::

    x_{k+1} = x_k - J(x_k)^{-1} · G(x_k)

The Jacobian ``J = ∂φ/∂x - I`` is built by finite differences —
perturb each component of ``x_k`` by ``ε``, simulate one period,
take the finite-difference column. That costs ``(N+1)`` one-period
simulations per Newton iteration, where ``N`` is the state size.

For a buck/boost/flyback with ≤ 6 state variables, convergence in
3-6 Newton iterations is typical → 30-50 one-period simulations
to reach steady state. For comparison, the natural transient on a
poorly damped converter can take **thousands** of periods to
settle. Net speedup: typically **30-100×** in wall time.

When to use
-----------
* Steady-state efficiency analysis (η as function of duty / load).
* THD / harmonic analysis on the converter's "operating point".
* Bode / frequency-response sweeps that need the steady-state
  small-signal model.
* DC operating point of a switching converter (no closed-form
  exists; periodic shooting IS the operating point).

When NOT to use
---------------
* Transient response studies (turn-on, step load, fault) — that's
  what :func:`simulate` is for.
* Systems with multiple stable limit cycles (shooting finds one;
  which one depends on the initial guess).
* Pre-converged DC operating point: use :func:`compute_dc_op` for
  that — it's faster on linear circuits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np


__all__ = [
    "PeriodicShootingResult",
    "run_periodic_shooting",
]


@dataclass
class PeriodicShootingResult:
    """Output of :func:`run_periodic_shooting`.

    Attributes
    ----------
    x_steady_state
        The steady-state initial condition ``x*`` (vector,
        shape ``(state_size,)``).
    converged
        True if the Newton iteration met the convergence tolerance.
    n_iterations
        Newton steps actually performed.
    final_residual
        Final ``||G(x*)||_∞`` — should be ``≤ tol`` when
        ``converged=True``.
    residual_history
        ``||G(x_k)||_∞`` at each Newton step.
    n_simulations
        Total one-period simulations executed (= n_iterations ×
        (state_size + 1)). The cost meter.
    """
    x_steady_state: np.ndarray
    converged: bool
    n_iterations: int
    final_residual: float
    residual_history: List[float] = field(default_factory=list)
    n_simulations: int = 0


def run_periodic_shooting(builder,
                              *,
                              t_period: float,
                              dt: float,
                              switch_fn: Optional[Callable] = None,
                              b_extra_fn: Optional[Callable] = None,
                              step_observer: Optional[Callable] = None,
                              x_initial: Optional[np.ndarray] = None,
                              tol: float = 1e-6,
                              max_iterations: int = 20,
                              fd_step: float = 1e-6,
                              verbose: bool = False,
                              **simulate_kwargs,
                              ) -> PeriodicShootingResult:
    """Find the periodic steady-state via single-shooting Newton.

    Parameters
    ----------
    builder
        Populated :class:`CircuitBuilder` whose drive is periodic
        with period ``t_period``.
    t_period
        Period of the driving signal (s). For PWM converters this
        is ``1 / f_pwm``; for line-driven circuits ``1 / f_line``.
    dt
        Fixed simulation step (s). Same constraints as for
        :func:`simulate`.
    switch_fn, b_extra_fn, step_observer
        Forwarded to :func:`simulate` for each one-period run.
        **All three must be deterministic functions of time** (i.e.,
        depend only on ``t`` mod ``t_period``). A stateful
        controller breaks the periodicity assumption — use
        :func:`simulate` for that.
    x_initial
        Initial guess for the steady-state ``x*``. Default: zero
        vector of the correct size. Pass a result of
        :func:`compute_dc_op` for a much closer starting point on
        nonlinear circuits.
    tol
        Newton convergence tolerance on ``||G||_∞``. Default 1e-6.
    max_iterations
        Maximum Newton steps. Default 20.
    fd_step
        Finite-difference perturbation magnitude. Default 1e-6.
        Larger when ``x*`` has very large entries (e.g. 800 V cap
        voltages); smaller can underflow.
    verbose
        Print per-iteration residual + Δx info. Useful for
        diagnosing convergence problems.
    **simulate_kwargs
        Additional keyword arguments forwarded to :func:`simulate`
        (``enable_nonlinear_refresh``, ``max_event_iterations``,
        etc.).

    Returns
    -------
    PeriodicShootingResult
        With the converged ``x_steady_state``, plus diagnostics.
    """
    from . import v2 as _v2  # late import (circular)

    if t_period <= 0:
        raise ValueError("t_period must be > 0")
    if dt <= 0:
        raise ValueError("dt must be > 0")

    state_size = builder.pool.state_size(builder.graph)
    if state_size == 0:
        raise ValueError(
            "Circuit has no state variables — periodic shooting "
            "has nothing to solve.")

    # Initial guess.
    if x_initial is None:
        x_k = np.zeros(state_size, dtype=float)
    else:
        x_k = np.asarray(x_initial, dtype=float).copy()
        if x_k.size != state_size:
            raise ValueError(
                f"x_initial has {x_k.size} entries; expected "
                f"{state_size}")

    def phi(x_start: np.ndarray) -> np.ndarray:
        """Integrate one period from x_start. Returns x(T)."""
        res = _v2.simulate(
            builder,
            t_end=t_period, dt=dt,
            switch_fn=switch_fn,
            b_extra_fn=b_extra_fn,
            step_observer=step_observer,
            initial_state=x_start.tolist(),
            **simulate_kwargs)
        last = res.states[-1]
        if hasattr(last, "shape"):
            return np.asarray(last, dtype=float)
        return np.asarray(list(last), dtype=float)

    residuals: List[float] = []
    n_sims = 0
    converged = False

    for k in range(max_iterations):
        # Evaluate G(x_k) = φ(x_k) - x_k.
        x_end = phi(x_k)
        n_sims += 1
        G = x_end - x_k
        res_norm = float(np.max(np.abs(G)))
        residuals.append(res_norm)
        if verbose:
            print(f"  shoot[{k:>2d}] ||G||_∞ = {res_norm:.3e}, "
                  f"||x|| = {np.linalg.norm(x_k):.3e}")
        if res_norm < tol:
            converged = True
            break

        # Build the Jacobian J = ∂φ/∂x - I by forward differences.
        # Column j: (φ(x_k + ε e_j) - x_end) / ε.
        J_phi = np.empty((state_size, state_size), dtype=float)
        for j in range(state_size):
            xj = x_k.copy()
            # Scale ε with |x_j| to avoid loss of precision when
            # x_j is large.
            eps = fd_step * max(1.0, abs(x_k[j]))
            xj[j] += eps
            x_end_j = phi(xj)
            n_sims += 1
            J_phi[:, j] = (x_end_j - x_end) / eps
        J = J_phi - np.eye(state_size)

        # Newton step: solve J · dx = -G.
        try:
            dx = np.linalg.solve(J, -G)
        except np.linalg.LinAlgError:
            # Singular Jacobian — bail out, return what we have.
            if verbose:
                print(f"  shoot[{k}] singular Jacobian, stopping")
            break
        x_k = x_k + dx

    return PeriodicShootingResult(
        x_steady_state=x_k,
        converged=converged,
        n_iterations=len(residuals),
        final_residual=residuals[-1] if residuals else float("inf"),
        residual_history=residuals,
        n_simulations=n_sims,
    )
