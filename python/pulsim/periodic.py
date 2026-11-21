"""Pulsim v2 — ``run_periodic_shooting``, the v1-compatible front end.

Deprecated: it delegates to :func:`pulsim.periodic_steady_state` with
``engine='pwl'`` (or ``'trbdf2'``), add-periodic-steady-state-shooting
task 5.3. The v1 algorithm it replaced restarted every period from the
whole MNA vector by finite differences and converged to the wrong orbit
on a DCM buck (8.8998 V against 8.996247 V, the trapezoidal history of
the inductor carried across the section); the delegate shoots on the
true state with the fixed engine's restart at the section, so the orbit
is the one the engine's own transient reaches.
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
        ``max|P(x*) - x*|`` on the true state at the orbit.
    residual_history
        ``||G(x_k)||_∞`` at each Newton step.
    n_simulations
        Total one-period simulations executed. The cost meter.
    message
        Why it did not converge (empty when it did).
    """
    x_steady_state: np.ndarray
    converged: bool
    n_iterations: int
    final_residual: float
    residual_history: List[float] = field(default_factory=list)
    n_simulations: int = 0
    message: str = ""


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
    """Deprecated: use ``periodic_steady_state(builder, period=T,
    switch_fn=..., engine='pwl', dt=dt)``, which this calls.

    ``x_initial`` and ``x_steady_state`` are full MNA vectors (as
    before); ``x_steady_state`` is the engine's vector at the section
    after one period from the orbit. ``engine`` in ``simulate_kwargs``
    may be ``'auto'``/``'pwl'`` (the fixed engine, the default) or
    ``'trbdf2'``; the other keywords go to every period run.
    ``b_extra_fn`` and ``step_observer`` are refused by name (a hidden
    controller state is not part of the periodic state). ``tol``,
    ``max_iterations`` and ``fd_step`` are accepted and ignored: the
    delegate stops on its own scaled-norm rule within 60 period runs and
    sizes its own differences. Not converged: ``converged=False``,
    ``x_steady_state`` all NaN and the reason in ``message``.
    """
    import warnings

    import pulsim as _v2  # late import (circular)
    from .periodic_steady_state import ConvergenceError, _unsup

    warnings.warn(
        "run_periodic_shooting is deprecated; use pulsim.periodic_"
        "steady_state(..., engine='pwl', dt=dt), which it calls.",
        DeprecationWarning, stacklevel=2)
    if t_period <= 0:
        raise ValueError("t_period must be > 0")
    if dt <= 0:
        raise ValueError("dt must be > 0")
    for name, v in (("b_extra_fn", b_extra_fn),
                    ("step_observer", step_observer)):
        if v is not None:
            raise _unsup(name, f"run_periodic_shooting: {name} is a "
                         "Python callback whose state is not part of the "
                         "periodic state; it is no longer supported. Use "
                         "simulate() for closed-loop runs.")
    engine = simulate_kwargs.pop("engine", "auto")
    if engine not in ("auto", "pwl", "trbdf2"):
        raise _unsup("engine", f"run_periodic_shooting: engine={engine!r};"
                     " use periodic_steady_state(engine='exact') for the "
                     "exact stepper.")
    engine = "trbdf2" if engine == "trbdf2" else "pwl"
    state_size = builder.pool.state_size(builder.graph)
    if state_size == 0:
        raise ValueError(
            "Circuit has no state variables — periodic shooting "
            "has nothing to solve.")
    start = None
    if x_initial is not None:
        x_full = np.asarray(x_initial, dtype=float).reshape(-1)
        if x_full.size != state_size:
            raise ValueError(
                f"x_initial has {x_full.size} entries; expected "
                f"{state_size}")
        from ._transient_shooting import TransientMap
        start = TransientMap(builder, t_period, switch_fn, engine, dt,
                             simulate_kwargs).Cm @ x_full
    try:
        r = _v2.periodic_steady_state(
            builder, period=t_period, switch_fn=switch_fn, engine=engine,
            dt=dt, options=simulate_kwargs, start_from=start)
    except ConvergenceError as e:
        return PeriodicShootingResult(
            x_steady_state=np.full(state_size, np.nan), converged=False,
            n_iterations=0, final_residual=float("nan"), message=str(e))
    if verbose:
        print(f"  shoot: {r.n_period_runs} period runs, residuals "
              f"{r.residual_history}")
    return PeriodicShootingResult(
        x_steady_state=np.asarray(r.full_state, dtype=float),
        converged=True,
        n_iterations=len(r.residual_history),
        final_residual=r.residual,
        residual_history=list(r.residual_history),
        n_simulations=r.n_period_runs,
    )
