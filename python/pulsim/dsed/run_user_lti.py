"""User-supplied LTI escape hatch for the C++ DSED scheduler.

The everyday DSED entry point is :func:`pulsim.simulate` with
``engine='dsed'`` — it extracts the per-mask `(A, b)` state-space
from a :class:`pulsim.CircuitBuilder` automatically. Some workflows
(custom physics models, system-identification benchmarks, FMU
integrations) need to drive the scheduler from a hand-rolled LTI
system that isn't built through :class:`CircuitBuilder`. That's
what this module is for.

The functions here are thin wrappers around the C++ scheduler
symbols (``run_ped_native`` / ``run_bdf2_native`` /
``run_auto_native``) that document the expected ``system`` protocol
and validate it at the call site so the user gets a clear error
instead of a cryptic C++ side-effect.

System protocol
---------------

Pass any object that implements the methods below — typically a
small dataclass or a custom class. All four kinds of method are
required (the C++ scheduler dispatches between RK45 and BDF2 per
mask transparently)::

    class MyLTI:
        def A_matrix(self) -> np.ndarray:
            '''Current state matrix for the active mask (n×n).'''

        def b_vector(self, t: float) -> np.ndarray:
            '''Time-varying forcing for the active mask (length n).'''

        def rhs(self, t: float, x: np.ndarray) -> np.ndarray:
            '''Continuous-time RHS y' = A·x + b(t) (length n).
            Called by the RK45 inner loop.'''

        def current_mask(self) -> int:
            '''Encoded representation of the active switch mask.'''

        def set_mask(self, mask: int) -> None:
            '''Switch mask hand-off from the scheduler.'''

``switch_fn`` is any callable ``t -> mask`` returning the same
``int`` encoding the system protocol uses internally.

Example
-------

::

    import numpy as np
    import pulsim as p

    class SecondOrderLTI:
        def __init__(self):
            self._mask = 0
            self._A = np.array([[0.0, 1.0], [-1.0, -0.1]])
            self._b = np.zeros(2)
        def A_matrix(self): return self._A
        def b_vector(self, t): return self._b
        def rhs(self, t, x): return self._A @ x + self._b
        def current_mask(self): return self._mask
        def set_mask(self, m): self._mask = m

    sys = SecondOrderLTI()
    res = p.dsed.run_user_lti(
        sys,
        switch_fn=lambda t: 0,
        x0=np.array([1.0, 0.0]),
        t_end=10.0,
        rtol=1e-6, atol=1e-9,
    )

    # res is a dict with 'times', 'states', 'n_accept', 'n_reject',
    # 'n_events', 'cpu_time_seconds', and (for 'auto') the
    # per-mask integrator log.

See ``docs/how-pulsim-works/11-dsed-engine-overview.md`` for
algorithm details and ``docs/how-pulsim-works/16-dsed-benchmarks.md``
for performance characteristics on canonical circuits.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from .. import _pulsim as _k

__all__ = ["run_user_lti"]


_REQUIRED_METHODS = (
    "A_matrix",
    "b_vector",
    "rhs",
    "current_mask",
    "set_mask",
)


def _validate_system(system: Any) -> None:
    """Raise ``TypeError`` if ``system`` is missing any protocol
    method. Failing here gives a much clearer error than the
    pybind11 attribute-error you would otherwise hit deep inside
    the C++ scheduler step."""
    missing = [m for m in _REQUIRED_METHODS if not hasattr(system, m)]
    if missing:
        raise TypeError(
            f"pulsim.dsed.run_user_lti: `system` is missing "
            f"required method(s) {missing}. The system protocol "
            f"is documented in `pulsim.dsed.run_user_lti.__doc__`. "
            f"Every system must implement: {list(_REQUIRED_METHODS)}.")


def run_user_lti(
    system: Any,
    switch_fn: Callable[[float], int],
    x0: np.ndarray,
    t_end: float,
    *,
    integrator: str = "auto",
    rtol: float = 1.0e-6,
    atol: float = 1.0e-9,
    dt_init: float = 1.0e-9,
    dt_max: float = 1.0e-5,
    h_bdf2: Optional[float] = None,
    stiffness_threshold: float = 100.0,
    store_every: int = 1,
) -> dict:
    """Run the DSED scheduler over a user-supplied LTI system.

    Parameters
    ----------
    system
        Any object satisfying the system protocol described in
        the module docstring (5 methods: ``A_matrix``, ``b_vector``,
        ``rhs``, ``current_mask``, ``set_mask``).
    switch_fn
        Callable ``t -> mask`` returning the active switch-state
        encoding. Same protocol the C++ scheduler uses when driven
        through ``CircuitBuilder``.
    x0
        Initial state vector (numpy 1-D, dtype float).
    t_end
        End time, in seconds. ``t_start`` is implicitly 0.0.
    integrator
        Selects which native C++ scheduler to run:

        * ``'auto'`` (default) — adaptive RK45 ↔ BDF2 dispatch.
        * ``'rk45'``           — fixed-choice DOPRI5 RK45.
        * ``'bdf2'``           — fixed-choice implicit BDF2.

        Unknown names raise ``ValueError``.
    rtol, atol
        PI step-size controller tolerances. Defaults
        ``rtol=1e-6, atol=1e-9`` match the YAML loader.
    dt_init
        Initial step size in seconds. The scheduler refines this
        from the first error estimate.
    dt_max
        Hard cap on the step size, in seconds. Also applied as
        the upper bound when an event prediction overshoots.
    h_bdf2
        Fixed step size for the BDF2 path (``integrator='bdf2'``
        or ``'auto'``). Defaults to ``dt_max`` if unspecified.
    stiffness_threshold
        Eigenvalue-ratio cutoff for the auto-dispatch heuristic.
        Above this, the scheduler switches RK45 → BDF2. Ignored
        for the non-auto integrators.
    store_every
        Decimate stored output by this factor (``1`` keeps every
        accepted step). Useful for long runs.

    Returns
    -------
    dict
        Native C++ scheduler output:

        * ``times`` : list[float] — accepted time stamps.
        * ``states`` : list[ndarray] — state vector at each stamp.
        * ``n_accept`` / ``n_reject`` — step-controller statistics.
        * ``n_events`` — total event-bisection invocations.
        * ``cpu_time_seconds`` — wall-clock of the scheduler loop.
        * (for ``'auto'``) ``n_rk45_steps``, ``n_bdf2_steps`` and
          ``event_log`` — per-mask integrator history.

    Raises
    ------
    TypeError
        If ``system`` doesn't satisfy the documented protocol.
    ValueError
        On unknown ``integrator`` name.
    """
    _validate_system(system)
    if integrator not in ("auto", "rk45", "bdf2"):
        raise ValueError(
            f"pulsim.dsed.run_user_lti: integrator={integrator!r} "
            f"is not supported. Use one of 'auto' (default), "
            f"'rk45', or 'bdf2'.")

    if h_bdf2 is None:
        h_bdf2 = dt_max

    x0 = np.ascontiguousarray(x0, dtype=float)

    if integrator == "rk45":
        return _k.run_ped_native(
            system, switch_fn, x0, t_end,
            rtol=rtol, atol=atol,
            dt_init=dt_init, dt_max=dt_max,
            store_every=store_every,
        )
    if integrator == "bdf2":
        return _k.run_bdf2_native(
            system, switch_fn, x0, t_end,
            h_fixed=h_bdf2,
            store_every=store_every,
        )
    # 'auto'
    return _k.run_auto_native(
        system, switch_fn, x0, t_end,
        rtol=rtol, atol=atol,
        dt_init=dt_init, dt_max=dt_max,
        h_bdf2=h_bdf2,
        stiffness_threshold=stiffness_threshold,
        store_every=store_every,
    )
