"""DOPRI5 (Dormand-Prince RK 5(4)) with FSAL.

Reference: Hairer, Norsett, Wanner, *Solving ODE I: Nonstiff Problems*,
Springer 1993/2008, §II.5 (Butcher tableau on p. 178).

7-stage, order 5(4) embedded RK with FSAL (First-Same-As-Last):
the 7th stage of step n equals the 1st stage of step n+1 on
accepted steps, reducing the per-step cost to 6 RHS evaluations.

Used as the default integrator in Pulsim's PED engine (Gate 0
selector rule: events_per_step > 0.2 OR stiffness < 50 → RK45).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


# --- Butcher tableau coefficients (Hairer/Norsett/Wanner 1993, p. 178) ---
_C2, _C3, _C4, _C5, _C6, _C7 = 1/5, 3/10, 4/5, 8/9, 1.0, 1.0

_A21 = 1/5
_A31, _A32 = 3/40, 9/40
_A41, _A42, _A43 = 44/45, -56/15, 32/9
_A51, _A52, _A53, _A54 = 19372/6561, -25360/2187, 64448/6561, -212/729
_A61, _A62, _A63, _A64, _A65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656

# b coefficients (5th-order solution)
_B1, _B3, _B4, _B5, _B6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84

# (b - bhat) coefficients — embedded error estimate
_E1, _E3, _E4, _E5, _E6, _E7 = (
    71/57600,
    -71/16695,
    71/1920,
    -17253/339200,
    22/525,
    -1/40,
)


@dataclass
class RK45State:
    """Persistent FSAL state between accepted DOPRI5 steps.

    On rejection, set `fsal_valid = False` to force k1 recompute.
    On topology change (event), do the same — f discontinuously
    changes so the cached k1 is stale.
    """
    k1: Optional[np.ndarray] = None
    fsal_valid: bool = False

    def invalidate(self) -> None:
        self.fsal_valid = False


def step(
    f: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    x: np.ndarray,
    h: float,
    state: RK45State,
) -> tuple[np.ndarray, np.ndarray]:
    """Take one DOPRI5 step.

    Parameters
    ----------
    f
        RHS function ``f(t, x) -> dx/dt``.
    t
        Current time.
    x
        Current state ``(n_state,)``.
    h
        Proposed step size.
    state
        Persistent FSAL state. Mutated in place: on return, ``state.k1``
        is ``k7`` of this step (= ``k1`` of next step), and
        ``state.fsal_valid`` is True.

    Returns
    -------
    x_new : ndarray
        Advanced state ``x(t + h)`` (5th-order solution).
    err : ndarray
        Embedded error estimate ``h · (b - bhat) · k`` (componentwise).

    Notes
    -----
    Cost: 6 RHS evaluations on accepted steps (FSAL), 7 on rejected.
    """
    # Stage 1: FSAL — reuse k7 from previous accepted step
    if state.fsal_valid and state.k1 is not None:
        k1 = state.k1
    else:
        k1 = f(t, x)

    # Stages 2-7
    k2 = f(t + _C2 * h, x + h * (_A21 * k1))
    k3 = f(t + _C3 * h, x + h * (_A31 * k1 + _A32 * k2))
    k4 = f(t + _C4 * h, x + h * (_A41 * k1 + _A42 * k2 + _A43 * k3))
    k5 = f(t + _C5 * h, x + h * (_A51 * k1 + _A52 * k2 + _A53 * k3 + _A54 * k4))
    k6 = f(t + _C6 * h, x + h * (_A61 * k1 + _A62 * k2 + _A63 * k3 + _A64 * k4 + _A65 * k5))

    # 5th-order solution
    x_new = x + h * (_B1 * k1 + _B3 * k3 + _B4 * k4 + _B5 * k5 + _B6 * k6)

    # FSAL stage — k7 = f(t+h, x_new) → reused as k1 next step
    k7 = f(t + _C7 * h, x_new)

    # Embedded error estimate
    err = h * (
        _E1 * k1 + _E3 * k3 + _E4 * k4 + _E5 * k5 + _E6 * k6 + _E7 * k7
    )

    # Stash for next step (FSAL)
    state.k1 = k7
    state.fsal_valid = True

    return x_new, err


def interpolate(
    f: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    x: np.ndarray,
    h: float,
    theta: float,
    state: RK45State,
) -> np.ndarray:
    """Dense-output interpolation: evaluate x(t + theta·h) for theta ∈ [0, 1].

    Used by the event predictor to evaluate predicates at sub-step
    times without re-integrating. Cubic Hermite is enough for
    DOPRI5's order-5 inside the interval.

    Parameters
    ----------
    f, t, x, h, state
        Same as ``step()`` — must be called BEFORE step() to capture
        current k1; or with state.k1 still pointing at the start-of-step
        derivative.
    theta
        Fractional position in [0, 1].

    Returns
    -------
    x_theta : ndarray
        State at ``t + theta·h``.

    Notes
    -----
    Simple Hermite cubic ``x(τ) = x + h·θ·k1 + θ²(3-2θ)·(x_new - x - h·k1)``
    where ``x_new`` is at θ=1. Cost: 1 extra RHS eval at θ=1 if not
    already available.
    """
    if not state.fsal_valid or state.k1 is None:
        k1 = f(t, x)
    else:
        k1 = state.k1

    # Compute full step's x_new to anchor interpolation
    # (this is wasted if we already did a step, but caller responsible
    #  for caching x_new themselves in that case via a custom impl)
    if theta == 0.0:
        return x
    if theta == 1.0:
        x_new, _ = step(f, t, x, h, state)
        return x_new

    # Cubic Hermite between (t, x, k1) and (t+h, x_new, k7)
    x_new, _ = step(f, t, x, h, state)
    assert state.k1 is not None  # step() just set this
    k7 = state.k1
    theta_sq = theta * theta
    theta_cu = theta_sq * theta
    h00 = 2 * theta_cu - 3 * theta_sq + 1
    h10 = theta_cu - 2 * theta_sq + theta
    h01 = -2 * theta_cu + 3 * theta_sq
    h11 = theta_cu - theta_sq
    return h00 * x + h10 * h * k1 + h01 * x_new + h11 * h * k7
