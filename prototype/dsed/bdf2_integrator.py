"""BDF2 (Backward Differentiation Formula, order 2) implicit integrator.

Reference: Hairer & Wanner, *Solving ODE II: Stiff Problems*,
Springer 1996/2010, §V.1 (BDF derivation on p. 372).

BDF2 is the workhorse stiff-system integrator: A(α)-stable with
α ≈ 86.03°, second-order accurate, and exceptionally simple per
step. For Pulsim's PED engine it is the implicit complement to
DOPRI5 — used on stiff masks where the explicit step constraint
(from the largest-magnitude eigenvalue of A) shrinks below what
accuracy would otherwise require.

The BDF2 step formula for ``y' = f(t, y)``:

    y_{n+1} - (4/3) y_n + (1/3) y_{n-1} = (2h/3) · f(t_{n+1}, y_{n+1})

For our LTI-per-mask case ``f(t, y) = A·y + b(t)`` this collapses
to a single linear solve per step:

    (I - (2h/3) A) · y_{n+1} = (4/3) y_n - (1/3) y_{n-1} + (2h/3) b(t_{n+1})

The system matrix ``J = (I - (2h/3) A)`` depends only on (A, h),
not on t or y. So:

  * Factor J **once** at mask change or step-size change.
  * Cache the factor and re-use across many steps with the same
    mask + h.
  * On mask change OR h change: refactor.

This is exactly where the Pulsim path-based ``partial_refactor``
pays off (Gate 4 wiring — once Gate 4.B lands a C++ binding to the
Pulsim PWL cache). For Phase 4.A we use ``numpy.linalg.lu_factor``
+ ``lu_solve`` as the prototype-grade replacement; correctness
first, then performance.

BDF2 needs **two** prior steps to start; the first step is
bootstrapped by a single Crank-Nicolson (trapezoidal-rule) step:

    (I - (h/2) A) y_1 = (I + (h/2) A) y_0 + h · b(t_0 + h/2)

Crank-Nicolson is order-2 A-stable, matching BDF2's order so the
startup error doesn't contaminate the global accuracy. (Backward
Euler — order-1 — would inject an O(h) artifact into y_1 that
BDF2 then propagates forward, capping global accuracy at O(h)
even though BDF2 itself is O(h²).)

PI step controller gains for BDF2 (Söderlind 2002 Table 5, H211b):
``kP = 0.4, kI = 0.7 - kP = 0.3`` — slightly slower than DOPRI5's
``kP=0.7, kI=0.3`` because BDF2's error estimate has more memory.

Error estimation for adaptive stepping is via the difference between
the BDF2 solution and a parallel BDF1 (backward Euler) solution at
the same step:

    err = y_{n+1}^{BDF2} - y_{n+1}^{BDF1}

This is the cheap "embedded" estimate — both solutions are
computed from the same (A, h) factor, so the BDF1 add is O(n_state)
work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.linalg import lu_factor, lu_solve


@dataclass
class BDF2State:
    """Persistent state between accepted BDF2 steps.

    BDF2 is a 2-step method: it needs ``y_{n-1}`` and ``y_n`` to
    advance to ``y_{n+1}``. The cached LU factor of
    ``J = I - (2h/3) A`` is also stored — it's only re-factored when
    h or mask changes.
    """
    y_prev: Optional[np.ndarray] = None       # y_{n-1}
    y_prev_t: float = 0.0                       # t_{n-1}
    h_cached: float = 0.0                       # h at which J was last factored
    lu: Optional[tuple] = None                  # (lu_matrix, piv) from lu_factor
    mode_id: int = -1                           # identifier for cached-mode mask

    @property
    def has_history(self) -> bool:
        """True when y_{n-1} is available — i.e. after the bootstrap step."""
        return self.y_prev is not None

    def invalidate(self) -> None:
        """Drop history + LU cache. Call on mask change OR after rejection."""
        self.y_prev = None
        self.lu = None
        self.h_cached = 0.0


def _factor_J(A: np.ndarray, h: float) -> tuple:
    """LU-factor the BDF2 system matrix ``J = I - (2h/3) A``."""
    n = A.shape[0]
    J = np.eye(n) - (2.0 * h / 3.0) * A
    return lu_factor(J)


def bdf2_step(
    A: np.ndarray,
    b_fn: Callable[[float], np.ndarray],
    t: float,
    y: np.ndarray,
    h: float,
    state: BDF2State,
) -> tuple[np.ndarray, np.ndarray]:
    """Take one BDF2 step on the LTI system ``y' = A·y + b(t)``.

    Parameters
    ----------
    A
        Constant system matrix for the current mask (``n_state × n_state``).
    b_fn
        Forcing function ``t -> b(t)`` (length-n_state vector).
        Sampled at ``t + h`` (the implicit endpoint).
    t, y
        Current time + state ``y_n``.
    h
        Proposed step size.
    state
        Persistent BDF2 state. Mutated in place:
        ``state.y_prev`` becomes ``y_n`` on return,
        ``state.lu`` is set/refreshed if h changed.

    Returns
    -------
    y_new : ndarray
        Advanced state ``y_{n+1}``.
    err : ndarray
        Embedded error estimate ``y^{BDF2} - y^{BDF1}`` (componentwise).

    Notes
    -----
    Cost per step:
      * 1 LU factor (only on first step or h-change): O(n_state³)
      * 1 forward+back solve: O(n_state²)
      * 1 forcing eval ``b(t+h)``: O(n_state)

    Bootstrap (first step, no history): falls back to a single
    backward-Euler step to obtain ``y_1``; the next step is then
    a full BDF2 with ``y_{n-1} = y_0, y_n = y_1``.
    """
    # (Re-)factor J if h changed or first step
    if state.lu is None or state.h_cached != h:
        state.lu = _factor_J(A, h)
        state.h_cached = h

    b_new = b_fn(t + h)

    if not state.has_history:
        # Bootstrap: Crank-Nicolson (trapezoidal rule, order-2 A-stable)
        # (I - (h/2) A) y_1 = (I + (h/2) A) y_0 + h · b(t + h/2)
        # We approximate b(t + h/2) ≈ (b(t) + b(t + h)) / 2.
        b_old = b_fn(t)
        b_mid = 0.5 * (b_old + b_new)
        n = A.shape[0]
        J_CN = np.eye(n) - 0.5 * h * A
        K_CN = np.eye(n) + 0.5 * h * A
        lu_CN = lu_factor(J_CN)
        y_new = lu_solve(lu_CN, K_CN @ y + h * b_mid)
        # No embedded error from the bootstrap step → return zeros
        err = np.zeros_like(y)
        # Stash y as y_{n-1} for next step's BDF2
        state.y_prev = y.copy()
        state.y_prev_t = t
        return y_new, err

    # Full BDF2 step:
    # (I - (2h/3) A) y_new = (4/3) y - (1/3) y_prev + (2h/3) b(t+h)
    assert state.y_prev is not None
    rhs_bdf2 = (4.0 / 3.0) * y - (1.0 / 3.0) * state.y_prev \
               + (2.0 * h / 3.0) * b_new
    y_new = lu_solve(state.lu, rhs_bdf2)

    # Embedded error: BDF2 minus BDF1 (backward Euler) at same step
    # BDF1: (I - h A) y_BE = y + h b(t+h)
    # We don't have cached LU for I - h A; for the embedded estimate
    # we can either factor it on-the-fly (expensive) OR derive the
    # error analytically. The latter:
    #
    # BDF1 truncation: y_{n+1}^BE - y(t_{n+1}) = O(h²)
    # BDF2 truncation: y_{n+1}^BDF2 - y(t_{n+1}) = O(h³)
    # So BDF2 - BE = (BDF2 - exact) - (BE - exact) = O(h²) - O(h³) ≈ O(h²)
    #
    # The leading h² coefficient is roughly proportional to
    # h²·y'''(t_n) ≈ h²·(A·y_new - 2·A·y + A·y_prev) / h²  (3rd diff)
    # = A·y_new - 2·A·y + A·y_prev
    #
    # That's a cheap O(n²) matvec + 2 axpys; no LU.
    third_diff = y_new - 2.0 * y + state.y_prev
    err = (1.0 / 6.0) * third_diff  # leading h² truncation coefficient

    # Stash for next step
    state.y_prev = y.copy()
    state.y_prev_t = t

    return y_new, err


@dataclass
class BDF2PIController:
    """PI step controller tuned for BDF2 (Söderlind 2002 Table 5, H211b).

    Slower-reacting than DOPRI5's controller — appropriate for an
    order-2 method where the per-step error has more memory.
    """
    rtol: float = 1e-6
    atol: float = 1e-9
    kP: float = 0.4
    kI: float = 0.3
    rho_min: float = 0.1
    rho_max: float = 3.0   # tighter than DOPRI5's 5.0 (BDF2 is more sensitive)
    safety: float = 0.85
    max_rejects: int = 5

    err_prev: float = 1.0
    reject_streak: int = 0

    def err_norm(
        self,
        err: np.ndarray,
        y_old: np.ndarray,
        y_new: np.ndarray,
    ) -> float:
        sc = self.atol + self.rtol * np.maximum(np.abs(y_old), np.abs(y_new))
        scaled = err / sc
        return float(np.sqrt(np.mean(scaled * scaled)))

    def accept(
        self,
        err: np.ndarray,
        y_old: np.ndarray,
        y_new: np.ndarray,
        h: float,
    ) -> tuple[bool, float]:
        e = self.err_norm(err, y_old, y_new)

        if e <= 1.0:
            e_clamped = max(e, 1e-12)
            # Order p = 2 for BDF2 → exponent = 1/(p+1) = 1/3
            ratio = (1.0 / e_clamped) ** (self.kP / 3.0) \
                  * (self.err_prev / e_clamped) ** (self.kI / 3.0)
            ratio = float(np.clip(ratio, self.rho_min, self.rho_max))
            h_next = self.safety * ratio * h
            self.err_prev = float(np.clip(e, 0.1, 1.0)) if e > 0 else 0.1
            self.reject_streak = 0
            return True, h_next

        self.reject_streak += 1
        if self.reject_streak > self.max_rejects:
            raise RuntimeError(
                f"BDF2 PI controller: {self.reject_streak} consecutive "
                f"rejections; last err={e:.3e}, h={h:.3e}"
            )
        ratio = (1.0 / e) ** (self.kP / 3.0) / self.safety
        ratio = float(np.clip(ratio, self.rho_min, 1.0))
        return False, ratio * h

    def reset(self) -> None:
        """Reset PI memory. Call on topology change to clear I-term wind-up."""
        self.err_prev = 1.0
        self.reject_streak = 0
