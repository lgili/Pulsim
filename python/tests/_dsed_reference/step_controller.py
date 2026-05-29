"""PI step-size controller for adaptive ODE integration.

Reference: Söderlind, *Automatic Control and Adaptive Time-Stepping*,
Numerical Algorithms 31 (2002) §4, with gains tuned for order-5
DOPRI5 (Table 4): kP=0.7, kI=0.3.

This is a textbook PI controller acting on the per-step local-
truncation-error estimate. It replaces the cruder I-only controller
(``h_new = h · (tol/err)^(1/p)``) by adding memory of the previous
step's error — eliminates the oscillation that pure-I exhibits at
mode boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PIController:
    """PI step controller (Söderlind 2002).

    Defaults are tuned for DOPRI5 (order 5). For BDF2 use
    ``kP=0.4, kI=0.3`` (Söderlind Table 5, H211b for stiff).
    """

    rtol: float = 1e-6
    atol: float = 1e-9
    kP: float = 0.7
    kI: float = 0.3
    rho_min: float = 0.1
    rho_max: float = 5.0
    safety: float = 0.9
    max_rejects: int = 5

    err_prev: float = 1.0
    reject_streak: int = 0

    def err_norm(
        self,
        err: np.ndarray,
        x_old: np.ndarray,
        x_new: np.ndarray,
    ) -> float:
        """Componentwise weighted RMS error norm (Hairer/Wanner §IV.8.4).

        ``sc_i = atol + rtol · max(|x_old_i|, |x_new_i|)``,
        ``||err|| = sqrt(mean((err_i / sc_i)²))``.

        With this normalisation, ``tol = 1`` always and acceptance is
        ``||err|| <= 1``.
        """
        sc = self.atol + self.rtol * np.maximum(np.abs(x_old), np.abs(x_new))
        scaled = err / sc
        return float(np.sqrt(np.mean(scaled * scaled)))

    def accept(
        self,
        err: np.ndarray,
        x_old: np.ndarray,
        x_new: np.ndarray,
        h: float,
    ) -> tuple[bool, float]:
        """Decide whether to accept the step + compute next step size.

        Returns
        -------
        accepted : bool
            True if the step's error was within tolerance.
        h_next : float
            Recommended next step size. On rejection, the caller
            should retry the SAME ``t`` with this smaller ``h``.

        Raises
        ------
        RuntimeError
            On more than ``max_rejects`` consecutive rejections —
            indicates a fundamental problem (stiffness regime change,
            discontinuity inside the step, etc.).
        """
        e = self.err_norm(err, x_old, x_new)

        if e <= 1.0:
            # Accept — compute next h via PI formula
            # Clamp e away from zero to avoid div by zero
            e_clamped = max(e, 1e-12)
            ratio = (1.0 / e_clamped) ** self.kP * (self.err_prev / e_clamped) ** self.kI
            ratio = float(np.clip(ratio, self.rho_min, self.rho_max))
            h_next = self.safety * ratio * h
            # Wind-up protection: clamp err_prev to a reasonable range
            self.err_prev = float(np.clip(e, 0.1, 1.0)) if e > 0 else 0.1
            self.reject_streak = 0
            return True, h_next

        # Reject — shrink h, retry same t
        self.reject_streak += 1
        if self.reject_streak > self.max_rejects:
            raise RuntimeError(
                f"PI controller: {self.reject_streak} consecutive rejections; "
                f"last err={e:.3e}, h={h:.3e}"
            )
        # Pure-I shrink (no kI on rejection to avoid wind-up).
        # Formula: h_new = safety · (1/e)^kP · h  — Hairer & Wanner
        # Solving ODE I, §II.4 eq. (4.13). Bug fix vs earlier ``/safety``:
        # for e only slightly > 1 the divide-by-safety inflated the
        # ratio above 1, clamped to 1, and h NEVER shrank → infinite
        # reject loop until max_rejects fired. Surfaced in Gate 4.D
        # auto-dispatch transitions where mode changes can push err
        # just above 1.
        ratio = self.safety * (1.0 / e) ** self.kP
        ratio = float(np.clip(ratio, self.rho_min, 1.0))  # shrink only
        return False, ratio * h

    def reset(self) -> None:
        """Reset PI memory. Call on topology change to clear I-term wind-up."""
        self.err_prev = 1.0
        self.reject_streak = 0
