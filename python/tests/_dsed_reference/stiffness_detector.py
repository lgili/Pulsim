"""Stiffness detection: which integrator (DOPRI5 vs BDF2) for a given mask.

A mode's ODE ``y' = A·y + b`` is "stiff" when the explicit-method
step-size constraint (from the largest-magnitude eigenvalue of A,
``λ_max``) is much tighter than the accuracy constraint. Equivalently:

  - DOPRI5 stability region: |λ·h| ≲ 3 (boundary of order-5
    Butcher tableau on the imaginary axis is roughly h·λ ≈ 3i;
    on the negative-real axis the constraint is similar).
  - If |λ_max| · h_target > 10, the explicit method is forced
    to take ~10 steps per accuracy-required step → switch to BDF2
    (A(α)-stable; can take h limited only by accuracy).

For Pulsim's PED engine, ``A`` is constant within a mode. We
compute λ_max once per mode and cache it. The selection is
made at mask-change time:

    A_max = max(|eigenvalues of A_mask|)
    h_explicit_max ≈ 3 / A_max   (DOPRI5 stability)
    h_target = current PI controller's recommended h

    if A_max * h_target > stiffness_threshold:
        use BDF2
    else:
        use DOPRI5

Default threshold of 10 means BDF2 kicks in when the explicit
method is forced to take ≥ 10 steps per accuracy-required step.

For Gate 4 prototype this is intentionally simple — a single
threshold + the largest eigenvalue (computed once per mask via
``np.linalg.eigvals``, O(n³) but n is small for the converters
we care about: ≤ 8 states for any topology in the v1.4.0 suite,
≤ ~30 for MMC N=3).

Future refinements (Gate 5+):
  * Use power-iteration for O(n²) λ_max on large MMC systems
  * Detect stiffness ratio (λ_max / λ_min) for selecting the
    integrator's order, not just RK45 vs BDF2
  * Handle complex eigenvalues correctly (BDF2 has phase error;
    backward DSED variant in Gate 4.3 addresses parasitic ringing)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class IntegratorChoice(Enum):
    """Which integrator the scheduler should use for the current mode."""
    DOPRI5 = "RK45"      # explicit, non-stiff
    BDF2 = "BDF2"        # implicit, stiff


@dataclass
class StiffnessDetector:
    """Pick the right integrator per mode based on |λ_max| · h.

    Attributes
    ----------
    threshold : float
        |λ_max| · h ratio above which BDF2 is preferred.
        Default ``10.0`` per Hairer & Wanner §IV.4.
    cache : dict[Any, float]
        Per-mode-id cache of |λ_max|. Computed lazily on first
        ``select()`` call for a mode.
    """
    threshold: float = 10.0
    cache: dict[Any, float] = field(default_factory=dict)

    def lambda_max(self, mode_id: Any, A: np.ndarray) -> float:
        """Compute (and cache) the largest-magnitude eigenvalue of A.

        Uses ``numpy.linalg.eigvals`` (O(n³)). For our converters
        n is small (≤ 30) so this is negligible; for MMC N≥10 a
        Krylov power-iteration would be cheaper (Gate 5 work).
        """
        if mode_id in self.cache:
            return self.cache[mode_id]
        eigs = np.linalg.eigvals(A)
        lam_max = float(np.max(np.abs(eigs)))
        self.cache[mode_id] = lam_max
        return lam_max

    def select(
        self,
        mode_id: Any,
        A: np.ndarray,
        h: float,
    ) -> IntegratorChoice:
        """Return DOPRI5 or BDF2 for the given (mode, h)."""
        lam_max = self.lambda_max(mode_id, A)
        if lam_max * h > self.threshold:
            return IntegratorChoice.BDF2
        return IntegratorChoice.DOPRI5

    def explain(
        self,
        mode_id: Any,
        A: np.ndarray,
        h: float,
    ) -> dict:
        """Diagnostic: return all relevant numbers for the choice."""
        lam_max = self.lambda_max(mode_id, A)
        ratio = lam_max * h
        return {
            "mode_id": mode_id,
            "lambda_max": lam_max,
            "h": h,
            "lambda_h": ratio,
            "threshold": self.threshold,
            "stiff": ratio > self.threshold,
            "choice": (
                IntegratorChoice.BDF2
                if ratio > self.threshold
                else IntegratorChoice.DOPRI5
            ),
        }

    def clear_cache(self) -> None:
        """Drop all cached eigenvalues. Call after parameter changes."""
        self.cache.clear()
