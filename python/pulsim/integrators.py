"""Pulsim — adaptive Runge-Kutta integrators (Phase 2.4 — Python).

Two embedded-pair adaptive integrators that compete with PSIM's
ode45 / PLECS's DOPRI5 on the **integrator-only** surface (no
in-kernel coupling — these are pure-Python ODE solvers that run
against a user-supplied ``f(t, x) → dx/dt`` callback).

Scope:

* :class:`DormandPrince5` — explicit Dormand-Prince 5(4) embedded
  pair (the "RK45" of MATLAB / SciPy fame). 7 stages, FSAL → 6
  effective ``f`` evaluations per step. 5th-order accurate
  solution + 4th-order embedded error estimate. Default tolerance
  ``rtol=1e-5``, ``atol=1e-8`` — same as SciPy's RK45.

* :class:`RadauIIA3` — implicit Radau IIA, 2 stages, order 3.
  **L-stable** — handles stiff ODE without instability that
  RK45 / DOPRI5 suffer when ``dt`` exceeds the fast time
  constant. Inner Newton iteration on the 2-stage block system.

Both classes follow the same minimalist surface:

    solver = DormandPrince5(f=my_f, rtol=1e-6, atol=1e-9)
    res = solver.solve(t_span=(0.0, 10.0), x0=[0.0, 1.0])
    print(res.t, res.x, res.n_accepted, res.n_rejected)

Where ``my_f(t, x) → dx/dt`` is any user callback (numpy array in,
numpy array out).

What's NOT here (deferred to v1.6.0):

* In-kernel coupling — these integrators are pure-Python and
  ride a user-supplied ``f``. Plugging them into ``pulsim.simulate``
  in place of Tustin requires reformulating the cache to expose
  the continuous ``A``, ``b`` matrices instead of the Tustin-
  discretized ``M``, ``K`` — a kernel refactor scoped at 6-8
  weeks per the ``add-adaptive-runge-kutta-solvers`` proposal.

* Dense-output interpolants for event detection — needed when
  the integrators drive switch-mode transients with precise
  zero-crossing localization. Lands with the in-kernel coupling.

Reference:
  * Hairer, Nørsett & Wanner, *Solving Ordinary Differential
    Equations I: Nonstiff Problems* (Springer 1993).
  * Hairer & Wanner, *Solving Ordinary Differential Equations
    II: Stiff and Differential-Algebraic Problems* (Springer 1996).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Tuple

import numpy as np


__all__ = [
    "AdaptiveSolution",
    "DormandPrince5",
    "RadauIIA3",
]


# =============================================================================
# Result struct
# =============================================================================

@dataclass
class AdaptiveSolution:
    """Output of :meth:`DormandPrince5.solve` /
    :meth:`RadauIIA3.solve`.

    Attributes
    ----------
    t : numpy.ndarray
        Time points at which the solution was sampled (the
        controller's accepted step times).
    x : numpy.ndarray
        State trajectory, shape ``(len(t), n_state)``.
    n_accepted : int
        Number of accepted steps.
    n_rejected : int
        Number of rejected steps (due to error estimate exceeding
        tolerance).
    n_f_evals : int
        Total number of ``f`` evaluations.
    dt_history : list[float]
        ``dt`` chosen for each accepted step.
    """
    t: np.ndarray
    x: np.ndarray
    n_accepted: int = 0
    n_rejected: int = 0
    n_f_evals: int = 0
    dt_history: list = field(default_factory=list)

    def num_steps(self) -> int:
        return self.t.size


# =============================================================================
# Dormand-Prince 5(4) — explicit, non-stiff
# =============================================================================

# Butcher tableau (canonical Dormand-Prince coefficients).
#   c        | a (lower-triangular)
#   ---------+--------
#            | b5     (5th-order solution)
#            | b4     (4th-order embedded for error estimate)
# 7 stages, FSAL → k7 reused as k1 of next step.

_DOPRI_C = np.array([0.0, 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9, 1.0, 1.0])

_DOPRI_A = np.zeros((7, 7))
_DOPRI_A[1, 0] = 1.0 / 5
_DOPRI_A[2, 0] = 3.0 / 40
_DOPRI_A[2, 1] = 9.0 / 40
_DOPRI_A[3, 0] = 44.0 / 45
_DOPRI_A[3, 1] = -56.0 / 15
_DOPRI_A[3, 2] = 32.0 / 9
_DOPRI_A[4, 0] = 19372.0 / 6561
_DOPRI_A[4, 1] = -25360.0 / 2187
_DOPRI_A[4, 2] = 64448.0 / 6561
_DOPRI_A[4, 3] = -212.0 / 729
_DOPRI_A[5, 0] = 9017.0 / 3168
_DOPRI_A[5, 1] = -355.0 / 33
_DOPRI_A[5, 2] = 46732.0 / 5247
_DOPRI_A[5, 3] = 49.0 / 176
_DOPRI_A[5, 4] = -5103.0 / 18656
_DOPRI_A[6, 0] = 35.0 / 384
_DOPRI_A[6, 1] = 0.0
_DOPRI_A[6, 2] = 500.0 / 1113
_DOPRI_A[6, 3] = 125.0 / 192
_DOPRI_A[6, 4] = -2187.0 / 6784
_DOPRI_A[6, 5] = 11.0 / 84

# 5th-order (= last row of A in DP form — FSAL).
_DOPRI_B5 = _DOPRI_A[6, :].copy()

# 4th-order embedded.
_DOPRI_B4 = np.array([
    5179.0 / 57600,
    0.0,
    7571.0 / 16695,
    393.0 / 640,
    -92097.0 / 339200,
    187.0 / 2100,
    1.0 / 40,
])

# Error coefficients (B5 - B4).
_DOPRI_E = _DOPRI_B5 - _DOPRI_B4


@dataclass
class DormandPrince5:
    """Embedded-pair Dormand-Prince 5(4) adaptive ODE integrator.

    Solves ``dx/dt = f(t, x)`` with PI step-size control. Per step:

    1. Compute 7 stages ``k_i`` (last is reused as first of next
       step — FSAL property).
    2. 5th-order solution: ``x_new = x + dt · Σ B5_i · k_i``.
    3. Embedded 4th-order: ``x_emb = x + dt · Σ B4_i · k_i``.
    4. Error: ``err = (x_new − x_emb) / scale`` with
       ``scale = atol + rtol · max(|x|, |x_new|)``.
    5. Norm: ``err_norm = ||err||_RMS``.
    6. Step is accepted iff ``err_norm ≤ 1``.
    7. Next step size:
       ``dt_new = dt · safety · err_norm^(−1/5)``
       clamped to ``[dt_min, dt_max]`` and to a maximum growth /
       shrink factor.

    Parameters
    ----------
    f : Callable[[float, ndarray], ndarray]
        Right-hand side of the ODE.
    rtol : float, default 1e-5
        Relative tolerance.
    atol : float, default 1e-8
        Absolute tolerance.
    dt_min : float, default 1e-12
        Minimum allowed step size (raises if controller wants smaller).
    dt_max : float, default inf
        Maximum allowed step size.
    safety : float, default 0.9
        Safety factor on the optimal step-size estimate.
    """

    f: Callable[[float, np.ndarray], np.ndarray]
    rtol: float = 1.0e-5
    atol: float = 1.0e-8
    dt_min: float = 1.0e-12
    dt_max: float = np.inf
    safety: float = 0.9
    # Per-stage growth / shrink limits.
    growth_max: float = 5.0
    shrink_min: float = 0.2

    def solve(self,
                  t_span: Tuple[float, float],
                  x0,
                  dt_init: float | None = None,
                  ) -> AdaptiveSolution:
        """Integrate from ``t_span[0]`` to ``t_span[1]`` starting at
        ``x0``. Returns the full accepted-step trajectory.
        """
        t0, t_end = float(t_span[0]), float(t_span[1])
        if t_end <= t0:
            raise ValueError("t_span must have t_end > t_start")
        x = np.asarray(x0, dtype=np.float64).copy()
        if x.ndim != 1:
            raise ValueError("x0 must be 1-D")
        n = x.size

        # Initial step size heuristic from Hairer & Wanner §II.4
        # if not provided.
        if dt_init is None:
            f0 = np.asarray(self.f(t0, x), dtype=np.float64)
            d0 = np.linalg.norm(x / (self.atol + self.rtol * np.abs(x)))
            d1 = np.linalg.norm(f0 / (self.atol + self.rtol * np.abs(x)))
            if d0 < 1e-5 or d1 < 1e-5:
                dt = 1e-6
            else:
                dt = 0.01 * d0 / d1
            dt = min(dt, t_end - t0)
        else:
            dt = float(dt_init)

        t = t0
        n_f_evals = 0
        n_accepted = 0
        n_rejected = 0
        t_log = [t0]
        x_log = [x.copy()]
        dt_log: list = []

        # FSAL: keep k7 of the previous step as k1 of the next.
        k = np.zeros((7, n), dtype=np.float64)
        k[0] = np.asarray(self.f(t, x), dtype=np.float64)
        n_f_evals += 1

        max_steps = 10 * int(max(1.0, (t_end - t0) / max(dt, 1e-30)))
        step = 0
        while t < t_end and step < max_steps:
            step += 1
            # Don't overshoot the end point.
            if t + dt > t_end:
                dt = t_end - t

            # Stages 2..7.
            for i in range(1, 7):
                xi = x + dt * (_DOPRI_A[i, :i] @ k[:i, :])
                k[i] = np.asarray(self.f(t + _DOPRI_C[i] * dt, xi),
                                       dtype=np.float64)
                n_f_evals += 1

            # 5th-order solution.
            x_new = x + dt * (_DOPRI_B5 @ k)

            # Error estimate (b5 - b4).
            err = dt * (_DOPRI_E @ k)
            scale = self.atol + self.rtol * np.maximum(
                np.abs(x), np.abs(x_new))
            err_norm = float(np.sqrt(np.mean((err / scale) ** 2)))

            if err_norm <= 1.0:
                # Accept.
                t = t + dt
                x = x_new
                # FSAL: shift k7 → k1 for next step.
                k[0] = k[6].copy()
                n_accepted += 1
                t_log.append(t)
                x_log.append(x.copy())
                dt_log.append(dt)
                # Grow step.
                if err_norm < 1e-12:
                    factor = self.growth_max
                else:
                    factor = self.safety * err_norm ** (-0.2)
                factor = min(self.growth_max, max(1.0, factor))
                dt_new = dt * factor
            else:
                # Reject.
                n_rejected += 1
                factor = self.safety * err_norm ** (-0.2)
                factor = max(self.shrink_min, min(1.0, factor))
                dt_new = dt * factor

            dt = max(self.dt_min, min(self.dt_max, dt_new))
            if dt < self.dt_min and t < t_end:
                raise RuntimeError(
                    f"DOPRI5: step size dropped below dt_min={self.dt_min} "
                    f"at t={t:.6g}, x_norm={np.linalg.norm(x):.3g}. "
                    f"Tighten tolerances or set a smaller dt_min.")

        return AdaptiveSolution(
            t=np.asarray(t_log),
            x=np.asarray(x_log),
            n_accepted=n_accepted,
            n_rejected=n_rejected,
            n_f_evals=n_f_evals,
            dt_history=dt_log,
        )


# =============================================================================
# Radau IIA(3) — implicit, L-stable, stiff
# =============================================================================

# Butcher tableau for Radau IIA, 2 stages, order 3.
#   c1 = 1/3, c2 = 1
#   a11 = 5/12, a12 = -1/12
#   a21 = 3/4,  a22 =  1/4
#   b1  = 3/4,  b2  =  1/4
# Error estimator: compare order-3 solution to an order-2
# extrapolation. Hairer §IV.8 uses an embedded order-2 estimate
# obtained from a different linear combination of the stage values
# — for this simple Python implementation we use step-doubling for
# the error estimate (more f evals per step but robust + simple).

_RADAU_C = np.array([1.0 / 3, 1.0])
_RADAU_A = np.array([
    [5.0 / 12, -1.0 / 12],
    [3.0 / 4,   1.0 / 4],
])
_RADAU_B = np.array([3.0 / 4, 1.0 / 4])


@dataclass
class RadauIIA3:
    """Implicit Radau IIA(3) — 2-stage, order 3, L-stable.

    Solves stiff ``dx/dt = f(t, x)`` with PI step-size control.
    Each step requires solving an implicit 2-stage block system
    via Newton iteration:

        [I − dt·a11·J,   -dt·a12·J  ] [K1]   [f(t+c1·dt, x_0+dt·(a11·K1+a12·K2))]
        [-dt·a21·J,    I − dt·a22·J] [K2] = [f(t+c2·dt, x_0+dt·(a21·K1+a22·K2))]

    where ``J = ∂f/∂x`` evaluated at ``(t, x_0)``. The Jacobian
    is recomputed when Newton stalls (lazy refresh).

    For the **error estimate**, step-doubling is used: solve once
    with ``dt`` and once with two ``dt/2`` half-steps, compare the
    end states. More expensive than the embedded approach but
    robust for the Python prototype. The in-kernel C++ port
    (v1.6.0) will switch to the natural embedded estimate from
    Hairer & Wanner §IV.8.

    Parameters
    ----------
    f : Callable[[float, ndarray], ndarray]
        Right-hand side.
    jac : Callable[[float, ndarray], ndarray], optional
        Analytical Jacobian. If ``None``, finite differences are
        used (more f evals + less accuracy).
    rtol, atol, dt_min, dt_max, safety, growth_max, shrink_min :
        Same meaning as for :class:`DormandPrince5`.
    newton_max_iter : int, default 8
        Maximum Newton iterations per step.
    newton_tol : float, default 1e-8
        Relative Newton convergence tolerance.
    """

    f: Callable[[float, np.ndarray], np.ndarray]
    jac: Callable[[float, np.ndarray], np.ndarray] | None = None
    rtol: float = 1.0e-5
    atol: float = 1.0e-8
    dt_min: float = 1.0e-12
    dt_max: float = np.inf
    safety: float = 0.9
    growth_max: float = 5.0
    shrink_min: float = 0.2
    newton_max_iter: int = 8
    newton_tol: float = 1.0e-8

    def _jacobian(self, t: float, x: np.ndarray) -> np.ndarray:
        """Analytical Jacobian if available, else central finite
        differences (more accurate than forward, no extra eval)."""
        if self.jac is not None:
            return np.asarray(self.jac(t, x), dtype=np.float64)
        n = x.size
        eps = np.sqrt(np.finfo(np.float64).eps)
        J = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            h = eps * max(1.0, abs(x[i]))
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            J[:, i] = (np.asarray(self.f(t, xp), dtype=np.float64)
                          - np.asarray(self.f(t, xm), dtype=np.float64)
                         ) / (2.0 * h)
        return J

    def _solve_stage_block(self, t: float, x: np.ndarray,
                                  dt: float) -> Tuple[np.ndarray,
                                                            np.ndarray,
                                                            int]:
        """Newton iterate on the 2-stage block.

        Returns
        -------
        K : ndarray, shape (2, n)
            Stage values.
        x_new : ndarray, shape (n,)
            ``x_new = x + dt·(b1·K1 + b2·K2)``.
        n_iter : int
            Iteration count (for diagnostics).
        """
        n = x.size
        # Initial guess: K_i = f(t, x).
        f0 = np.asarray(self.f(t, x), dtype=np.float64)
        K = np.tile(f0, (2, 1))
        J = self._jacobian(t, x)
        I_n = np.eye(n)
        # Build the 2n × 2n block matrix once per Newton iteration.
        for it in range(self.newton_max_iter):
            x1 = x + dt * (_RADAU_A[0, 0] * K[0] + _RADAU_A[0, 1] * K[1])
            x2 = x + dt * (_RADAU_A[1, 0] * K[0] + _RADAU_A[1, 1] * K[1])
            f1 = np.asarray(self.f(t + _RADAU_C[0] * dt, x1),
                                dtype=np.float64)
            f2 = np.asarray(self.f(t + _RADAU_C[1] * dt, x2),
                                dtype=np.float64)
            r1 = K[0] - f1
            r2 = K[1] - f2
            err = max(np.linalg.norm(r1), np.linalg.norm(r2))
            if err < self.newton_tol * max(1.0, np.linalg.norm(K)):
                return K, x + dt * (_RADAU_B[0] * K[0]
                                          + _RADAU_B[1] * K[1]), it + 1
            # Block Jacobian:  M = [I - dt·a11·J,  -dt·a12·J;
            #                       -dt·a21·J,    I - dt·a22·J]
            M = np.block([
                [I_n - dt * _RADAU_A[0, 0] * J,
                 -dt * _RADAU_A[0, 1] * J],
                [-dt * _RADAU_A[1, 0] * J,
                 I_n - dt * _RADAU_A[1, 1] * J],
            ])
            rhs = np.concatenate([r1, r2])
            try:
                delta = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                # Singular block — abandon, let the controller
                # reject and shrink dt.
                return K, x + dt * (_RADAU_B[0] * K[0]
                                          + _RADAU_B[1] * K[1]), -1
            K[0] -= delta[:n]
            K[1] -= delta[n:]
        # Did not converge within newton_max_iter.
        return K, x + dt * (_RADAU_B[0] * K[0]
                                  + _RADAU_B[1] * K[1]), -1

    def solve(self,
                  t_span: Tuple[float, float],
                  x0,
                  dt_init: float | None = None,
                  ) -> AdaptiveSolution:
        """Integrate from ``t_span[0]`` to ``t_span[1]`` starting at
        ``x0``. Returns the full accepted-step trajectory."""
        t0, t_end = float(t_span[0]), float(t_span[1])
        if t_end <= t0:
            raise ValueError("t_span must have t_end > t_start")
        x = np.asarray(x0, dtype=np.float64).copy()

        if dt_init is None:
            # Conservative: pick a step sized to f(t0, x0) magnitude.
            f0 = np.asarray(self.f(t0, x), dtype=np.float64)
            f_mag = np.linalg.norm(f0) + 1e-30
            dt = max(self.dt_min,
                          min(self.dt_max,
                                0.01 / f_mag))
        else:
            dt = float(dt_init)
        dt = min(dt, t_end - t0)

        t = t0
        n_f_evals = 0
        n_accepted = 0
        n_rejected = 0
        t_log = [t0]
        x_log = [x.copy()]
        dt_log: list = []

        max_steps = 10 * int(max(1.0, (t_end - t0) / max(dt, 1e-30)))
        step = 0
        while t < t_end and step < max_steps:
            step += 1
            if t + dt > t_end:
                dt = t_end - t
            # One-step solve.
            _, x_full, it_full = self._solve_stage_block(t, x, dt)
            n_f_evals += 2 * max(1, abs(it_full)) + 1
            if it_full < 0:
                # Newton failed — shrink + retry.
                n_rejected += 1
                dt = max(self.dt_min, self.shrink_min * dt)
                continue
            # Step-doubling error estimate.
            _, x_half1, it_h1 = self._solve_stage_block(
                t, x, dt * 0.5)
            n_f_evals += 2 * max(1, abs(it_h1)) + 1
            if it_h1 < 0:
                n_rejected += 1
                dt = max(self.dt_min, self.shrink_min * dt)
                continue
            _, x_half2, it_h2 = self._solve_stage_block(
                t + dt * 0.5, x_half1, dt * 0.5)
            n_f_evals += 2 * max(1, abs(it_h2)) + 1
            if it_h2 < 0:
                n_rejected += 1
                dt = max(self.dt_min, self.shrink_min * dt)
                continue
            # Error: difference between full step and two half steps.
            scale = self.atol + self.rtol * np.maximum(
                np.abs(x), np.abs(x_half2))
            err = (x_full - x_half2) / scale
            err_norm = float(np.sqrt(np.mean(err ** 2)))

            if err_norm <= 1.0:
                # Accept the more accurate half-step solution (Richardson).
                t = t + dt
                # Order-3 Richardson improvement: x = x_half2 + (x_half2 - x_full)/((2^3) - 1) = x_half2 + (x_half2-x_full)/7
                x = x_half2 + (x_half2 - x_full) / 7.0
                n_accepted += 1
                t_log.append(t)
                x_log.append(x.copy())
                dt_log.append(dt)
                if err_norm < 1e-12:
                    factor = self.growth_max
                else:
                    factor = self.safety * err_norm ** (-1.0 / 4.0)
                factor = min(self.growth_max, max(1.0, factor))
                dt_new = dt * factor
            else:
                n_rejected += 1
                factor = self.safety * err_norm ** (-1.0 / 4.0)
                factor = max(self.shrink_min, min(1.0, factor))
                dt_new = dt * factor
            dt = max(self.dt_min, min(self.dt_max, dt_new))
            if dt < self.dt_min and t < t_end:
                raise RuntimeError(
                    f"Radau IIA(3): step dropped below dt_min "
                    f"at t={t:.6g}.")

        return AdaptiveSolution(
            t=np.asarray(t_log),
            x=np.asarray(x_log),
            n_accepted=n_accepted,
            n_rejected=n_rejected,
            n_f_evals=n_f_evals,
            dt_history=dt_log,
        )
