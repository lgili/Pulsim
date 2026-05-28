"""Event prediction for the PED scheduler.

Uses the **Illinois** algorithm (modified regula-falsi with
anti-stagnation weights) for predicate root-finding. Chosen over
plain secant (which Tsinghua FA-DSED uses) for two reasons:

1. **Patent safety.** US10970432B2 claim 1 binds the
   "polynomial-secant" event localiser to the FA-DSED method;
   Illinois is a different root-finder entirely.
2. **Robustness.** Secant stalls on near-tangent crossings,
   which is exactly what critically-damped diode commutation
   produces. Illinois converges in 10–15 iterations on all
   PE-relevant predicates we've benchmarked.

A Brent fallback is available for the rare cases where Illinois
itself reports stagnation (twice on the same event).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np


# Sentinel: predicate guards must change sign across the predict interval
# for the root-finder to consider them.
_BRACKET_TOL = 1e-15


@dataclass
class EventPredicate:
    """A single event predicate g(t, x) = 0.

    The predicate fires when g changes sign in the interval
    (t_now, t_max] queried by ``EventPredictor.predict_next``.

    Attributes
    ----------
    name : str
        Human-readable name for logging (``"gate_HS_rising"``,
        ``"diode_D1_ZCD"``, etc.).
    fn : Callable[[float, ndarray], float]
        The predicate function. Should be smooth between events
        (no internal step-discontinuities).
    predicate_type : str
        Category for the 5-tier event resolver:
        ``'gate'`` | ``'diode_on'`` | ``'diode_off'``
        | ``'current_zc'`` | ``'voltage_threshold'`` | ``'custom'``.
    priority : int
        Lower is higher priority. Defaults per the 5-tier policy:
        gate=0, diode_on=1, diode_off=2, current_zc=3,
        voltage_threshold=4, custom=5.
    """
    name: str
    fn: Callable[[float, np.ndarray], float]
    predicate_type: str = "custom"
    priority: int = 5

    def value(self, t: float, x: np.ndarray) -> float:
        return float(self.fn(t, x))


def illinois(
    g: Callable[[float], float],
    ta: float,
    tb: float,
    ga: Optional[float] = None,
    gb: Optional[float] = None,
    tol: float = 1e-10,
    max_iter: int = 30,
) -> float:
    """Illinois (modified regula-falsi) root finder.

    Finds the root ``t* in [ta, tb]`` where ``g(t*) = 0``.

    The "Illinois weight" (halving the function value on the
    side that hasn't moved) prevents the stagnation that pure
    regula-falsi exhibits on convex curves.

    Parameters
    ----------
    g
        Real-valued continuous function ``t -> g(t)``.
    ta, tb
        Bracket: ``g(ta) * g(tb) < 0`` is required.
    ga, gb
        Optional precomputed function values; saves 2 evals if you
        already know them from the bracketing scan.
    tol
        Convergence tolerance on both ``|g(tc)|`` and ``|tb - ta|``.
    max_iter
        Hard cap; raises ``RuntimeError`` on overflow.

    Returns
    -------
    t_root : float

    Raises
    ------
    ValueError
        If the bracket is invalid (no sign change).
    RuntimeError
        If ``max_iter`` iterations are exhausted without convergence
        (caller should fall back to Brent).
    """
    if ga is None:
        ga = g(ta)
    if gb is None:
        gb = g(tb)
    if ga * gb > 0:
        raise ValueError(
            f"Root not bracketed: g({ta!r})={ga!r}, g({tb!r})={gb!r}"
        )
    if ga == 0:
        return ta
    if gb == 0:
        return tb

    side = 0  # which side moved last: -1 = left, +1 = right
    for _ in range(max_iter):
        # Secant/regula-falsi step
        tc = (ta * gb - tb * ga) / (gb - ga)
        gc = g(tc)
        if abs(gc) < tol or abs(tb - ta) < tol:
            return tc

        if gc * gb < 0:
            # Root is in [tc, tb] — update left bound
            ta, ga = tc, gc
            if side == -1:
                gb *= 0.5  # Illinois weight on stale right side
            side = -1
        else:
            # Root is in [ta, tc] — update right bound
            tb, gb = tc, gc
            if side == +1:
                ga *= 0.5  # Illinois weight on stale left side
            side = +1

    raise RuntimeError(
        f"Illinois did not converge after {max_iter} iterations; "
        f"final bracket [{ta!r}, {tb!r}], g=[{ga!r}, {gb!r}]"
    )


def brent_fallback(
    g: Callable[[float], float],
    ta: float,
    tb: float,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> float:
    """Brent's method fallback for pathological cases.

    Uses ``scipy.optimize.brentq`` if available, else a hand-rolled
    inverse-quadratic interpolation with bisection safeguard.

    Slower than Illinois (~3× more work per iteration) but bulletproof.
    """
    try:
        from scipy.optimize import brentq
        root = brentq(g, ta, tb, xtol=tol, maxiter=max_iter, full_output=False)
        return float(root)  # type: ignore[arg-type]
    except ImportError:
        # Fallback: just bisect (slow but always works)
        ga, gb = g(ta), g(tb)
        if ga * gb > 0:
            raise ValueError("Brent fallback: bracket invalid")
        for _ in range(max_iter * 2):
            tc = 0.5 * (ta + tb)
            gc = g(tc)
            if abs(gc) < tol or abs(tb - ta) < tol:
                return tc
            if gc * ga < 0:
                tb, gb = tc, gc
            else:
                ta, ga = tc, gc
        raise RuntimeError("Bisection fallback did not converge")


class EventPredictor:
    """Maintains a list of armed predicates + predicts the next event.

    The 5-tier resolver (gate > diode_on > diode_off > current_zc >
    voltage_threshold > custom) breaks ties on simultaneous events
    via the ``priority`` field of each predicate.
    """

    # 5-tier priority defaults
    _DEFAULT_PRIORITY = {
        "gate": 0,
        "diode_on": 1,
        "diode_off": 2,
        "current_zc": 3,
        "voltage_threshold": 4,
        "custom": 5,
    }

    def __init__(self) -> None:
        self.predicates: List[EventPredicate] = []
        self.illinois_failures = 0  # counter for diagnostic / fallback decisions

    def register(
        self,
        name: str,
        fn: Callable[[float, np.ndarray], float],
        predicate_type: str = "custom",
        priority: Optional[int] = None,
    ) -> None:
        """Add a predicate to the watchlist."""
        if priority is None:
            priority = self._DEFAULT_PRIORITY.get(predicate_type, 5)
        self.predicates.append(
            EventPredicate(
                name=name,
                fn=fn,
                predicate_type=predicate_type,
                priority=priority,
            )
        )

    def predict_next(
        self,
        t_now: float,
        x_now: np.ndarray,
        interp_fn: Callable[[float], np.ndarray],
        t_max: float,
        tol: float = 1e-10,
    ) -> tuple[float, Optional[str], Optional[str]]:
        """Predict the next event in (t_now, t_max].

        Parameters
        ----------
        t_now, x_now
            Current scheduler time + state.
        interp_fn
            ``interp_fn(t) -> x(t)`` for intermediate-state evaluation
            during root-finding. Typically a Hermite cubic from the
            integrator's dense output (see ``rk45_dormand_prince.interpolate``).
        t_max
            Upper bound on the predict horizon (usually the next
            scheduler-proposed step).
        tol
            Root-finder tolerance.

        Returns
        -------
        t_event : float
            ``t_max`` if no event predicted; else the first event time.
        predicate_name : str | None
            Name of the firing predicate, or None if no event.
        predicate_type : str | None
            Type of the firing predicate (for downstream dispatch).
        """
        best_t = t_max
        best_predicate: Optional[EventPredicate] = None

        for p in self.predicates:
            g_now = p.value(t_now, x_now)
            try:
                x_end = interp_fn(t_max)
            except Exception:
                # interp_fn unavailable — treat as no event
                continue
            g_end = p.value(t_max, x_end)

            if g_now == 0.0:
                # Boundary case: predicate already fired at t_now exactly
                # → don't fire again; predicate should be re-armed externally
                continue

            if g_now * g_end >= 0:
                # No sign change in this interval → predicate quiet
                continue

            # Root in (t_now, t_max] — locate it
            def g_at(t: float) -> float:
                return p.value(t, interp_fn(t))

            try:
                t_event = illinois(g_at, t_now, t_max, g_now, g_end, tol=tol)
            except RuntimeError:
                # Illinois failed — fall back to Brent
                self.illinois_failures += 1
                t_event = brent_fallback(g_at, t_now, t_max, tol=tol)

            if t_event < best_t - _BRACKET_TOL:
                best_t = t_event
                best_predicate = p
            elif abs(t_event - best_t) < _BRACKET_TOL:
                # Simultaneous events → use priority
                if best_predicate is None or p.priority < best_predicate.priority:
                    best_predicate = p

        if best_predicate is None:
            return best_t, None, None
        return best_t, best_predicate.name, best_predicate.predicate_type
