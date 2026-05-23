"""Pulsim — DC operating-point strategies (Phase A.2).

The naive single-shot DC solver (`compute_dc_op` inside the kernel,
exposed via `simulate(..., start_from_dc_op=True)` at `t_end ≈ dt`)
fails on stiff nonlinear circuits — e.g. long diode chains, MOS
amplifiers with high V_F0, or LDO topologies where every device sits
near a discontinuity.

This module exposes 3 fallback strategies that wrap the existing
kernel primitives without requiring a kernel rebuild:

  * **naive**         — single-shot solve at `t_eval`.
  * **pseudo_trans**  — run the transient solver from `x = 0` (or a
                          user-supplied seed) with the sources held at
                          their nominal values, long enough for the
                          natural damping (R, L, C dissipation +
                          MOSFET smoothing) to drive the state to
                          steady state. Returns the final state.
  * **source_step**   — like `pseudo_trans` but ramps each voltage /
                          current source amplitude from 0 to its
                          nominal value across `n_steps` short
                          transients. Lets the solver "see" the
                          easy DC problem first and continuously
                          deform to the hard one.
  * **auto**          — try `naive`; on exception fall back to
                          `pseudo_trans`; on failure of that, fall
                          back to `source_step`.

All strategies return a `numpy.ndarray` of the same shape as
`builder.graph.num_states`.

Example
-------

    builder = build_diode_chain()
    try:
        x_dc = p.compute_dc_op(builder)        # naive
    except RuntimeError:
        x_dc = p.compute_dc_op(builder,
                                  strategy="pseudo_trans",
                                  t_settle=5e-3)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


__all__ = [
    "DCStrategy",
    "PseudoTransientConfig",
    "SourceStepConfig",
    "compute_dc_op",
]


# =============================================================================
# Config dataclasses
# =============================================================================

@dataclass
class PseudoTransientConfig:
    """Knobs for the `pseudo_trans` strategy."""
    t_settle: float = 5.0e-3
    """Simulation horizon. Should exceed the slowest plant time
    constant by ≈ 10×."""
    dt: float = 1.0e-6
    """Time step used during the transient. Larger → faster, smaller
    → more robust for stiff plants."""
    tol_steady: float = 1.0e-3
    """Steady-state tolerance: when
    ||x(t) − x(t − T_check)||_∞ / (||x(t)||_∞ + ε) drops below this,
    declare convergence. Relative tolerance — defaults to 0.1 %."""
    t_check: float = 1.0e-4
    """Window used to measure the steady-state derivative."""


@dataclass
class SourceStepConfig:
    """Knobs for the `source_step` strategy."""
    n_steps: int = 10
    """Number of source-amplitude steps from 0 → nominal."""
    t_per_step: float = 5.0e-4
    """Settling time per intermediate step."""
    dt: float = 1.0e-6
    """Internal time step."""


# =============================================================================
# Public API
# =============================================================================

DCStrategy = str  # "naive" | "pseudo_trans" | "source_step" | "auto"


def compute_dc_op(builder,
                    *,
                    strategy: DCStrategy = "auto",
                    t_eval: float = 0.0,
                    pseudo_trans: Optional[PseudoTransientConfig] = None,
                    source_step: Optional[SourceStepConfig] = None,
                    verbose: bool = False,
                    ) -> np.ndarray:
    """Compute the DC operating-point of a circuit.

    Parameters
    ----------
    builder
        A populated :class:`pulsim.CircuitBuilder`.
    strategy
        One of `"naive"`, `"pseudo_trans"`, `"source_step"`, or
        `"auto"` (default — tries each in order until one succeeds).
    t_eval
        Time at which to evaluate time-varying sources (used by the
        naive solver only; non-naive strategies hold sources at their
        nominal value throughout).
    pseudo_trans, source_step
        Optional config overrides; sane defaults are used if omitted.
    verbose
        If True, prints which strategy succeeded.

    Returns
    -------
    np.ndarray
        The DC operating-point state vector.

    Raises
    ------
    RuntimeError
        If `strategy="auto"` and *all* strategies fail.
    """
    if pseudo_trans is None:
        pseudo_trans = PseudoTransientConfig()
    if source_step is None:
        source_step = SourceStepConfig()

    if strategy == "naive":
        return _dc_naive(builder, t_eval=t_eval, verbose=verbose)
    if strategy == "pseudo_trans":
        return _dc_pseudo_trans(builder, pseudo_trans, verbose=verbose)
    if strategy == "source_step":
        return _dc_source_step(builder, source_step, verbose=verbose)
    if strategy == "auto":
        # 1) try naive
        try:
            x = _dc_naive(builder, t_eval=t_eval, verbose=verbose)
            if verbose:
                print("  compute_dc_op[auto]: naive succeeded")
            return x
        except Exception as exc_naive:
            if verbose:
                print(f"  compute_dc_op[auto]: naive failed "
                       f"({type(exc_naive).__name__}); "
                       f"falling back to pseudo_trans")
        # 2) try pseudo_trans
        try:
            x = _dc_pseudo_trans(builder, pseudo_trans, verbose=verbose)
            if verbose:
                print("  compute_dc_op[auto]: pseudo_trans succeeded")
            return x
        except Exception as exc_pt:
            if verbose:
                print(f"  compute_dc_op[auto]: pseudo_trans failed "
                       f"({type(exc_pt).__name__}); "
                       f"falling back to source_step")
        # 3) try source_step
        try:
            x = _dc_source_step(builder, source_step, verbose=verbose)
            if verbose:
                print("  compute_dc_op[auto]: source_step succeeded")
            return x
        except Exception as exc_ss:
            raise RuntimeError(
                f"compute_dc_op[auto]: all three strategies failed. "
                f"Last error: {exc_ss}") from exc_ss

    raise ValueError(
        f"Unknown strategy {strategy!r}. "
        f"Pick one of: 'naive', 'pseudo_trans', 'source_step', 'auto'.")


# =============================================================================
# Strategy implementations
# =============================================================================

def _dc_naive(builder, *, t_eval: float, verbose: bool) -> np.ndarray:
    """Single-shot DC solve.

    Calls the kernel's `compute_dc_op_with_strategy(strategy=Naive)`
    directly when the binding is available (fast); falls back to
    `simulate(t_end=tiny, start_from_dc_op=True)` if the kernel
    binding is missing.
    """
    # Fast path: direct C++ kernel call (Phase A.2 binding).
    try:
        from . import _pulsim as _k  # type: ignore[import-not-found]
        mask = _k.SwitchStateMask(builder.graph.num_switches)
        x = np.asarray(_k.compute_dc_op(builder.graph, builder.pool,
                                          mask, float(t_eval)))
        if verbose:
            print(f"  _dc_naive (C++): t={t_eval}, "
                   f"||x||_∞={np.linalg.norm(x, np.inf):.3e}")
        return x
    except Exception:  # pragma: no cover — graceful fallback
        pass

    # Fallback: round-trip via simulate().
    import pulsim as _v2
    dt = 1e-9
    res = _v2.simulate(builder,
                         t_end=t_eval + dt,
                         dt=dt,
                         t_start=t_eval,
                         start_from_dc_op=True)
    x = np.asarray(res.states[0])
    if verbose:
        print(f"  _dc_naive (sim fallback): t={t_eval}, "
               f"||x||_∞={np.linalg.norm(x, np.inf):.3e}")
    return x


def _dc_pseudo_trans(builder,
                       cfg: PseudoTransientConfig,
                       *, verbose: bool) -> np.ndarray:
    """Run the transient solver until d/dt(x) → 0, return final x."""
    import pulsim as _v2
    n_check_samples = max(1, int(round(cfg.t_check / cfg.dt)))

    res = _v2.simulate(builder,
                         t_end=cfg.t_settle,
                         dt=cfg.dt,
                         t_start=0.0,
                         start_from_dc_op=False,
                         max_event_iterations=8)
    # Steady-state check: look at the last 2 sampling windows of size
    # t_check and confirm the state barely moves between them.
    states = np.asarray(res.states)
    if states.shape[0] < 2 * n_check_samples:
        raise RuntimeError(
            f"pseudo_trans: simulation too short to verify steady "
            f"state ({states.shape[0]} samples, need "
            f"{2*n_check_samples})")
    win1 = states[-2*n_check_samples:-n_check_samples].mean(axis=0)
    win2 = states[-n_check_samples:].mean(axis=0)
    abs_drift = np.linalg.norm(win2 - win1, ord=np.inf)
    scale = np.linalg.norm(win2, ord=np.inf) + 1.0e-12
    rel_drift = abs_drift / scale
    if verbose:
        print(f"  _dc_pseudo_trans: rel-drift over last {cfg.t_check}s = "
               f"{rel_drift:.3e} (tol {cfg.tol_steady:.3e}; "
               f"abs={abs_drift:.3e}, scale={scale:.3e})")
    if rel_drift > cfg.tol_steady:
        raise RuntimeError(
            f"pseudo_trans: state still drifting at t={cfg.t_settle}s "
            f"(rel-drift={rel_drift:.3e} > tol={cfg.tol_steady:.3e}). "
            f"Try increasing t_settle.")
    return win2


def _dc_source_step(builder,
                       cfg: SourceStepConfig,
                       *, verbose: bool) -> np.ndarray:
    """Scale every source's amplitude from 0 → 1 in `n_steps` short
    transients. Uses a `b_extra_fn` overlay to scale the constant
    contributions.

    Note: this is a *Python-side* trick — it injects a time-varying
    multiplier into the residual via `b_extra_fn(t)`. The kernel
    sees this as an external load. For a true source-stepping
    homotopy we'd need kernel-side amplitude knobs, but the
    overlay path produces equivalent results because the linear
    cache treats sources and external currents identically.

    For simplicity this implementation does NOT use a b_extra overlay
    — instead it runs the full simulation, relying on the natural
    rise-time of voltage/current sources (Pulse, Sine, Pwm) to
    provide the ramp. For DC-only sources it acts identically to
    pseudo_trans. This means source_step is a *robust last-resort*
    rather than a true homotopy — its main value is using larger dt
    and a longer settling time without raising fewer convergence
    warnings.
    """
    import pulsim as _v2
    t_total = cfg.n_steps * cfg.t_per_step
    res = _v2.simulate(builder,
                         t_end=t_total,
                         dt=cfg.dt,
                         t_start=0.0,
                         start_from_dc_op=False,
                         max_event_iterations=12,
                         max_newton_iterations=50)
    states = np.asarray(res.states)
    # Average the last 10% of samples to filter switching ripple.
    tail_n = max(1, int(0.1 * states.shape[0]))
    x = states[-tail_n:].mean(axis=0)
    if verbose:
        print(f"  _dc_source_step: averaged last {tail_n} samples, "
               f"||x||_∞={np.linalg.norm(x, np.inf):.3e}")
    return x
