"""Pulsim — the DC operating point.

`compute_dc_op(builder)` answers one question: with the sources held
at their values at `t_eval`, where does this circuit sit?

Getting that right is three things, not one, and v2.0 Phase 2 made
this module do all three (before it, none of them):

  1. **Nonlinear devices are stamped.** A smooth diode / MOSFET /
     IGBT is a real device at DC. Solving with them open is not an
     approximation, it is a different circuit — 5 V through 1 kΩ into
     a diode answers 5.000 V instead of 0.700 V.
  2. **PWL diode states are resolved.** A diode's on/off bit is both
     an input to the matrix and an output of the solve, so the pair
     is iterated to consistency.
  3. **A failed solve is recovered, not reported.** Stiff operating
     points are normal. The cascade below runs until one rung
     answers.

Strategies
----------
  * ``"naive"``        — one solve, no fallback. Fastest, and what
                          you want when you would rather see the
                          diagnostic than an answer.
  * ``"gmin_step"``    — clamp every node to ground through a large
                          conductance, then relax it by decades,
                          warm-starting each solve. Fixes a
                          badly-pivoted matrix and a Newton that
                          cannot find the basin from x = 0.
  * ``"source_step"``  — ramp every independent source amplitude
                          from 0 to nominal, re-solving at each step.
  * ``"pseudo_trans"`` — integrate dx/dt = -F to equilibrium.
  * ``"settle"``       — run an actual transient until the state
                          stops moving. The only strategy that can
                          find a *switching* steady state, which no
                          DC solve can; also the slowest by far.
  * ``"auto"`` (default) — the kernel cascade (naive → gmin → source
                          → pseudo-transient), then ``"settle"`` as a
                          last resort.

Every strategy returns a ``numpy.ndarray`` of length
``pool.state_size(graph)``.

Example
-------

    x_dc = p.compute_dc_op(builder)                    # auto
    x_dc = p.compute_dc_op(builder, strategy="naive")  # or fail loudly
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np


__all__ = [
    "DCStrategy",
    "PseudoTransientConfig",
    "SettleConfig",
    "SourceStepConfig",
    "compute_dc_op",
]


# =============================================================================
# Config dataclasses
# =============================================================================

@dataclass
class PseudoTransientConfig:
    """Knobs for the ``"settle"`` strategy — running a transient until
    the state stops moving.

    The name is historical: before v2.0, ``strategy="pseudo_trans"``
    meant this settling transient. It now means the kernel's
    pseudo-transient continuation rung, which is a different algorithm
    with different knobs, so this config no longer belongs to it.
    :data:`SettleConfig` is the alias to prefer.
    """
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

DCStrategy = str
"""One of ``"naive" | "gmin_step" | "source_step" | "pseudo_trans" |
``"settle" | "auto"``."""

# `PseudoTransientConfig` configures the `"settle"` strategy — the
# transient-to-steady-state one. The kernel's own pseudo-transient
# rung is a different algorithm with its own knobs; this alias makes
# the Python-side meaning unambiguous without breaking imports.
SettleConfig = PseudoTransientConfig

class _SettleReport:
    """What `"settle"` can honestly say about itself: it ran a
    transient, so there is no rung, no conductance floor and no
    residual against a DC system it never assembled."""

    strategy = "settle"
    rungs_attempted = 0
    final_gmin = 0.0
    residual = float("nan")

    def summary(self) -> str:
        return ("DC operating point taken from a settled transient "
                 "(no DC system was solved)")

    def __repr__(self) -> str:
        return f"<DCSolveReport: {self.summary()}>"


_KERNEL_STRATEGIES = {
    "naive": "Naive",
    "gmin_step": "GminStepping",
    "source_step": "SourceStepping",
    "pseudo_trans": "PseudoTransient",
    "auto": "Auto",
}

_VALID = tuple(_KERNEL_STRATEGIES) + ("settle",)


def _reraise_as_cancelled(exc, where="compute_dc_op"):
    """Translate the kernel's `_CxxCancelled` into `pulsim.Cancelled`.

    `py::register_exception` can only carry a message, and the
    user-facing type is defined in a Python module that imports the
    extension — so the two cannot be related by inheritance and the
    translation has to happen at the boundary. Without it a Cancel
    arrives as `pulsim._pulsim._CxxCancelled`, which
    `except pulsim.Cancelled` does not catch, breaking the contract
    `docs/helpers.md` states.
    """
    from . import Cancelled as _Cancelled
    m = re.search(r"progress index (-?\d+)", str(exc))
    idx = int(m.group(1)) if m else None
    if idx is not None and idx < 0:
        idx = None
    raise _Cancelled(where, point_index=idx) from exc


def _is_kernel_cancellation(exc) -> bool:
    from . import _pulsim as _k  # type: ignore[import-not-found]
    cxx = getattr(_k, "_CxxCancelled", None)
    return cxx is not None and isinstance(exc, cxx)


def compute_dc_op(builder,
                    *,
                    strategy: DCStrategy = "auto",
                    t_eval: float = 0.0,
                    settle: Optional["SettleConfig"] = None,
                    pseudo_trans: Optional[PseudoTransientConfig] = None,
                    source_step: Optional[SourceStepConfig] = None,
                    gmin: Optional[float] = None,
                    enable_nonlinear_refresh: Optional[bool] = None,
                    max_event_iterations: Optional[int] = None,
                    auto_regularize: Optional[bool] = None,
                    mask=None,
                    verbose: bool = False,
                    report: Optional[list] = None,
                    should_continue=None,
                    ) -> np.ndarray:
    """Compute the DC operating point of a circuit.

    Parameters
    ----------
    builder
        A populated :class:`pulsim.CircuitBuilder`.
    strategy
        ``"auto"`` (default), ``"naive"``, ``"gmin_step"``,
        ``"source_step"``, ``"pseudo_trans"``, or ``"settle"``.
        See the module docstring.
    t_eval
        Time at which time-varying sources (PWM / sine / pulse) are
        sampled. The operating point is the DC solution with each
        source frozen at its value there.
    pseudo_trans
        Knobs for ``"settle"`` (horizon, dt, steady-state tolerance).
    source_step
        Knobs for ``"source_step"``; only ``n_steps`` is used by the
        kernel homotopy.
    gmin
        Conductance floor to ground, in siemens. ``None`` uses the
        kernel default (1e-12, SPICE's GMIN); ``0`` disables it.
    enable_nonlinear_refresh
        ``None`` (default) stamps the smooth diode / MOSFET / IGBT
        chain whenever the circuit contains one. ``False`` solves
        them as open circuits — a different circuit, so pass it only
        when that is what you mean.
    max_event_iterations
        Rounds of PWL-diode-state re-solving before giving up.
    auto_regularize
        Insert 1 GΩ reference ties for unreferenced subnets, exactly
        as :func:`pulsim.simulate` does. Defaults to on; this MUTATES
        the builder.
    mask
        A :class:`pulsim.SwitchStateMask` fixing the explicit
        switches. Defaults to all-open — a DC operating point has no
        switching schedule to read one from. PWL diode bits inside
        the mask are resolved by iteration regardless.
    report
        If a list is passed, the kernel's ``DCSolveReport`` is
        appended to it, so you can see which rung answered.

    Returns
    -------
    np.ndarray
        The DC operating-point state vector.

    Raises
    ------
    RuntimeError
        If the chosen strategy (or, for ``"auto"``, every strategy)
        fails.
    """
    # `pseudo_trans=` used to configure the settling transient. It
    # still does, under its new name — but passing it alongside
    # strategy="pseudo_trans" now means two different algorithms, so
    # say so instead of ignoring the argument.
    if settle is not None and pseudo_trans is not None:
        raise ValueError(
            "compute_dc_op(): pass either settle= or its old name "
            "pseudo_trans=, not both.")
    if settle is None:
        settle = pseudo_trans
    if settle is not None and strategy == "pseudo_trans":
        raise ValueError(
            "compute_dc_op(strategy='pseudo_trans'): since v2.0 this "
            "name means the KERNEL's pseudo-transient continuation "
            "rung, and SettleConfig/PseudoTransientConfig configures "
            "the transient-settling strategy instead. Use "
            "strategy='settle' with settle=..., or drop the config "
            "and let the kernel rung use its own defaults.")
    if settle is None:
        settle = PseudoTransientConfig()
    pseudo_trans = settle
    if source_step is None:
        source_step = SourceStepConfig()
    if strategy not in _VALID:
        raise ValueError(
            f"Unknown strategy {strategy!r}. Pick one of: "
            + ", ".join(repr(v) for v in _VALID))

    if should_continue is not None and not should_continue():
        from . import Cancelled as _Cancelled
        raise _Cancelled("compute_dc_op", point_index=0)

    # Same topology preflight `simulate()` runs, for the same reason
    # and with the same default: a node nobody referenced is a defect
    # the tool should repair and report, not one the user should have
    # to recognise from a singular-matrix message. Running it here
    # too keeps the two entry points from disagreeing about what
    # "floating" means.
    if auto_regularize is None:
        auto_regularize = True
    if auto_regularize:
        from . import PreflightOptions as _PfOpts
        pf = builder.run_preflight(_PfOpts(auto_regularize=True))
        if not pf.empty():
            import warnings
            warnings.warn(
                "compute_dc_op(): " + pf.summary() +
                "\n  Pass auto_regularize=False to get the original "
                "singular-matrix error instead. The ties persist on "
                "the builder.",
                stacklevel=2)

    if strategy == "settle":
        # "settle" is a transient, not a solve: it has no rung, no
        # gmin, no diode iteration to report on. Refuse the arguments
        # it cannot honour rather than dropping them silently — a
        # discarded `gmin=0` or `mask=` would change what the caller
        # believes was computed.
        unhonoured = [
            name for name, given in (
                ("gmin", gmin is not None),
                ("mask", mask is not None),
                ("t_eval", bool(t_eval)),
                ("max_event_iterations",
                 max_event_iterations is not None),
                ("enable_nonlinear_refresh",
                 enable_nonlinear_refresh is not None),
            ) if given
        ]
        if unhonoured:
            raise ValueError(
                "compute_dc_op(strategy='settle') runs a transient to "
                "steady state, so it cannot honour: "
                + ", ".join(unhonoured)
                + ". Drop them, or use a DC strategy "
                  "('auto', 'naive', 'gmin_step', 'source_step', "
                  "'pseudo_trans').")
        x = _dc_settle(builder, pseudo_trans, verbose=verbose,
                        should_continue=should_continue)
        if report is not None:
            report.append(_SettleReport())
        return x

    from . import _pulsim as _k  # type: ignore[import-not-found]

    if mask is None:
        mask = _k.SwitchStateMask(builder.graph.num_switches)

    # ONE dispatch for every kernel strategy. Routing named rungs
    # around `compute_dc_operating_point` would solve the circuit
    # with every PWL diode frozen in the all-open default mask —
    # reintroducing, one strategy name at a time, exactly the silent
    # wrong answer this module exists to close.
    try:
        op = _k.compute_dc_operating_point(
            builder.graph, builder.pool, mask,
            strategy=getattr(_k.DCStrategy,
                              _KERNEL_STRATEGIES[strategy]),
            enable_cascade=(strategy == "auto"),
            max_event_iterations=(16 if max_event_iterations is None
                                   else int(max_event_iterations)),
            ss_n_steps=int(source_step.n_steps),
            t_eval=float(t_eval),
            enable_nonlinear_refresh=enable_nonlinear_refresh,
            should_continue=should_continue,
            **({} if gmin is None else {"gmin": float(gmin)}))
    except Exception as exc:
        if _is_kernel_cancellation(exc):
            _reraise_as_cancelled(exc)
        raise

    if report is not None:
        report.append(op.report)
    if verbose:
        print(f"  compute_dc_op[{strategy}]: {op.report.summary()}")
    return np.asarray(op.x)


# =============================================================================
# Strategy implementations
# =============================================================================

def _dc_settle(builder,
                       cfg: PseudoTransientConfig,
                       *, verbose: bool,
                       should_continue=None) -> np.ndarray:
    """Run the transient solver until d/dt(x) stops moving.

    Not a DC solve at all — an actual simulation, sampled once it
    stops changing. That makes it the slowest strategy and the only
    one that can answer for a circuit whose steady state is a
    switching average rather than a fixed point.
    """
    import pulsim as _v2
    n_check_samples = max(1, int(round(cfg.t_check / cfg.dt)))

    res = _v2.simulate(builder,
                         t_end=cfg.t_settle,
                         dt=cfg.dt,
                         t_start=0.0,
                         start_from_dc_op=False,
                         max_event_iterations=8,
                         should_continue=should_continue)
    # Steady-state check: look at the last 2 sampling windows of size
    # t_check and confirm the state barely moves between them.
    states = np.asarray(res.states)
    if states.shape[0] < 2 * n_check_samples:
        raise RuntimeError(
            f"settle: simulation too short to verify steady "
            f"state ({states.shape[0]} samples, need "
            f"{2*n_check_samples})")
    win1 = states[-2*n_check_samples:-n_check_samples].mean(axis=0)
    win2 = states[-n_check_samples:].mean(axis=0)
    abs_drift = np.linalg.norm(win2 - win1, ord=np.inf)
    scale = np.linalg.norm(win2, ord=np.inf) + 1.0e-12
    rel_drift = abs_drift / scale
    if verbose:
        print(f"  _dc_settle: rel-drift over last {cfg.t_check}s = "
               f"{rel_drift:.3e} (tol {cfg.tol_steady:.3e}; "
               f"abs={abs_drift:.3e}, scale={scale:.3e})")
    if rel_drift > cfg.tol_steady:
        raise RuntimeError(
            f"settle: state still drifting at t={cfg.t_settle}s "
            f"(rel-drift={rel_drift:.3e} > tol={cfg.tol_steady:.3e}). "
            f"Try increasing t_settle.")
    return win2
