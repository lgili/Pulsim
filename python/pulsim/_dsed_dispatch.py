"""DSED engine dispatch — validation, option resolution, and bridge stub.

This module is internal (note the leading underscore). The PUBLIC
entry point is :func:`pulsim.simulate` with ``engine='dsed'``; the
machinery here is what runs when that flag is selected.

The dispatch has three responsibilities:

1. **Validation** (:func:`_validate_engine_kwargs`) — catch user
   mistakes BEFORE any heavy work happens. Each error message
   tells the user EXACTLY what's wrong and how to fix it.
2. **Option resolution** (:func:`_resolve_dsed_options`) — apply
   sensible defaults + user overrides, producing a fully-resolved
   :class:`DSEDOptions` struct that the scheduler can consume.
3. **Bridge stub** (:func:`run_dsed_from_builder`) — the entry
   point that would adapt a :class:`pulsim.CircuitBuilder` to the
   PED scheduler. Currently raises :class:`NotImplementedError` with
   an honest path-forward message; the PED kernel itself (Gates 1-5)
   is fully validated and accessible via :mod:`pulsim.dsed` directly.

User-facing knobs:

* ``rtol`` / ``atol`` — RK45 tolerances (default 1e-6 / 1e-9)
* ``dt`` — interpreted as ``dt_max`` (step cap) for the dsed engine
* ``integrator`` — ``'auto'`` (default), ``'rk45'``, or ``'bdf2'``
* ``stiffness_threshold`` — when ``|λ_max|·h > threshold``, dispatcher
  picks BDF2 (default 10.0)
* ``h_bdf2`` — fixed step for BDF2 segments (default 1e-6)

See ``notes/GATE5_PROGRESS.md`` for captured performance across 13
reference scenarios.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, cast


# =============================================================================
# Type aliases (public for downstream type-hints)
# =============================================================================

# Auto = pick per mode via stiffness detector;
# rk45/bdf2 = force one integrator for ALL modes.
IntegratorChoice = Literal["auto", "rk45", "bdf2"]


# =============================================================================
# Resolved options struct (internal — user doesn't construct directly)
# =============================================================================


@dataclass
class DSEDOptions:
    """Fully-resolved DSED options after validation + defaults.

    User-supplied kwargs to :func:`pulsim.simulate` are funneled
    through :func:`_resolve_dsed_options` which produces one of these.
    The scheduler downstream reads ONLY this struct.
    """
    rtol: float = 1e-6
    atol: float = 1e-9
    dt_max: float = 1e-5
    dt_init: float = 1e-9
    integrator: IntegratorChoice = "auto"
    stiffness_threshold: float = 10.0
    h_bdf2: float = 1e-6
    store_every: int = 1
    t_start: float = 0.0
    switch_fn: Optional[Callable[[float], Any]] = None
    initial_state: Optional[Any] = None
    progress: Any = False


# =============================================================================
# Validation
# =============================================================================


_VALID_ENGINES = ("pwl", "dsed")
_VALID_INTEGRATORS = ("auto", "rk45", "bdf2")

# (The dsed-only kwargs that get the leak-warning if user passes
# them while engine='pwl' are enumerated inline in
# _validate_engine_kwargs below — keeping the list local to its
# usage site so it doesn't drift out of sync.)


def _validate_engine_kwargs(
    *,
    engine: str,
    dt: Optional[float],
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    dt_init: Optional[float] = None,
    integrator: Optional[str] = None,
    stiffness_threshold: Optional[float] = None,
    h_bdf2: Optional[float] = None,
) -> None:
    """Defensive validation of engine + related kwargs.

    Raises :class:`ValueError` on the common mistakes; emits a
    :class:`UserWarning` (rather than erroring) on dsed-only kwargs
    leaked into ``engine='pwl'`` so existing scripts that copy-paste
    kwargs don't suddenly break.
    """
    # --- engine name itself ---
    if engine not in _VALID_ENGINES:
        raise ValueError(
            f"simulate(): unknown engine={engine!r}; "
            f"expected one of {_VALID_ENGINES}"
        )

    # --- engine='pwl' branch ---
    if engine == "pwl":
        if dt is None or dt <= 0:
            raise ValueError(
                "simulate(engine='pwl'): a positive `dt` is required. "
                "Either pass `dt=...` for fixed-step trapezoidal, or "
                "switch to `engine='dsed'` for variable-step (no dt "
                "needed)."
            )
        leaked = {
            name: val for name, val in (
                ("rtol", rtol), ("atol", atol),
                ("dt_init", dt_init),
                ("integrator", integrator),
                ("stiffness_threshold", stiffness_threshold),
                ("h_bdf2", h_bdf2),
            )
            if val is not None
        }
        if leaked:
            warnings.warn(
                f"simulate(engine='pwl'): the following kwargs are "
                f"ignored by the fixed-step engine: "
                f"{sorted(leaked.keys())}. Either remove them or "
                f"switch to engine='dsed'.",
                UserWarning,
                stacklevel=3,
            )
        return

    # --- engine='dsed' branch ---
    if rtol is not None and rtol <= 0:
        raise ValueError(
            f"simulate(engine='dsed'): rtol must be positive, got {rtol!r}"
        )
    if atol is not None and atol <= 0:
        raise ValueError(
            f"simulate(engine='dsed'): atol must be positive, got {atol!r}"
        )
    if dt_init is not None and dt_init <= 0:
        raise ValueError(
            f"simulate(engine='dsed'): dt_init must be positive, "
            f"got {dt_init!r}"
        )
    if integrator is not None and integrator not in _VALID_INTEGRATORS:
        raise ValueError(
            f"simulate(engine='dsed'): unknown integrator={integrator!r}; "
            f"expected one of {_VALID_INTEGRATORS}"
        )
    if stiffness_threshold is not None and stiffness_threshold <= 0:
        raise ValueError(
            f"simulate(engine='dsed'): stiffness_threshold must be "
            f"positive, got {stiffness_threshold!r}"
        )
    if h_bdf2 is not None and h_bdf2 <= 0:
        raise ValueError(
            f"simulate(engine='dsed'): h_bdf2 must be positive, "
            f"got {h_bdf2!r}"
        )
    if dt is not None and dt <= 0:
        raise ValueError(
            f"simulate(engine='dsed'): dt (used as dt_max here) must "
            f"be positive, got {dt!r}"
        )


# =============================================================================
# Option resolution (apply defaults + heuristics + user overrides)
# =============================================================================


def _resolve_dsed_options(
    *,
    dt: Optional[float] = None,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    dt_init: Optional[float] = None,
    integrator: Optional[str] = None,
    stiffness_threshold: Optional[float] = None,
    h_bdf2: Optional[float] = None,
    store_every: int = 1,
    t_start: float = 0.0,
    switch_fn: Optional[Callable[[float], Any]] = None,
    initial_state: Optional[Any] = None,
    progress: Any = False,
) -> DSEDOptions:
    """Apply defaults + heuristics to produce a :class:`DSEDOptions`.

    Heuristics (none of which the user has to know about):

    * **dt_max**: if user passed ``dt``, use it. Otherwise default to
      10 µs (covers typical PWM up to ~10 kHz; the PI controller
      adapts down naturally for higher f_sw).
    * **h_bdf2**: default 1 µs (reasonable for stiff segments at
      typical power-electronics scales).
    * **dt_init**: default 1 ns (PI controller grows from here).

    Validation must be called FIRST — this function assumes its
    inputs are sane.
    """
    dt_max = dt if (dt is not None and dt > 0) else 1e-5

    # integrator was already validated to be in _VALID_INTEGRATORS;
    # cast for the type checker.
    resolved_integrator = cast(
        IntegratorChoice,
        integrator if integrator is not None else "auto",
    )

    return DSEDOptions(
        rtol=rtol if rtol is not None else 1e-6,
        atol=atol if atol is not None else 1e-9,
        dt_max=dt_max,
        dt_init=dt_init if dt_init is not None else 1e-9,
        integrator=resolved_integrator,
        stiffness_threshold=(
            stiffness_threshold if stiffness_threshold is not None else 10.0
        ),
        h_bdf2=h_bdf2 if h_bdf2 is not None else 1e-6,
        store_every=store_every,
        t_start=t_start,
        switch_fn=switch_fn,
        initial_state=initial_state,
        progress=progress,
    )


# =============================================================================
# Bridge — CircuitBuilder → PED scheduler
# =============================================================================


# The bridge fallback message is now ONLY surfaced when the underlying
# C++ extractor throws (e.g. a circuit with floating caps that the
# current LTI bridge doesn't support). In that case we want a clear
# explanation + the escape hatch.
_BRIDGE_FALLBACK_MESSAGE = (
    "pulsim.simulate(engine='dsed', builder=...): the C++ extractor "
    "rejected this topology — most commonly because the circuit has "
    "**floating capacitors** (caps between two non-ground nodes; the "
    "current LTI bridge supports caps-to-ground only — Phase 5.1b "
    "TODO in notes/DSED_BRIDGE_DESIGN.md), or because the circuit "
    "has **nonlinear devices** (diodes / MOSFETs / IGBTs / saturable "
    "inductors), which the PED engine doesn't model.\n"
    "\n"
    "For PED today on circuits the bridge can't handle, use "
    "`pulsim.dsed.PEDSimulatorAuto` directly with a user-supplied "
    "LTI system (see the module docstring for an example). Captured "
    "benchmarks across 13 scenarios: ~1× on non-stiff PWM converters "
    "(no penalty), 4-90× on stiff multi-state systems.\n"
    "\n"
    "Resolved DSED options for this call:\n"
    "  rtol = {rtol}, atol = {atol}\n"
    "  dt_max = {dt_max}, integrator = {integrator!r}\n"
    "  stiffness_threshold = {stiffness_threshold}\n"
    "\n"
    "Underlying C++ error: {underlying}"
)


class _PEDSimulationResult:
    """Duck-typed mimic of :class:`pulsim.SimulationResult` populated
    by the PED scheduler.

    Pulsim's C++ ``SimulationResult`` has read-only pybind11 properties
    (``def_readonly``), so we can't construct one from Python and
    populate its fields. This wrapper provides the SAME public
    interface (``times``, ``states``, ``num_steps()``, ``empty()``)
    so downstream code that consumes ``simulate(...)`` results works
    regardless of whether the engine was ``pwl`` or ``dsed``.

    The PED-specific fields are exposed as additional attributes:

    * ``n_accept`` / ``n_reject`` — PI controller accept/reject counts
    * ``n_events`` — number of mask transitions fired
    * ``n_rk45_steps`` / ``n_bdf2_steps`` — per-integrator step counts
      (only populated when ``integrator='auto'`` was selected)
    * ``cpu_time_seconds`` — wall-clock from the PED scheduler
    * ``event_iteration_count`` / ``commutation_events`` — empty lists
      (PED uses a different event model)
    """

    def __init__(
        self,
        times: list,
        states: list,
        *,
        n_accept: int = 0,
        n_reject: int = 0,
        n_events: int = 0,
        n_rk45_steps: int = 0,
        n_bdf2_steps: int = 0,
        cpu_time_seconds: float = 0.0,
    ) -> None:
        self.times = times
        self.states = states
        self.n_accept = n_accept
        self.n_reject = n_reject
        self.n_events = n_events
        self.n_rk45_steps = n_rk45_steps
        self.n_bdf2_steps = n_bdf2_steps
        self.cpu_time_seconds = cpu_time_seconds
        # PWL-engine diagnostic fields — empty for PED, but exposed
        # for API-compat with code that destructures both result types.
        self.event_iteration_count: list = []
        self.commutation_events: list = []

    def num_steps(self) -> int:
        return len(self.times)

    def empty(self) -> bool:
        return len(self.times) == 0

    def __repr__(self) -> str:
        return (
            f"_PEDSimulationResult(n_steps={len(self.times)}, "
            f"n_accept={self.n_accept}, n_events={self.n_events}, "
            f"cpu={self.cpu_time_seconds * 1e3:.2f}ms)"
        )


def run_dsed_from_builder(
    *,
    builder: Any,                          # pulsim.CircuitBuilder
    t_end: float,
    dt: Optional[float] = None,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    dt_init: Optional[float] = None,
    integrator: Optional[str] = None,
    stiffness_threshold: Optional[float] = None,
    h_bdf2: Optional[float] = None,
    store_every: int = 1,
    t_start: float = 0.0,
    switch_fn: Optional[Callable[[float], Any]] = None,
    b_extra_fn: Optional[Callable[[float], Any]] = None,
    initial_state: Optional[Any] = None,
    progress: Any = False,
):
    """Run a PED simulation from a :class:`pulsim.CircuitBuilder`.

    **Current status**: raises :class:`NotImplementedError` because the
    pybind11 bridge that exposes ``(A, b)`` per mask from
    ``PwlStateSpaceCache`` hasn't landed yet. The error message
    includes the resolved DSED options + a pointer to the working
    user-supplied-LTI path (:class:`pulsim.dsed.PEDSimulatorAuto`).

    The implementation path forward (~3-5 days of focused work):

    1. **Extend the pybind11 binding for ``PwlStateSpaceCache``** to
       expose ``state_space_for_mask(mask) -> tuple[ndarray, ndarray]``
       (dense `n_state × n_state` A + `n_state` b vector) and
       ``n_state`` read-only.
    2. **Implement ``CircuitBuilderAdapter``** in
       ``pulsim/dsed/_builder_bridge.py`` that wraps a
       ``CircuitBuilder`` + ``PwlStateSpaceCache`` to satisfy the
       ``HasLTIPerMode`` C++ contract.
    3. **Wire ``PEDSimulator`` / ``PEDSimulatorAuto``** to consume
       the adapter, switched on ``opts.integrator``.
    4. **Convert ``PEDResult`` → ``SimulationResult``** so callers
       get back the same return type as ``engine='pwl'``.

    Once those land, this function will route to the right scheduler
    (RK45-only, BDF2-only, or auto-dispatch) based on
    ``opts.integrator``, and the NotImplementedError disappears.
    """
    import numpy as np

    opts = _resolve_dsed_options(
        dt=dt,
        rtol=rtol, atol=atol, dt_init=dt_init,
        integrator=integrator,
        stiffness_threshold=stiffness_threshold,
        h_bdf2=h_bdf2, store_every=store_every,
        t_start=t_start, switch_fn=switch_fn,
        initial_state=initial_state, progress=progress,
    )

    # ---- Stage 1: build the cache + adapter ---------------------------
    try:
        from . import PwlStateSpaceCache as _PwlCache  # type: ignore[attr-defined]
        from . import SwitchStateMask  # type: ignore[attr-defined]
        from .dsed._builder_bridge import CircuitBuilderAdapter
    except ImportError as e:
        raise RuntimeError(
            f"pulsim.simulate(engine='dsed'): required Pulsim bindings "
            f"not available ({e}). Either the install is broken or "
            f"the Pulsim Python package was built without the dsed "
            f"sub-package. See docs/dsed.md for installation notes."
        ) from e

    cache = _PwlCache(builder.graph, builder.pool)
    cache.build(opts.dt_max)

    # Default switch_fn — all switches closed (matches the engine='pwl'
    # default behaviour).
    sf = opts.switch_fn
    if sf is None:
        n_sw = builder.graph.num_switches
        default_mask = SwitchStateMask(n_sw)
        for i in range(n_sw):
            default_mask.set(i, True)
        sf = lambda _t: default_mask  # noqa: E731

    adapter = CircuitBuilderAdapter(
        builder=builder, cache=cache, switch_fn=sf,
        b_extra_fn=b_extra_fn)

    # Eagerly resolve the initial mask so any extractor failure
    # (floating cap, nonlinear device, etc.) surfaces NOW with the
    # bridge-fallback message rather than mid-simulation.
    initial_mask = sf(opts.t_start)
    adapter.set_mask(initial_mask)
    try:
        _ = adapter.A_matrix()
    except NotImplementedError as e:
        raise NotImplementedError(_BRIDGE_FALLBACK_MESSAGE.format(
            rtol=opts.rtol,
            atol=opts.atol,
            dt_max=opts.dt_max,
            integrator=opts.integrator,
            stiffness_threshold=opts.stiffness_threshold,
            underlying=str(e).split('\n')[0],
        )) from e

    # ---- Stage 3: initial state -------------------------------------
    n_state = adapter.n_state
    if opts.initial_state is not None:
        x0 = np.asarray(opts.initial_state, dtype=float)
        if x0.shape != (n_state,):
            raise ValueError(
                f"simulate(engine='dsed'): initial_state has shape "
                f"{x0.shape}, expected ({n_state},)."
            )
    else:
        x0 = np.zeros(n_state, dtype=float)

    t_window = float(t_end) - float(opts.t_start)
    if t_window <= 0:
        raise ValueError(
            f"simulate(engine='dsed'): t_end ({t_end}) must be > "
            f"t_start ({opts.t_start})."
        )

    # ---- Stage 4: prefer Bridge.11 (full-native adapter + scheduler),
    # fall back to Bridge.10 (Python adapter + native scheduler),
    # fall back to Bridge.5 (Python adapter + Python scheduler).
    _native = None
    try:
        from . import _pulsim as _native_mod   # type: ignore[attr-defined]
        _native = _native_mod
    except ImportError:
        _native = None

    NativeAdapterCls = (getattr(_native, "_NativeCircuitBuilderAdapter", None)
                        if _native is not None else None)
    run_ped_b11 = (getattr(_native, "run_ped_native_builder", None)
                   if _native is not None else None)
    run_bdf2_b11 = (getattr(_native, "run_bdf2_native_builder", None)
                    if _native is not None else None)
    run_auto_b11 = (getattr(_native, "run_auto_native_builder", None)
                    if _native is not None else None)
    bridge11_ok = (NativeAdapterCls is not None
                   and run_ped_b11 is not None
                   and run_bdf2_b11 is not None
                   and run_auto_b11 is not None)

    if bridge11_ok:
        try:
            native_adapter = NativeAdapterCls(
                builder, cache,
                b_extra_fn if b_extra_fn is not None else None)
            if opts.integrator == "rk45":
                native_res = run_ped_b11(
                    native_adapter, sf, x0, t_window,
                    rtol=opts.rtol, atol=opts.atol,
                    dt_init=opts.dt_init, dt_max=opts.dt_max,
                    store_every=opts.store_every)
            elif opts.integrator == "bdf2":
                native_res = run_bdf2_b11(
                    native_adapter, sf, x0, t_window,
                    h_fixed=opts.h_bdf2,
                    store_every=opts.store_every)
            else:  # 'auto'
                native_res = run_auto_b11(
                    native_adapter, sf, x0, t_window,
                    rtol=opts.rtol, atol=opts.atol,
                    dt_init=opts.dt_init, dt_max=opts.dt_max,
                    h_bdf2=opts.h_bdf2,
                    stiffness_threshold=opts.stiffness_threshold,
                    store_every=opts.store_every)
            times_arr = native_res["times"]
            states_arr = native_res["states"]
            times = [float(t) + float(opts.t_start) for t in times_arr]
            states = [np.asarray(states_arr[i], dtype=float)
                      for i in range(len(times))]
            return _PEDSimulationResult(
                times=times,
                states=states,
                n_accept=native_res.get("n_accept", 0),
                n_reject=native_res.get("n_reject", 0),
                n_events=native_res.get("n_events", 0),
                n_rk45_steps=native_res.get("n_rk45_steps", 0),
                n_bdf2_steps=native_res.get("n_bdf2_steps", 0),
                cpu_time_seconds=native_res.get("cpu_time_seconds", 0.0),
            )
        except (RuntimeError, ValueError):
            # Bridge.11 native adapter rejected the circuit (e.g. nonlinear
            # device that the C++ extractor can't handle). Silently fall
            # through to Bridge.10 — the Python adapter has identical
            # semantics + the same error path on the same circuits.
            pass

    # Bridge.10 path — Python adapter + native scheduler.
    run_ped_native = (getattr(_native, "run_ped_native", None)
                      if _native is not None else None)
    run_bdf2_native = (getattr(_native, "run_bdf2_native", None)
                       if _native is not None else None)
    run_auto_native = (getattr(_native, "run_auto_native", None)
                       if _native is not None else None)
    bridge10_ok = (run_ped_native is not None
                   and run_bdf2_native is not None
                   and run_auto_native is not None)

    if bridge10_ok:
        if opts.integrator == "rk45":
            native_res = run_ped_native(
                adapter, sf, x0, t_window,
                rtol=opts.rtol, atol=opts.atol,
                dt_init=opts.dt_init, dt_max=opts.dt_max,
                store_every=opts.store_every)
        elif opts.integrator == "bdf2":
            native_res = run_bdf2_native(
                adapter, sf, x0, t_window,
                h_fixed=opts.h_bdf2,
                store_every=opts.store_every)
        else:  # 'auto'
            native_res = run_auto_native(
                adapter, sf, x0, t_window,
                rtol=opts.rtol, atol=opts.atol,
                dt_init=opts.dt_init, dt_max=opts.dt_max,
                h_bdf2=opts.h_bdf2,
                stiffness_threshold=opts.stiffness_threshold,
                store_every=opts.store_every)
        times_arr = native_res["times"]
        states_arr = native_res["states"]
        times = [float(t) + float(opts.t_start) for t in times_arr]
        states = [np.asarray(states_arr[i], dtype=float)
                  for i in range(len(times))]
        return _PEDSimulationResult(
            times=times,
            states=states,
            n_accept=native_res.get("n_accept", 0),
            n_reject=native_res.get("n_reject", 0),
            n_events=native_res.get("n_events", 0),
            n_rk45_steps=native_res.get("n_rk45_steps", 0),
            n_bdf2_steps=native_res.get("n_bdf2_steps", 0),
            cpu_time_seconds=native_res.get("cpu_time_seconds", 0.0),
        )

    # ---- Stage 4 fallback: pure-Python scheduler (Bridge.5) --------
    # Used when the C++ native bindings aren't built (e.g. dev install
    # without recompile). Identical semantics, ~3× slower wall-clock.
    from .dsed import (
        PEDSimulator,
        PEDSimulatorBDF2,
        PEDSimulatorAuto,
        PIController,
        EventPredictor,
        StiffnessDetector,
    )
    if opts.integrator == "rk45":
        sim = PEDSimulator(
            system=adapter, switch_fn=sf,
            controller=PIController(rtol=opts.rtol, atol=opts.atol),
            predictor=EventPredictor(),
            dt_init=opts.dt_init, dt_max=opts.dt_max,
            store_every=opts.store_every,
        )
    elif opts.integrator == "bdf2":
        sim = PEDSimulatorBDF2(
            system=adapter, switch_fn=sf,
            h_fixed=opts.h_bdf2,
            store_every=opts.store_every,
        )
    else:  # 'auto'
        sim = PEDSimulatorAuto(
            system=adapter, switch_fn=sf,
            controller=PIController(rtol=opts.rtol, atol=opts.atol),
            detector=StiffnessDetector(opts.stiffness_threshold),
            dt_init=opts.dt_init, dt_max=opts.dt_max,
            h_bdf2=opts.h_bdf2,
            store_every=opts.store_every,
        )
    ped_result = sim.simulate(x0, t_window)
    times = [float(t) + float(opts.t_start) for t in ped_result.times]
    states = [np.asarray(x, dtype=float) for x in ped_result.states]
    return _PEDSimulationResult(
        times=times,
        states=states,
        n_accept=getattr(ped_result, "n_accept", 0),
        n_reject=getattr(ped_result, "n_reject", 0),
        n_events=getattr(ped_result, "n_events", 0),
        n_rk45_steps=getattr(ped_result, "n_rk45_steps", 0),
        n_bdf2_steps=getattr(ped_result, "n_bdf2_steps", 0),
        cpu_time_seconds=getattr(ped_result, "cpu_time", 0.0),
    )


__all__ = [
    "IntegratorChoice",
    "DSEDOptions",
    "_validate_engine_kwargs",
    "_resolve_dsed_options",
    "run_dsed_from_builder",
]
