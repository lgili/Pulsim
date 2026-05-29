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

from ._pulsim import (  # type: ignore[import-not-found]
    NativeMultiMaskPwm,
    NativePwm2Switch,
)


# ---------------------------------------------------------------------------
# _PolledPySwitchFn — fixes a silent-wrong-answer bug on plain Python
# switch_fns in the DSED engine.
# ---------------------------------------------------------------------------
class _PolledPySwitchFn:
    """Wraps a Python ``switch_fn`` so the C++ DSED scheduler polls it.

    **The bug this exists to fix.** The C++ DSED scheduler computes the
    next gate-edge time via ``switch_fn.next_edge_after(t)`` (see
    ``core/include/pulsim/dsed/scheduler*.hpp``). The native PWM
    classes (:class:`NativePwm2Switch`, :class:`NativeMultiMaskPwm`)
    implement that method exactly; plain Python callables don't.
    The pybind11 adapter ``PySwitchFn::next_edge_after`` falls back to
    ``std::numeric_limits<Real>::infinity()`` in that case, which the
    scheduler caps at ``t_end``. The integrator's step size is then
    bounded only by the PI controller + ``dt_max`` — and the scheduler
    **never re-queries the user's switch_fn between accepted steps**.

    For a PWM circuit, this means the scheduler integrates the *whole
    window* with the mask sampled at ``t = t_start`` — silently
    producing the trajectory of a circuit with the switch frozen
    instead of modulated. The user sees a DC-like steady state at the
    wrong setpoint and (reasonably) suspects DSED is "stuck".

    **What this wrapper does.** Reports ``next_edge_after(t) = t +
    dt_max/2``, forcing the scheduler to call ``switch_fn`` at most
    every ``dt_max/2``. The scheduler then resolves the mask change as
    a gate event, refactors ``(A, b)``, and advances correctly. The
    cost is one extra Python callback per ``dt_max/2`` interval —
    significant overhead but the only way to get a *correct* result
    when the caller can't tell us when the next edge will be.

    **Performance escape hatch.** For PWM-driven simulations, wrap your
    switch_fn in :class:`pulsim.NativePwm2Switch` or
    :class:`pulsim.NativeMultiMaskPwm` instead. Those expose
    ``next_edge_after`` analytically and avoid the polling overhead —
    the scheduler then only re-queries on real gate edges (~25× faster
    per scheduler step for dense PWM).
    """

    __slots__ = ("_fn", "_step")

    def __init__(self, fn: Callable[[float], Any], dt_max: float):
        self._fn = fn
        # The poll step has to be small enough that two consecutive
        # mask transitions can't fall between polls without being
        # noticed. PWM at frequency `f_sw` has edges every
        # `T_sw/2 = 1/(2·f_sw)` seconds. Empirically, polling at
        # `dt_max/10` captures every edge for the common case where
        # the user picks `dt_max ≈ T_sw` (the same heuristic the
        # buck/boost helpers' `suggested_dt` follows). Stricter
        # PWMs need either a tighter `dt`/`dt_max` from the user or
        # — much better — wrapping in `pulsim.NativePwm2Switch` /
        # `NativeMultiMaskPwm`, both of which expose an analytical
        # `next_edge_after` so the scheduler never has to poll.
        self._step = float(dt_max) / 10.0

    def __call__(self, t: float):
        return self._fn(t)

    def next_edge_after(self, t: float) -> float:
        return float(t) + self._step


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
    "`pulsim.dsed.run_user_lti(system, switch_fn, x0, t_end, ...)` "
    "with a hand-rolled LTI system (see "
    "`pulsim.dsed.run_user_lti.__doc__` for the 5-method system "
    "protocol). Captured "
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

    Production path. Tries the native C++ adapter (Bridge.11) and
    falls back to the Python adapter + native scheduler (Bridge.10)
    when the C++ extractor rejects the circuit. If both reject,
    raises :class:`NotImplementedError` with a pointer at the
    user-supplied-LTI escape hatch (:func:`pulsim.dsed.run_user_lti`).

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

    # Auto-wrap a plain Python switch_fn so the C++ event predictor
    # polls it at every `dt_max/10` and detects mask changes via
    # comparison. Without this, switch_fns that don't expose
    # `next_edge_after(t)` get scheduled at `t_end` (== ∞), so the
    # scheduler integrates the whole window with the *initial* mask —
    # silently producing the wrong trajectory for PWM / state-machine
    # modulation. See `_PolledPySwitchFn` for the exact contract.
    #
    # Emit a one-time warning recommending the native PWM classes for
    # PWM-heavy workloads (they expose `next_edge_after` analytically
    # and avoid the polling overhead entirely).
    if (not isinstance(sf, (
            NativePwm2Switch, NativeMultiMaskPwm))
            and not hasattr(sf, "next_edge_after")):
        warnings.warn(
            "simulate(engine='dsed'): wrapping a plain Python "
            "switch_fn for polled event detection (dt_max/10 = "
            f"{opts.dt_max / 10:.3g} s). This is correct but slow "
            "— each scheduler step calls back into Python. For "
            "best performance on PWM-driven simulations, wrap "
            "your switch logic in `pulsim.NativePwm2Switch` or "
            "`pulsim.NativeMultiMaskPwm`, which expose "
            "`next_edge_after` analytically and let the scheduler "
            "skip directly to the next gate edge.",
            UserWarning,
            stacklevel=3,
        )
        sf = _PolledPySwitchFn(sf, opts.dt_max)

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

    # ---- No fallback. ----
    # The Bridge.11 native-builder path and the Bridge.10
    # Python-adapter-+-native-scheduler path together cover every
    # CircuitBuilder we currently know how to extract an (A, b)
    # state-space from. If we reach this line, both rejected the
    # circuit — typically due to a nonlinear device the C++
    # extractor doesn't yet support. The pure-Python scheduler
    # fallback that used to live here was demoted to the test
    # tree in v1.6.3 (see `python/tests/_dsed_reference/`); the
    # production path is C++-only.
    raise NotImplementedError(
        "pulsim.simulate(engine='dsed'): both the native C++ "
        "builder adapter (Bridge.11) and the Python builder "
        "adapter (Bridge.10) rejected this circuit. This usually "
        "means the circuit contains a nonlinear device or an "
        "unsupported topology that the per-mask (A, b) extractor "
        "can't handle. Two options: (a) switch to "
        "engine='pwl' which handles the same circuits via the "
        "trap-companion path; or (b) build a hand-rolled LTI "
        "system that satisfies `pulsim.dsed.run_user_lti`'s "
        "system protocol and drive the native scheduler "
        "directly. Please open an issue describing the failing "
        "circuit so we can extend the extractor."
    )


__all__ = [
    "IntegratorChoice",
    "DSEDOptions",
    "_validate_engine_kwargs",
    "_resolve_dsed_options",
    "run_dsed_from_builder",
]
