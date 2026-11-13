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
* ``integrator`` — ``'auto'`` (default), ``'rk45'``, or ``'bdf2'``.
  ``'auto'`` takes, in order: the exact (closed-form) stepper when the
  mode has one; otherwise the stiffness detector's choice of DOPRI5 or
  BDF2. ``'rk45'`` also takes the exact stepper when it exists.
* ``stiffness_threshold`` — when ``|λ_max|·h > threshold``, dispatcher
  picks BDF2 (default 10.0) for a mode with no exact stepper
* ``h_bdf2`` — fixed step for BDF2 segments (default 1e-6)

See ``notes/GATE5_PROGRESS.md`` for captured performance across 13
reference scenarios.
"""

from __future__ import annotations

import warnings

import numpy as _np
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, cast

from ._pulsim import (  # type: ignore[import-not-found]
    NativeMultiMaskPwm,
    NativePwm2Switch,
    SwitchSchedule,
    UnsupportedError,
)
from ._refusal import _UnsupportedNotImplemented, unsupported


# =============================================================================
# Type aliases (public for downstream type-hints)
# =============================================================================

# Auto = exact stepper when the mode has one, else the stiffness
# detector's DOPRI5/BDF2 choice; rk45/bdf2 = force one integrator for
# ALL modes (rk45 still takes the exact stepper where it exists).
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


_VALID_ENGINES = ("auto", "pwl", "dsed", "trbdf2")
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

    # --- engine='auto' / 'trbdf2' branch. 'auto' ROUTES (to the
    #     variable-step engine when the circuit qualifies, to the
    #     fixed-step one otherwise); 'trbdf2' names the
    #     variable-step engine explicitly and refuses instead of
    #     routing. dt is optional in both and acts as the step
    #     ceiling h_max. ---
    if engine in ("auto", "trbdf2"):
        if dt is not None and dt <= 0:
            raise ValueError(
                f"simulate(engine={engine!r}): dt (used as the step "
                f"ceiling h_max here) must be positive, got {dt!r}")
        if rtol is not None and rtol <= 0:
            raise ValueError(
                f"simulate(engine={engine!r}): rtol must be "
                f"positive, "
                f"got {rtol!r}")
        if atol is not None and atol <= 0:
            raise ValueError(
                f"simulate(engine={engine!r}): atol must be "
                f"positive, "
                f"got {atol!r}")
        if dt_init is not None and dt_init <= 0:
            raise ValueError(
                f"simulate(engine={engine!r}): dt_init must be "
                f"positive, "
                f"got {dt_init!r}")
        leaked = {
            name: val for name, val in (
                ("integrator", integrator),
                ("stiffness_threshold", stiffness_threshold),
                ("h_bdf2", h_bdf2),
            )
            if val is not None
        }
        if leaked:
            warnings.warn(
                f"simulate(engine={engine!r}): the following kwargs "
                "belong to the dsed engine and are ignored here: "
                f"{sorted(leaked.keys())}.",
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
    * ``n_rk45_steps`` / ``n_bdf2_steps`` / ``n_exact_steps`` —
      per-integrator step counts (``n_exact_steps``: closed-form LTI
      propagator)
      (only populated when ``integrator='auto'`` was selected)
    * ``cpu_time_seconds`` — wall-clock from the PED scheduler
    * ``event_iteration_count`` / ``commutation_events`` — empty lists
      (PED uses a different event model)
    * ``switch_states`` — the per-sample mode mask, ``(n, n_switches)``
      bool, the same record the pwl / trbdf2 engines carry. Each full
      MNA sample is reconstructed THROUGH that sample's mask, so a
      switched branch's current is exactly ``g(bit) · v`` there.
      ``None`` when the kernel returned only reduced states (no node
      voltages to apply it to) or no typed mask (the Python-adapter
      path, which never carries PWL diodes).
    * ``right_limit_index`` / ``right_limit_states`` /
      ``right_limit_switch_states`` — at every sample where an event
      fired, the RIGHT limit: the full MNA state and the mask the next
      step starts from (the post-event state rebuilt through the
      post-event mask). The sample itself is the LEFT limit, the state
      the step that ended there produced, under ``switch_states[k]``.
      A switched branch's current jumps between the two, so a time
      integral must start each step from the right limit of its first
      sample (``pulsim._result_views.step_start_states``); a trapezoid
      from the left limit read a DCM boost diode's average current at
      0.0234 A against 0.0535 A. Empty when no event fired, ``None``
      when ``switch_states`` is ``None``.
    """

    def __init__(
        self,
        times: list,
        states: list,
        *,
        states_full=None,
        switch_states=None,
        right_limit_index=None,
        right_limit_states=None,
        right_limit_switch_states=None,
        n_accept: int = 0,
        n_reject: int = 0,
        n_events: int = 0,
        n_rk45_steps: int = 0,
        n_bdf2_steps: int = 0,
        n_exact_steps: int = 0,
        cpu_time_seconds: float = 0.0,
    ) -> None:
        self.times = times
        # v2.0 Phase 3 item 5 — THE UNIFIED RESULT. When the kernel
        # hands back the reconstructed full-MNA trajectory (node
        # voltages, then source and inductor branch currents — the
        # pwl engine's exact row layout), `.states` IS that, so
        # `res.states[:, node_id]`, `res.v('name')` and
        # `res.i('name')` mean the same thing on both engines. The
        # reduced integrator state stays available as
        # `.states_reduced` for anyone studying the modes
        # themselves. Read-only either way: a result is an output,
        # not a scratch pad.
        arr = _np.asarray(states, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(len(times), -1) if times else arr.reshape(0, 0)
        arr.setflags(write=False)
        full = None
        if states_full is not None:
            full = _np.asarray(states_full, dtype=float)
            if full.size == 0 or full.ndim != 2                     or full.shape[0] != len(times):
                full = None
            else:
                full.setflags(write=False)
        self.states_reduced = arr
        self.states = full if full is not None else arr
        self.has_full_states = full is not None
        self.n_accept = n_accept
        self.n_reject = n_reject
        self.n_events = n_events
        self.n_rk45_steps = n_rk45_steps
        self.n_bdf2_steps = n_bdf2_steps
        self.n_exact_steps = n_exact_steps
        self.cpu_time_seconds = cpu_time_seconds
        # add-exact-propagator-strategy D10: per visited mask the
        # propagator tier (1 eig, 2 Pade, 3 none), kappa2 and the reason
        # for any tier but 1, plus n_eig / n_pade / n_phi_cache_hits /
        # n_predicate_evals. Event instants and predicate types as
        # arrays. Set by the native builder path; None elsewhere.
        self.exact = None
        self.initial_condition_adjustment = None
        self.event_times = None
        self.event_types = None
        # PWL-engine diagnostic fields — empty for PED, but exposed
        # for API-compat with code that destructures both result types.
        self.event_iteration_count: list = []
        self.commutation_events: list = []
        sw = None
        if switch_states is not None and full is not None:
            sw = _np.asarray(switch_states, dtype=bool)
            if sw.ndim != 2 or sw.shape[0] != len(times):
                sw = None
            else:
                sw.setflags(write=False)
        self.switch_states = sw
        self.switch_states_unsolved = (
            _np.zeros(len(times), dtype=bool) if sw is not None else None)
        rl_idx = rl_states = rl_sw = None
        if sw is not None and right_limit_index is not None:
            rl_idx = _np.asarray(right_limit_index, dtype=_np.int64)
            rl_states = _np.asarray(right_limit_states, dtype=float)
            rl_sw = _np.asarray(right_limit_switch_states, dtype=bool)
            if rl_idx.size == 0:
                rl_states = _np.zeros((0, full.shape[1]))
                rl_sw = _np.zeros((0, sw.shape[1]), dtype=bool)
            ok = (rl_idx.ndim == 1
                  and rl_states.shape == (rl_idx.size, full.shape[1])
                  and rl_sw.shape == (rl_idx.size, sw.shape[1])
                  and (rl_idx.size == 0
                       or (rl_idx.min() >= 0
                           and rl_idx.max() < len(times)
                           and _np.all(_np.diff(rl_idx) > 0))))
            if ok:
                for a in (rl_idx, rl_states, rl_sw):
                    a.setflags(write=False)
            else:
                rl_idx = rl_states = rl_sw = None
        self.right_limit_index = rl_idx
        self.right_limit_states = rl_states
        self.right_limit_switch_states = rl_sw

    @property
    def states_bytes(self) -> int:
        """Bytes held by the sample buffer (parity with the PWL
        engine's ``SimulationResult.states_bytes``)."""
        return int(self.states.nbytes)

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


def _has_edge_oracle(sf) -> bool:
    """Whether the DSED scheduler can ask ``sf`` for its next gate edge."""
    if isinstance(sf, (NativePwm2Switch, NativeMultiMaskPwm)):
        return True
    if isinstance(sf, SwitchSchedule):
        return bool(sf.has_edges)
    return hasattr(sf, "next_edge_after")


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
    builder_initial_state: Optional[Any] = None,
):
    """Run a PED simulation from a :class:`pulsim.CircuitBuilder`.

    ``builder_initial_state`` (not None) says the builder declared
    ``c0=`` / ``i0=`` / ``set_initial`` values; they are mapped per
    device onto the reduced state (add-exact-propagator-strategy D5)
    and the largest capacitor-voltage adjustment is
    ``result.initial_condition_adjustment``.

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

    # ------------------------------------------------------------------
    # v2.0 Phase 3 — structural refusal, BEFORE any work happens.
    #
    # The LTI extraction has no representation for either device
    # class below, and neither failure is loud:
    #
    #   * A PWL switched diode's bit is frozen at whatever switch_fn
    #     returns — measured: a reverse-biased series diode CONDUCTS
    #     BACKWARDS (vout = -10.909 V where the pwl engine correctly
    #     blocks at -1e-06 V). A rectifier run under dsed simulates a
    #     circuit with no rectification and reports it as fact.
    #   * A Nonlinear device (smooth diode / MOSFET / IGBT /
    #     saturable inductor) is skipped by the assembler entirely —
    #     measured: a 5 V source through 1 kΩ into a nonlinear diode
    #     just charges the cap toward 5 V as if the diode were absent.
    #
    # The fallback message further down claims the C++ extractor
    # rejects nonlinear devices; it does not — extraction SUCCEEDS on
    # the wrong circuit, which is why this check must be structural
    # rather than an exception handler. Same stance as
    # run_transient_bdf1's PWL-diode refusal (v2.0 Phase 2).
    # ------------------------------------------------------------------
    from . import _pulsim as _kern
    _n_sw, _n_diodes, _controlled = _kern._switch_census(
        builder.graph, builder.pool)
    if _n_diodes > 0:
        # v2.0 Phase 3 — PWL diodes are commutated by auto-derived
        # event predicates, which live on the RK45 scheduler
        # (PEDSimulator). The BDF2 and auto schedulers have no
        # predicate scan, so a diode circuit routed there would
        # freeze the diode bits — the silent wrong answer this
        # dispatch used to refuse outright. Route accordingly:
        # 'auto' quietly becomes 'rk45'; an EXPLICIT 'bdf2' request
        # is refused rather than silently downgraded.
        if integrator == "bdf2":
            raise unsupported(
                "dsed", "integrator", kind="Diode", message=
                f"simulate(engine='dsed', integrator='bdf2'): this "
                f"circuit has {_n_diodes} PWL diode(s), and diode "
                f"commutation is only implemented on the RK45 "
                f"scheduler — under bdf2 the diode bits would be "
                f"frozen at whatever switch_fn returns and a "
                f"rectifier would run with no rectification. Drop "
                f"integrator='bdf2' (the default routes correctly) "
                f"or use engine='pwl'.")
        integrator = "rk45"
    # Device kinds: the kernel's capability table, before any adapter
    # is built (the Python adapter below would extract the LTI system
    # with every Newton device silently ABSENT). The native adapter
    # repeats the check for C++ callers.
    _kern._require_supported(builder.graph, builder.pool, "dsed")

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
    # Lazy build (review finding PY-3): the DSED adapter derives its
    # per-mask LTI entries via compute_lti_state_space and never reads
    # the pre-built trap segments, so the eager 2^N enumeration was
    # pure waste — and made many-switch circuits hang before t=0.
    cache.build_lazy(opts.dt_max)

    # Default switch_fn — all switches closed (matches the engine='pwl'
    # default behaviour, INCLUDING its driverless-controlled-switch
    # warning — review finding PY-3).
    sf = opts.switch_fn
    if sf is None:
        n_sw = builder.graph.num_switches
        try:
            from ._pulsim import _switch_census
            _, _nd, _controlled = _switch_census(
                builder.graph, builder.pool)
        except Exception:  # noqa: BLE001
            _controlled = []
        if _controlled:
            warnings.warn(
                f"simulate(engine='dsed'): no switch_fn was given, "
                f"but the circuit has {len(_controlled)} controlled "
                f"switch(es) (indices {list(_controlled)}). They "
                "default to ALL CLOSED, which short-circuits bridge "
                "legs (shoot-through). Pass an explicit switch_fn.",
                stacklevel=3)
        default_mask = SwitchStateMask(n_sw)
        for i in range(n_sw):
            default_mask.set(i, True)
        sf = lambda _t: default_mask  # noqa: E731

    adapter = CircuitBuilderAdapter(
        builder=builder, cache=cache, switch_fn=sf,
        b_extra_fn=b_extra_fn)

    # A switch_fn that cannot name its next edge is sampled every
    # dt_max/10 (the "poll"), and each change found is bisected to the
    # edge (dsed_bindings.cpp::EdgeSearch). Before the bisection every
    # edge moved onto the poll grid: at the default dt_max a 100 ns
    # dead time vanished (a synchronous buck's body diodes never
    # conducted) and D = 0.25 ran as 0.30 (14.28 V against 11.49 V).
    # A pulse that starts and ends between two polls is still missed,
    # so say so. The built-in helpers return a SwitchSchedule, whose
    # edges the scheduler lands on directly. The default all-closed
    # mask (no switch_fn given) is constant: nothing to warn about.
    if opts.switch_fn is not None and not _has_edge_oracle(sf):
        poll = opts.dt_max / 10
        warnings.warn(
            "simulate(engine='dsed'): this switch_fn has no "
            "next_edge_after(t), so the scheduler samples it every "
            f"{poll:.3g} s (= dt_max/10) and bisects each change it "
            "finds to the edge. A pulse that starts and ends between "
            f"two samples (shorter than {poll:.3g} s) is missed. The "
            "built-in helpers (make_pwm_switch_fn, "
            "make_dead_time_pwm_pair_fn, make_spwm_pair_fn, "
            "make_phase_shift_full_bridge_fn, make_three_phase_spwm_fn) "
            "return a SwitchSchedule whose edges the scheduler lands "
            "on exactly; otherwise pass a smaller dt or give the "
            "callable a next_edge_after(t) method.",
            UserWarning,
            stacklevel=3,
        )

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
        raise unsupported(
            "dsed", "extraction", cls=_UnsupportedNotImplemented,
            message=_BRIDGE_FALLBACK_MESSAGE.format(
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
            ic_adjustment = None
            if builder_initial_state is not None:
                native_adapter.set_mask(initial_mask)
                x0, ic_adjustment = native_adapter.reduced_initial_state(
                    builder)
            if opts.integrator == "rk45":
                native_res = run_ped_b11(
                    native_adapter, sf, x0, t_window,
                    rtol=opts.rtol, atol=opts.atol,
                    dt_init=opts.dt_init, dt_max=opts.dt_max,
                    store_every=opts.store_every, event_log=False)
            elif opts.integrator == "bdf2":
                native_res = run_bdf2_b11(
                    native_adapter, sf, x0, t_window,
                    h_fixed=opts.h_bdf2,
                    store_every=opts.store_every, event_log=False)
            else:  # 'auto'
                native_res = run_auto_b11(
                    native_adapter, sf, x0, t_window,
                    rtol=opts.rtol, atol=opts.atol,
                    dt_init=opts.dt_init, dt_max=opts.dt_max,
                    h_bdf2=opts.h_bdf2,
                    stiffness_threshold=opts.stiffness_threshold,
                    store_every=opts.store_every, event_log=False)
            # One array conversion, not one per sample.
            times = (np.asarray(native_res["times"], dtype=float)
                     + float(opts.t_start)).tolist()
            res = _PEDSimulationResult(
                times=times,
                states=native_res["states"],
                states_full=native_res.get("states_full"),
                switch_states=native_res.get("switch_states"),
                right_limit_index=native_res.get("right_limit_index"),
                right_limit_states=native_res.get("right_limit_states"),
                right_limit_switch_states=native_res.get(
                    "right_limit_switch_states"),
                n_accept=native_res.get("n_accept", 0),
                n_reject=native_res.get("n_reject", 0),
                n_events=native_res.get("n_events", 0),
                n_rk45_steps=native_res.get("n_rk45_steps", 0),
                n_bdf2_steps=native_res.get("n_bdf2_steps", 0),
                n_exact_steps=native_res.get("n_exact_steps", 0),
                cpu_time_seconds=native_res.get("cpu_time_seconds", 0.0),
            )
            res.exact = native_res.get("exact")
            res.initial_condition_adjustment = ic_adjustment
            et = native_res.get("event_times")
            res.event_times = (None if et is None
                               else np.asarray(et) + float(opts.t_start))
            res.event_types = native_res.get("event_types")
            return res
        except UnsupportedError:
            # A refusal is an answer, not a reason to try the next
            # bridge (which would run the circuit without the device).
            raise
        except (RuntimeError, ValueError):
            # Bridge.11 native adapter rejected the circuit (e.g. nonlinear
            # device that the C++ extractor can't handle). Silently fall
            # through to Bridge.10 — the Python adapter has identical
            # semantics + the same error path on the same circuits.
            #
            # EXCEPT for diode circuits (v2.0 Phase 3): only the
            # Bridge.11 path carries the auto-derived predicates, so
            # falling through would freeze the diode bits and
            # reintroduce the silent wrong answer. Re-raise instead.
            if _n_diodes > 0:
                raise
            pass

    if builder_initial_state is not None:
        raise unsupported(
            "dsed", "initial_conditions", message=
            "simulate(engine='dsed'): builder initial conditions "
            "(c0= / i0= / set_initial) are mapped onto the reduced "
            "state by the native adapter only, which rejected this "
            "circuit. Use engine='pwl' or 'trbdf2'.")

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
            n_exact_steps=native_res.get("n_exact_steps", 0),
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
    raise unsupported(
        "dsed", "extraction", cls=_UnsupportedNotImplemented, message=
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
