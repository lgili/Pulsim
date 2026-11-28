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

import bisect as _bisect
import re

import numpy as _np
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, cast

from ._pulsim import (  # type: ignore[import-not-found]
    ConvergenceError,
    NativeMultiMaskPwm,
    NativePwm2Switch,
    SwitchSchedule,
    UnsupportedError,
)
from ._pulsim import run_option_defaults as _run_option_defaults  # type: ignore[import-not-found]
from ._refusal import _UnsupportedNotImplemented, unsupported
from ._plan import emit as _emit
from .errors import NetlistError, OptionsError
from .errors import options_error as _options_error


# =============================================================================
# Type aliases (public for downstream type-hints)
# =============================================================================

# Auto = exact stepper when the mode has one, else the stiffness
# detector's DOPRI5/BDF2 choice; rk45/bdf2 = force one integrator for
# ALL modes (rk45 still takes the exact stepper where it exists).
IntegratorChoice = Literal["auto", "rk45", "bdf2", "fixed"]

_DEFAULTS = _run_option_defaults()


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
    # Defaults: sim::Defaults (the one table, add-kernel-simulation-object).
    rtol: float = _DEFAULTS["rtol_exact"]
    atol: float = _DEFAULTS["atol_exact"]
    dt_max: float = _DEFAULTS["dt_max_exact"]
    dt_init: float = _DEFAULTS["dt_init_exact"]
    integrator: IntegratorChoice = "auto"
    stiffness_threshold: float = _DEFAULTS["stiffness_threshold"]
    h_bdf2: float = _DEFAULTS["h_bdf2"]
    store_every: int = 1
    t_start: float = 0.0
    switch_fn: Optional[Callable[[float], Any]] = None
    initial_state: Optional[Any] = None
    progress: Any = False


# =============================================================================
# Validation
# =============================================================================


# 'exact' is the canonical name of the event-driven exact engine, 'dsed' its
# legacy alias (add-exact-propagator-strategy D11; simulate() maps one to the other).
_VALID_ENGINES = ("auto", "pwl", "exact", "dsed", "trbdf2")
_VALID_INTEGRATORS = ("auto", "rk45", "bdf2", "fixed")

# (The dsed-only kwargs that get the leak-warning if user passes
# them while engine='pwl' are enumerated inline in
# _validate_engine_kwargs below — keeping the list local to its
# usage site so it doesn't drift out of sync.)


class _InconsistentInitialState(OptionsError):
    """A full-MNA initial_state that no reduced state reproduces (an
    OptionsError, so still a ValueError)."""
    option = "initial_state"


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
        raise _options_error(
            "engine",
            f"simulate(): unknown engine={engine!r}; "
            f"expected one of {_VALID_ENGINES}"
        )

    # --- engine='pwl' branch ---
    if engine == "pwl":
        if dt is None or dt <= 0:
            raise _options_error(
                "dt",
                "simulate(engine='pwl'): a positive `dt` is required. "
                "Either pass `dt=...` for fixed-step trapezoidal, or "
                "switch to `engine='dsed'` for variable-step (no dt "
                "needed)."
            )
        # Options this engine does not honour (rtol, atol, dt_init,
        # integrator, ...) are refused by simulate() from the option x
        # engine table once the run's engine is known; they used to be a
        # warning here, and then ignored.
        return

    # --- engine='auto' / 'trbdf2' branch. 'auto' ROUTES (to the
    #     variable-step engine when the circuit qualifies, to the
    #     fixed-step one otherwise); 'trbdf2' names the
    #     variable-step engine explicitly and refuses instead of
    #     routing. dt is optional in both and acts as the step
    #     ceiling h_max. ---
    if engine in ("auto", "trbdf2"):
        if dt is not None and dt <= 0:
            raise _options_error(
                "dt",
                f"simulate(engine={engine!r}): dt (used as the step "
                f"ceiling h_max here) must be positive, got {dt!r}")
        if rtol is not None and rtol <= 0:
            raise _options_error(
                "rtol",
                f"simulate(engine={engine!r}): rtol must be "
                f"positive, "
                f"got {rtol!r}")
        if atol is not None and atol <= 0:
            raise _options_error(
                "atol",
                f"simulate(engine={engine!r}): atol must be "
                f"positive, "
                f"got {atol!r}")
        if dt_init is not None and dt_init <= 0:
            raise _options_error(
                "dt_init",
                f"simulate(engine={engine!r}): dt_init must be "
                f"positive, "
                f"got {dt_init!r}")
        return

    # --- engine='dsed' branch ---
    if rtol is not None and rtol <= 0:
        raise _options_error(
            "rtol",
            f"simulate(engine='dsed'): rtol must be positive, got {rtol!r}"
        )
    if atol is not None and atol <= 0:
        raise _options_error(
            "atol",
            f"simulate(engine='dsed'): atol must be positive, got {atol!r}"
        )
    if dt_init is not None and dt_init <= 0:
        raise _options_error(
            "dt_init",
            f"simulate(engine='dsed'): dt_init must be positive, "
            f"got {dt_init!r}"
        )
    if integrator is not None and integrator not in _VALID_INTEGRATORS:
        raise _options_error(
            "integrator",
            f"simulate(engine='dsed'): unknown integrator={integrator!r}; "
            f"expected one of {_VALID_INTEGRATORS}"
        )
    if stiffness_threshold is not None and stiffness_threshold <= 0:
        raise _options_error(
            "stiffness_threshold",
            f"simulate(engine='dsed'): stiffness_threshold must be "
            f"positive, got {stiffness_threshold!r}"
        )
    if h_bdf2 is not None and h_bdf2 <= 0:
        raise _options_error(
            "h_bdf2",
            f"simulate(engine='dsed'): h_bdf2 must be positive, "
            f"got {h_bdf2!r}"
        )
    if dt is not None and dt <= 0:
        raise _options_error(
            "dt",
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
      adapts down naturally for higher f_sw). A gate that names no edge
      (no ``switch_fn``, or a plain callable without ``next_edge_after``)
      is polled every ``dt_max / 10`` = 1 µs, so such a run steps at 1 µs.
    * **h_bdf2**: default 1 µs (reasonable for stiff segments at
      typical power-electronics scales).
    * **dt_init**: default 1 ns (PI controller grows from here).

    Validation must be called FIRST — this function assumes its
    inputs are sane.
    """
    dt_max = dt if (dt is not None and dt > 0) else _DEFAULTS["dt_max_exact"]

    # integrator was already validated to be in _VALID_INTEGRATORS;
    # cast for the type checker.
    resolved_integrator = cast(
        IntegratorChoice,
        integrator if integrator is not None else "auto",
    )

    return DSEDOptions(
        rtol=rtol if rtol is not None else _DEFAULTS["rtol_exact"],
        atol=atol if atol is not None else _DEFAULTS["atol_exact"],
        dt_max=dt_max,
        dt_init=dt_init if dt_init is not None else _DEFAULTS["dt_init_exact"],
        integrator=resolved_integrator,
        stiffness_threshold=(
            stiffness_threshold if stiffness_threshold is not None
            else _DEFAULTS["stiffness_threshold"]
        ),
        h_bdf2=h_bdf2 if h_bdf2 is not None else _DEFAULTS["h_bdf2"],
        store_every=store_every,
        t_start=t_start,
        switch_fn=switch_fn,
        initial_state=initial_state,
        progress=progress,
    )


# =============================================================================
# Bridge — CircuitBuilder → PED scheduler
# =============================================================================


# Surfaced only when the C++ extractor throws: the underlying reason, the
# engines that take the circuit, and the options this call resolved. (It
# used to list long-fixed causes -- floating capacitors, nonlinear devices --
# and point to a design note that ships with no wheel.)
_BRIDGE_FALLBACK_MESSAGE = (
    "pulsim.simulate(engine='dsed', builder=...): the C++ extractor "
    "rejected this topology: {underlying}\n"
    "Run it with engine='pwl' (with dt) or engine='trbdf2'; "
    "pulsim.exact_route(...) reports which segments of a run the exact "
    "engine can take.\n"
    "\n"
    "Resolved DSED options for this call:\n"
    "  rtol = {rtol}, atol = {atol}\n"
    "  dt_max = {dt_max}, integrator = {integrator!r}\n"
    "  stiffness_threshold = {stiffness_threshold}"
)


def _exact_result(
    times,
    states,
    *,
    states_full=None,
    switch_states=None,
    right_limit_index=None,
    right_limit_states=None,
    right_limit_switch_states=None,
    reinit_records=(),
    n_accept: int = 0,
    n_reject: int = 0,
    n_events: int = 0,
    n_rk45_steps: int = 0,
    n_bdf2_steps: int = 0,
    n_exact_steps: int = 0,
    cpu_time_seconds: float = 0.0,
):
    """The exact engine's run as a :class:`pulsim.SimulationResult`
    (add-probes-and-recorders D3: one result type on every engine).

    ``times``, ``states``, ``switch_states`` and ``reinit_records`` are
    the kernel result's own fields. A ``reinit_records`` entry is written
    for each eigen-direction of a projection whose residual exceeds 1e-9 of
    the circuit's scale along it (each state counted at least at the
    circuit's vscale for a capacitor or iscale for an inductor): a located
    zero of an ideal diode's current writes none, and neither does a real
    cut-set jump below that bound (1e-7 A cut in a circuit carrying 1e4 A is
    projected silently, dW ~ 5e-18 J). ``states`` is the reconstructed
    full-MNA trajectory (node voltages, then source and inductor branch
    currents: the pwl engine's row layout) when the kernel returned one,
    else the reduced integrator state. The exact-engine extras are
    attributes of the instance:

    * ``states_reduced`` / ``has_full_states`` -- the reduced integrator
      state, and whether ``states`` is the full MNA vector;
    * ``n_accept`` / ``n_reject`` / ``n_events`` / ``n_rk45_steps`` /
      ``n_bdf2_steps`` / ``n_exact_steps`` / ``cpu_time_seconds`` --
      read-only properties over ``res.stats`` (exact-engine results
      only);
    * ``switch_states`` -- the per-sample mode mask, ``(n, n_switches)``
      bool, the same record the pwl / trbdf2 engines carry. Each full
      MNA sample is reconstructed THROUGH that sample's mask, so a
      switched branch's current is exactly ``g(bit) · v`` there.
      ``None`` when the kernel returned only reduced states (no node
      voltages to apply it to) or no typed mask (the Python-adapter
      path, which never carries PWL diodes).
    * ``right_limit_index`` / ``right_limit_states`` /
      ``right_limit_switch_states`` -- at every sample where an event
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
    from ._pulsim import GridKind, SimulationResult  # type: ignore[import-not-found]
    t_arr = _np.asarray(times, dtype=float).reshape(-1)
    arr = _np.asarray(states, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(len(t_arr), -1) if len(t_arr) else arr.reshape(0, 0)
    arr.setflags(write=False)
    full = None
    if states_full is not None:
        full = _np.asarray(states_full, dtype=float)
        if full.size == 0 or full.ndim != 2 or full.shape[0] != len(t_arr):
            full = None
        else:
            full.setflags(write=False)
    sw = None
    if switch_states is not None and full is not None:
        sw = _np.asarray(switch_states, dtype=bool)
        if sw.ndim != 2 or sw.shape[0] != len(t_arr):
            sw = None
    res = SimulationResult()
    res._adapter_fill(t_arr, full if full is not None else arr, sw,
                      list(reinit_records))
    res._set_run_info("dsed", GridKind.Accepted, {
        "n_steps": int(n_accept), "n_accept": int(n_accept),
        "n_reject": int(n_reject), "n_mode_events": int(n_events),
        "n_rk45_steps": int(n_rk45_steps), "n_bdf2_steps": int(n_bdf2_steps),
        "n_exact_steps": int(n_exact_steps),
        "cpu_time_seconds": float(cpu_time_seconds)})
    res.states_reduced = arr
    res.has_full_states = full is not None
    # add-exact-propagator-strategy D10: per visited mask the
    # propagator tier (1 eig, 2 Pade, 3 none), kappa2 and the reason
    # for any tier but 1, plus n_eig / n_pade / n_phi_cache_hits /
    # n_predicate_evals. Event instants and predicate types as
    # arrays. Set by the native builder path; None elsewhere.
    res.exact = None
    res.initial_condition_adjustment = None
    res.event_times = None
    res.event_types = None
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
                       and rl_idx.max() < len(t_arr)
                       and _np.all(_np.diff(rl_idx) > 0))))
        if ok:
            for a in (rl_idx, rl_states, rl_sw):
                a.setflags(write=False)
        else:
            rl_idx = rl_states = rl_sw = None
    res.right_limit_index = rl_idx
    res.right_limit_states = rl_states
    res.right_limit_switch_states = rl_sw
    return res


def _result_from_native(native_res, t_start: float):
    """The result of a Bridge.11 run (``run_*_native_builder``). ``t_start``:
    the engine ran on its own clock from 0; every time it reports is
    relabelled here (the loss events in the binding, ``t_offset``)."""
    import numpy as np
    if t_start:
        for rd in native_res.get("reinit_records", ()):
            rd["t"] += t_start
        for key in ("mode_shadow", "mode_selection", "mode_guard"):
            g = native_res.get(key)
            if g is not None:
                for s in (g, g.get("lcp", {})):
                    for e in s.get("disagreements", ()):
                        e["t"] += t_start
                if "repair_times" in g:
                    g["repair_times"] = [t + t_start for t in g["repair_times"]]
    # One array conversion, not one per sample.
    times = (np.asarray(native_res["times"], dtype=float)
             + float(t_start))
    res = _exact_result(
        times=times,
        states=native_res["states"],
        states_full=native_res.get("states_full"),
        switch_states=native_res.get("switch_states"),
        right_limit_index=native_res.get("right_limit_index"),
        right_limit_states=native_res.get("right_limit_states"),
        right_limit_switch_states=native_res.get(
            "right_limit_switch_states"),
        reinit_records=native_res.get("reinit_records", []),
        n_accept=native_res.get("n_accept", 0),
        n_reject=native_res.get("n_reject", 0),
        n_events=native_res.get("n_events", 0),
        n_rk45_steps=native_res.get("n_rk45_steps", 0),
        n_bdf2_steps=native_res.get("n_bdf2_steps", 0),
        n_exact_steps=native_res.get("n_exact_steps", 0),
        cpu_time_seconds=native_res.get("cpu_time_seconds", 0.0),
    )
    res.exact = native_res.get("exact")
    if res.exact is not None:
        # add-exact-propagator-strategy D11: the route record (report only)
        from ._exact_route import segments_from_exact
        res.exact["segments"] = segments_from_exact(
            res.exact, native_res.get("switch_states"))
        # 13.3, report only: would 'auto' route this run exact? (every
        # segment exact; the default route does not read it)
        res.exact["auto_would_route"] = bool(res.exact["segments"]) and all(
            seg["exact"] for seg in res.exact["segments"])
    et = native_res.get("event_times")
    res.event_times = (None if et is None
                       else np.asarray(et) + float(t_start))
    res.event_types = native_res.get("event_types")
    res.diagnostics = {}
    for key in ("mode_shadow", "mode_selection", "mode_guard"):
        if key in native_res:
            res.diagnostics[key] = native_res[key]
    if "projection_records" in native_res:
        # every projection of the parasitic cluster at an event, rule on or off (never
        # silent; F-SB-3 / F-R-3): the count, per stored column the largest amplitude it
        # moved, and the records (instant, largest amplitude, its column, the cluster's
        # slowest time constant; capped)
        ncol = res.states.shape[1] if getattr(res, "has_full_states", False) else 0
        pm = np.asarray(native_res.get("projection_max_row", []), dtype=float)
        proj = dict(
            n_projections=int(native_res.get("n_projections", 0)),
            projection_max_by_column=(pm[-ncol:] if pm.size >= ncol > 0 else np.zeros(ncol)),
            projections=[
                {"t": t + t_start, "amplitude": a, "tau": tau,
                 # a row of the recovered vector that is not a stored column: None
                 "column": (int(r) - (pm.size - ncol)) if 0 < ncol and int(r) >= pm.size - ncol else None}
                for t, a, r, tau in native_res.get("projection_records", [])])
        res.diagnostics["event_projections"] = proj
    else:
        proj = {}
    if "n_output_capped" in native_res:   # the output-grid rule (output_rtol)
        og = {k: int(native_res[k]) for k in ("n_output_capped", "n_output_unbounded")}
        # the rule's samples: read from a step's start, not steps (n_exact_steps)
        og["n_output_samples"] = int(native_res.get("n_output_samples", 0))
        # rows certified "constant up to rounding" at their rounding floor (never silent):
        # the chords that needed it, the largest floor, and per stored column its largest
        # floor (the recovered vector is the tail of the bounded rows; 0: never floored)
        og["n_output_floored"] = int(native_res.get("n_output_floored", 0))
        # of them, the calls certified only by the basis-inconsistency floor: a fast mode's
        # content that did not decay as its own closed form says (F-X-8; counted, reported)
        og["n_output_inconsistent"] = int(native_res.get("n_output_inconsistent", 0))
        fr = np.asarray(native_res.get("output_floor_row", []), dtype=float)
        og["output_floor_max"] = float(fr.max()) if fr.size else 0.0
        ncol = res.states.shape[1] if getattr(res, "has_full_states", False) else 0
        og["output_floor_by_column"] = (fr[-ncol:] if fr.size >= ncol > 0
                                        else np.zeros(ncol))
        og.update(proj)
        tb = native_res["output_budget_t"]
        og["output_budget_t"] = None if tb != tb else tb + t_start   # NaN: not spent
        res.diagnostics["output_grid"] = og
        if og["output_budget_t"] is not None:
            _emit("output_budget",
                f"simulate(engine='exact', output_rtol=...): the run's output sample budget "
                f"ran out at t = {og['output_budget_t']:.6g} s; from there the samples are the "
                "default grid and linear interpolation between them is NOT within output_rtol "
                "(diagnostics['output_grid']['output_budget_t']). Raise output_rtol, or pass dt.",
                RuntimeWarning, stacklevel=4)
        if og["n_output_inconsistent"]:
            _emit("output_inconsistent",
                f"simulate(engine='exact', output_rtol=...): {og['n_output_inconsistent']} output "
                "rule call(s) met a fast mode whose content did not decay as its own closed form "
                "says (the eigenbasis's or the propagator's error, not a transient); the step was "
                "bounded without that part, and the bound with it is those rows' floor there "
                "(diagnostics['output_grid']['output_floor_by_column']): linear interpolation is "
                "within max(output_rtol, that floor).",
                RuntimeWarning, stacklevel=4)
        if og["n_output_capped"] or og["n_output_unbounded"]:
            _emit("output_cap",
                f"simulate(engine='exact', output_rtol=...): the output grid's sample cap "
                f"bit in {og['n_output_capped']} interval(s) and {og['n_output_unbounded']} "
                "step(s) had no diagonalising basis to bound: linear interpolation between "
                "those samples is NOT within output_rtol (diagnostics['output_grid']), and "
                "integrals taken from the stored record there (res.avg / res.integral / "
                "with_event_limits) are chords over those gaps; in-loop recorders integrate "
                "the exact trajectory. Pass dt for a fixed output step.",
                RuntimeWarning, stacklevel=4)
    # dependent states: the impulsive re-initialisations at mode entry
    # are the result's own `reinit_records` (filled by _exact_result)
    return res


# The snapshot's controller block of the exact engine (step::ControllerState
# kind Exact): the reduced state, the schedule position and the step memory.
_EXACT_SNAPSHOT_KIND = 3


def _mark_no_snapshot(res, integrator: str, t_start, bridge: str = "") -> None:
    """A run that keeps no snapshot says why: its final_snapshot stays invalid
    with the cause and the workaround, which resume_from refuses by name."""
    from . import _pulsim as _k
    if bridge:
        why = (f"engine='dsed' ran this circuit on the {bridge} (the native "
               "adapter rejected it), which keeps no snapshot")
    elif integrator == "bdf2":
        why = ("engine='dsed' ran this circuit on integrator='bdf2' (fixed-step "
               "BDF2), which keeps no snapshot; the event-driven and the 'auto' "
               "integrators write one. Run the first segment without "
               "integrator='bdf2'")
    elif t_start:
        why = (f"engine='dsed' ran a fresh run with t_start={t_start!r} on its "
               "own clock from 0, so it writes no snapshot (its time would be "
               "wrong). Run the first segment without t_start")
    else:
        return
    s = _k.SolverSnapshot()
    s.invalid_reason = why
    res._set_final_snapshot(s)


_AUTO_SNAPSHOT_TAG = 101.0   # dsed/scheduler_auto.hpp AutoResumeState::kTag


def _auto_snapshot(snap) -> bool:
    """Whether an exact-engine snapshot was written by integrator='auto'."""
    cs = snap.controller_state
    return (cs is not None and cs[0] == _EXACT_SNAPSHOT_KIND and len(cs[1]) > 0
            and float(cs[1][0]) == _AUTO_SNAPSHOT_TAG)


def _check_resume_snapshot(snap, integrator, t_start) -> None:
    """Refuse by name a snapshot the exact engine cannot continue exactly
    (session 24): another integrator, a shifted clock, or another engine's
    device histories it has no state for. A recorder's state (field 18) is
    continued by the in-loop feed (session 26), or refused by
    simulate(recorders=...) when this run cannot feed it."""
    if integrator == "bdf2":
        raise unsupported(
            "dsed", "resume_from", message=
            "simulate(engine='dsed', integrator='bdf2', resume_from=...): the "
            "snapshot is written and read by the event-driven integrator only. "
            "Drop integrator='bdf2'.")
    if t_start and float(t_start) != float(snap.t):
        raise unsupported(
            "dsed", "resume_from", message=
            f"simulate(engine='dsed', resume_from=..., t_start={t_start!r}): the "
            f"exact engine resumes on its own clock at the snapshot's instant "
            f"({snap.t!r} s); a different t_start would relabel time. Drop t_start.")
    cs = snap.controller_state
    if cs is not None and cs[0] == _EXACT_SNAPSHOT_KIND:
        return
    if not len(snap.stamped_mask):
        raise unsupported(
            "dsed", "resume_from", message=
            "simulate(engine='dsed', resume_from=...): the snapshot does not say "
            "which mask its state was solved under (no stamped_mask: a hand-built "
            "or pre-v3 snapshot), so its state cannot be read as one mode's.")
    layout = list(getattr(snap, "dev_layout", []) or [])
    if any(int(n) for n in layout[1:]):
        raise unsupported(
            "dsed", "resume_from", message=
            "simulate(engine='dsed', resume_from=...): the snapshot carries "
            "stateful-device histories (saturable inductor, Coss, IGBT tail, ...) "
            "the exact engine has no state for. Resume on the engine that wrote it.")


def _resume_state(snap, native_adapter, mask, builder, n_state):
    """The reduced state the resumed run starts from: its own (C++ reads the
    snapshot's payload; zeros of the right size here), or another engine's
    full state reduced under its stamped mask — refused by name when that is
    not one state of the mode (a full state off the mode's constraints)."""
    import numpy as np
    cs = snap.controller_state
    if cs is not None and cs[0] == _EXACT_SNAPSHOT_KIND:
        return np.zeros(n_state, dtype=float)
    x_full = np.asarray(snap.x, dtype=float)
    n_mna = builder.pool.state_size(builder.graph)
    if x_full.shape != (n_mna,):
        raise unsupported(
            "dsed", "resume_from", message=
            f"simulate(engine='dsed', resume_from=...): the snapshot's state has "
            f"{x_full.shape[0]} entries, this circuit's full state {n_mna}: it "
            "belongs to a different circuit.")
    native_adapter.set_mask(mask)
    x0, adj = native_adapter.reduced_state_from_full(x_full)
    nn = builder.graph.num_nodes
    scale = max(1.0, float(np.abs(x_full[:nn]).max(initial=0.0)))
    if adj > 1e-9 * scale:
        raise unsupported(
            "dsed", "resume_from", message=
            f"simulate(engine='dsed', resume_from=...): the snapshot's state is "
            f"not one state of its mode on this engine (a capacitor voltage moves "
            f"by {adj:.3g} V when read as the mode's reduced state). Resume on "
            "the engine that wrote it.")
    return np.asarray(x0, dtype=float)


def _has_edge_oracle(sf) -> bool:
    """Whether the DSED scheduler can ask ``sf`` for its next gate edge."""
    if isinstance(sf, (NativePwm2Switch, NativeMultiMaskPwm)):
        return True
    if isinstance(sf, SwitchSchedule):
        return bool(sf.has_edges)
    return hasattr(sf, "next_edge_after")


# The circuit's own time-varying sources: the native adapter evaluates them at
# the engine's clock (C++), which no Python wrapper can move.
_TIMED_SOURCE_KINDS = frozenset((
    "sine_voltage_source", "sine_current_source", "pulse_voltage_source",
    "pulse_current_source", "pwm_voltage_source", "pwl_current_source"))


class _OnAbsoluteClock:
    """A gate / ``b_extra_fn`` read at the absolute time ``tau + t0`` while
    the exact engine's clock ``tau`` runs from 0."""

    __slots__ = ("fn", "t0")

    def __init__(self, fn, t0: float):
        self.fn, self.t0 = fn, t0

    def __call__(self, tau):
        return self.fn(tau + self.t0)


class _OnAbsoluteClockEdges:
    """A gate with an edge oracle on the absolute clock. Each edge ``e`` is
    handed to the engine as ``te = e - t0``; a read at ``tau`` keeps its order
    with every handed-out edge (``te`` reads ``e``, a later ``tau`` reads after
    ``e``, an earlier one before it): ``tau + t0`` alone rounds onto or across
    ``e`` once ulp(t0) exceeds the engine's own margin past an edge (a gate
    switching strictly after its edges ran inverted at t0 = 20 s)."""

    __slots__ = ("fn", "t0", "_te", "_e")

    def __init__(self, fn, t0: float):
        self.fn, self.t0, self._te, self._e = fn, t0, [], []

    def _abs(self, tau: float) -> float:
        a = tau + self.t0
        k = _bisect.bisect_right(self._te, tau)
        if k:                                   # the last edge at or before tau
            if self._te[k - 1] == tau:
                return self._e[k - 1]
            a = max(a, _np.nextafter(self._e[k - 1], _np.inf))
        if k < len(self._te):                   # the first edge after tau
            a = min(a, _np.nextafter(self._e[k], -_np.inf))
        return a

    def __call__(self, tau):
        return self.fn(self._abs(tau))

    def next_edge_after(self, tau):
        e = float(self.fn.next_edge_after(self._abs(tau)))
        if not (e < float("inf")):
            return e
        te = max(e - self.t0, _np.nextafter(tau, _np.inf))   # still after tau
        k = _bisect.bisect_left(self._te, te)
        if k == len(self._te) or self._te[k] != te:
            if len(self._te) >= 4096 and k:      # keep the recent edges, in order
                del self._te[:k // 2], self._e[:k // 2]
                k -= k // 2
            self._te.insert(k, te)
            self._e.insert(k, e)
        return te


def relabel_failure(e: BaseException, t0: float) -> None:
    """A typed give-up of a fresh ``t_start`` run (``ConvergenceError``,
    ``DiodeModeInconsistentError``): its ``t_failed`` is on the engine's
    clock; relabel it as the result's times are (the message text is the
    kernel's and keeps the engine's clock)."""
    t = getattr(e, "t_failed", None)
    if isinstance(t, float):
        e.t_failed = t + t0


def on_absolute_clock(builder, t0: float, switch_fn, b_extra_fn):
    """``simulate(t_start=t0)`` on the exact engine, a fresh run: the engine
    runs on its own clock from 0 and its outputs are relabelled by ``t0``
    (``_result_from_native``, the loss events in the binding), so every input
    it reads a time from is shifted here, the one place (``_run_dsed``): the
    gate and its edge oracle, ``b_extra_fn``. The circuit's own time-varying
    sources are evaluated by the native adapter and cannot be shifted from
    Python: refused by name. Control ticks with t_start are refused by
    ``_compose_for``."""
    for c in builder.components():
        if c["kind"] in _TIMED_SOURCE_KINDS:
            raise unsupported(
                "dsed", "t_start", device=c["name"], message=
                f"simulate(engine='dsed', t_start={t0!r}): the exact engine runs a "
                f"fresh run on its own clock from 0, and the time-varying source "
                f"{c['name']!r} ({c['kind']}) is evaluated on that clock (it would "
                f"run {t0!r} s early). Start at t_start=0 (resume_from a snapshot "
                "continues on the absolute clock), or use engine='pwl' / 'trbdf2'.")
    if switch_fn is not None:
        from .periodic_steady_state import _closed_loop_source
        src = _closed_loop_source(switch_fn)
        if src:   # its controller would run on the engine's clock (and the wrapper
            # would hide it from the sensitivity path's own refusal)
            raise unsupported(
                "dsed", "t_start", message=
                f"simulate(engine='dsed', t_start={t0!r}): switch_fn comes from "
                f"{src}: the exact engine runs a fresh run on its own clock from 0, "
                "and a gate that follows a controller is not shifted onto it. Start "
                "at t_start=0, or use engine='pwl' / 'trbdf2'.")
        if _has_edge_oracle(switch_fn):
            w = _OnAbsoluteClockEdges(switch_fn, t0)
            # t0 ON an edge: the start reads the mask after it, as the
            # uninterrupted run reads it after its edge at t0 (a gate that
            # switches strictly after its edges kept the old mask a period)
            if float(switch_fn.next_edge_after(_np.nextafter(t0, -_np.inf))) == t0:
                w._te, w._e = [0.0], [_np.nextafter(t0, _np.inf)]
            switch_fn = w
        else:
            switch_fn = _OnAbsoluteClock(switch_fn, t0)
    if b_extra_fn is not None:
        b_extra_fn = _OnAbsoluteClock(b_extra_fn, t0)
    return switch_fn, b_extra_fn


# add-complementarity-mode-selection D5: the event step of the shadow LCP.
# h_e = H_E_REL * T_ref (measured band 1e-9..1e-6, probes/h_e_band.out), T_ref the shortest declared period (sine / PWM
# source) or the run length (see DECISIONS.md for the measured band).
H_E_REL = 1e-7
# Slice 2, task 4: the default-mode guard (judge each settled mode, re-decide a
# violating one with the LCP). Module switch for A/B measurements only. Off, the
# settle still decides but is judged: a mode the guard would repair raises
# UnsupportedError('dsed', 'consistent_mode') instead (DECISIONS #26).
MODE_GUARD = True
# Test hook: the settle's round cap (the engine's default, 64). A test lowers it
# to reach the give-up deterministically: the cap is otherwise met only by the
# rounding of a diode pair fighting at rest (macOS yes, Linux / Windows no).
SETTLE_ROUND_CAP = 64
# Relative threshold of the kernel's diode-law check (0 turns it off: tests that
# isolate another refusal).
DIODE_LAW_REL = 1e-3
# The output rule's sample budget per run (cut steps; output_rtol): past it the rule stops
# cutting, with diagnostics['output_grid']['output_budget_t'] and one RuntimeWarning.
OUTPUT_BUDGET = 2_000_000


def _shadow_h_e(builder: Any, switch_fn: Any, t_window: float) -> float:
    import math
    from . import _pulsim as _kern
    periods = [float(t_window),
               float(_kern._min_source_period(builder.graph, builder.pool))]
    nxt = getattr(switch_fn, "next_edge_after", None)
    if nxt is not None:                 # a periodic gate: two edges per period
        e1 = float(nxt(0.0))
        e3 = float(nxt(float(nxt(e1))))
        if math.isfinite(e3) and e3 > e1:
            periods.append(e3 - e1)
    return H_E_REL * min(periods)


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
    mode_selection: Optional[str] = None,
    allow_implicit_segments: bool = True,
    ticks: Optional[Any] = None,
    output_rtol: Optional[float] = None,
    resume_from: Optional[Any] = None,
    tick_origins: Optional[Any] = None,
    tick_next_k: Optional[Any] = None,
    undriven: Optional[str] = None,
    mode_auto: Optional[dict] = None,
):
    """Run a PED simulation from a :class:`pulsim.CircuitBuilder`.

    ``resume_from`` (a ``SolverSnapshot``): continue at its committed
    instant on the run's own clock (``t_end`` absolute). A snapshot of this
    engine (controller kind 3) carries the reduced state, the mode and the
    schedule position; another engine's is reduced from its full state under
    its stamped mask, refused by name where that is not one state of the
    mode, and its modes are decided at t as at a start.

    ``mode_auto``: the record of ``mode_selection='auto'`` as simulate()'s
    plan resolved it (`_resolve_mode_auto`, the one place); ``mode_selection``
    then arrives as 'lcp' or None (a caller that passes 'auto' itself gets
    the same rule applied here).

    ``undriven`` ('open' / 'closed', from ``simulate(undriven=...)``): the
    mask of the controlled switches when ``switch_fn`` is None, without the
    all-closed warning (the plan holds the one diagnostic).

    ``ticks`` (``[(fn, period, offset), ...]``): control ticks scheduled
    by the event-driven integrator's event queue, ``fn(t, x_full)`` at
    every ``offset + k*period`` before ``t_end`` (add-unified-event-queue
    group 3). They need the native adapter and the event-driven
    integrator; anything else refuses them by name.

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
    if mode_selection == "auto":   # not through simulate(): the plan's rule, here
        from . import _resolve_mode_auto
        mode_selection, mode_auto = _resolve_mode_auto(builder, "dsed", None)

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
    # add-exact-propagator-strategy 10 (D8): integrator='fixed' is the
    # event-driven integrator on the grid k*dt, a full cell x <- Phi x + Gamma
    fixed_step = integrator == "fixed"
    if fixed_step:
        if dt is None:
            raise _options_error(
                "integrator",
                "simulate(engine='exact', integrator='fixed'): the fixed-step form "
                "steps the grid k*dt; pass dt.")
        integrator = "rk45"
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
    # add-in-kernel-event-losses 3.2: a loss port runs on the event-driven
    # integrator only (its step hooks feed solver::LossTracker); 'auto'
    # becomes 'rk45' as for diodes, an explicit 'bdf2' is refused by name.
    _loss_names = list(getattr(builder, "_device_loss_names", lambda: [])())
    if _loss_names:
        if integrator == "bdf2":
            raise unsupported(
                "dsed", "device_losses", device=_loss_names[0], message=
                f"simulate(engine='dsed', integrator='bdf2'): the loss port of "
                f"{_loss_names[0]!r} (set_device_losses) is read on the "
                "event-driven integrator only. Drop integrator='bdf2'.")
        # a fresh t_start run runs on the engine's clock from 0 (simulate()
        # shifts the drive onto it): its loss events are relabelled by t_start
        integrator = "rk45"
    # update-number-changing-release: the output-grid rule cuts the
    # event-driven integrator's exact steps; 'auto' becomes 'rk45' as for
    # diodes (the exact segments are the same propagator), 'bdf2' refuses.
    if output_rtol:
        if integrator == "bdf2":
            raise unsupported(
                "dsed", "output_rtol", message=
                "simulate(engine='dsed', integrator='bdf2', output_rtol=...): the "
                "output-grid rule cuts the event-driven integrator's exact steps. "
                "Drop integrator='bdf2', or drop output_rtol.")
        integrator = "rk45"
    if resume_from is not None:
        if initial_state is not None:
            raise unsupported(
                "dsed", "resume_from", message=
                "simulate(engine='dsed', resume_from=..., initial_state=...): the "
                "snapshot is the state; pass one of them.")
        _check_resume_snapshot(resume_from, integrator, t_start)
        if _auto_snapshot(resume_from):
            # task 20.2: written by PEDSimulatorAuto, continued by it only
            if integrator not in (None, "auto") or ticks:
                raise unsupported(
                    "dsed", "resume_from", message=
                    "simulate(engine='dsed', resume_from=...): the snapshot was "
                    "written by integrator='auto' (the per-mode RK45/BDF2 "
                    "integrator), and this run would take the event-driven one "
                    f"(integrator={integrator!r}"
                    f"{' from output_rtol' if output_rtol else ''}"
                    f"{', control ticks' if ticks else ''}"
                    "), which cannot continue its step history. Resume with the "
                    "first segment's options.")
            integrator = "auto"
        elif integrator == "auto" and not ticks:
            # diodes and loss ports made it 'rk45' above, ticks do below:
            # anything else is 'auto' choosing PEDSimulatorAuto, which
            # cannot continue an event-driven snapshot
            raise unsupported(
                "dsed", "resume_from", message=
                "simulate(engine='dsed', integrator='auto', resume_from=...): on "
                "this circuit 'auto' chooses the per-mode RK45/BDF2 integrator, "
                "which cannot continue the event-driven integrator's snapshot. "
                "Pass integrator='rk45' or leave integrator unset (a resume then "
                "runs on the integrator that wrote the snapshot).")
        else:   # unset: the integrator that wrote the snapshot (the event-driven one)
            integrator = "rk45"
    # Device kinds: the kernel's capability table, before any adapter
    # is built (the Python adapter below would extract the LTI system
    # with every Newton device silently ABSENT). The native adapter
    # repeats the check for C++ callers.
    _kern._require_supported(builder.graph, builder.pool, "dsed",
                             loss_port=bool(_loss_names))
    ticks = list(ticks or [])
    if ticks:
        if integrator not in (None, "auto", "rk45"):
            raise unsupported(
                "dsed", "step_observer", message=
                f"simulate(engine='dsed', integrator={integrator!r}): control "
                "ticks run on the event-driven integrator only. Drop integrator.")
        monitors_only = all(getattr(f, "_pulsim_monitor", False) for f, _p, _o in ticks)
        if monitors_only and _n_diodes == 0 and integrator != "rk45":
            # routing to rk45 would change the run the Monitor promises not to touch
            raise unsupported(
                "dsed", "step_observer", message=
                "simulate(engine='dsed') with a Monitor: monitors run on the "
                "event-driven integrator, and this circuit would otherwise run "
                "on the 'auto' one, so the Monitor would change the run. Pass "
                "integrator='rk45' (then the run is the same with or without it).")
        integrator = "rk45"

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
        from . import SwitchStateMask  # noqa: F401  # type: ignore[attr-defined]  (an install check)
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
        from . import _undriven
        sf = (_undriven("dsed", builder, stacklevel=5) if undriven is None
              else _undriven("dsed", builder, 5, undriven))

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
        _emit("unlocated_edges",
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
            stacklevel=4,  # (run_dsed_from_builder <- _run_dsed <- simulate)
        )

    adapter = CircuitBuilderAdapter(
        builder=builder, cache=cache, switch_fn=sf,
        b_extra_fn=b_extra_fn)

    # Eagerly resolve the initial mask so any extractor failure
    # (floating cap, nonlinear device, etc.) surfaces NOW with the
    # bridge-fallback message rather than mid-simulation.
    initial_mask = sf(0.0)   # the engine's clock (simulate() shifts the gate)
    if resume_from is not None:
        initial_mask = SwitchStateMask(builder.graph.num_switches)
        for _i, _b in enumerate(resume_from.stamped_mask):
            initial_mask.set(_i, bool(_b))
    adapter.set_mask(initial_mask)
    try:
        for _k in range(builder.graph.num_switches + 1):
            try:
                _ = adapter.A_matrix()
                break
            except ValueError as e:
                # the start mask's diode bits are not a decision: an ideal diode ON
                # in a loop of sources there starts OFF; the scheduler decides it
                # (or refuses the loop by name) at t_start
                feature = getattr(e, "feature", None)
                if _k == builder.graph.num_switches:
                    raise
                if feature == "ideal_voltage_loop" and getattr(e, "kind", None) == "Diode":
                    initial_mask.set(builder.switch_index_of(e.device), False)
                elif feature == "source_capacitor_loop" and _k == 0:
                    # it names the source, not the diode: every diode starts OFF
                    from ._pulsim import StoredKind  # type: ignore[import-not-found]
                    g = builder.graph
                    for b in range(g.num_branches):
                        if builder.pool.kind_of(b) == StoredKind.Diode:
                            initial_mask.set(builder.switch_index_of(g.branch_name(b)), False)
                else:
                    raise
                adapter.set_mask(initial_mask)
    except NotImplementedError as e:
        # a reduction numerically singular on a node group: the floating-island
        # refusal of the native run (DECISIONS #27), named here at the start
        m = re.search(r"numerically singular at node (\S+):", str(e))
        if m:
            raise unsupported(
                "dsed", "floating_island", device=m.group(1), kind="Node",
                message=f"dsed: the mode's reduction is numerically singular on the "
                f"floating island at node {m.group(1)}: its tie to ground is too weak "
                f"for the cancellation on the island's pivot to stay under rounding "
                f"(add-complementarity-mode-selection DECISIONS #27). Tie the island to "
                f"ground with a smaller resistance, or use engine='pwl' or 'trbdf2'. "
                f"Reduction: {str(e).splitlines()[0]}") from e
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
    full_x0 = None
    ambiguous = False
    if opts.initial_state is not None:
        x0 = np.asarray(opts.initial_state, dtype=float)
        n_mna = builder.pool.state_size(builder.graph)
        if x0.shape == (n_mna,):
            # a full-MNA vector (a DC operating point): projected onto the
            # reduced state by the native adapter below, checked consistent.
            # When the two sizes coincide the reading is ambiguous: it is
            # accepted only where both readings give the same state.
            full_x0, x0 = x0, np.zeros(n_state, dtype=float)
            ambiguous = n_mna == n_state
        elif x0.shape != (n_state,):
            raise _options_error(
                "initial_state",
                f"simulate(engine='dsed'): initial_state has shape "
                f"{x0.shape}, expected ({n_state},)."
            )
    else:
        x0 = np.zeros(n_state, dtype=float)

    t_window = float(t_end) - float(opts.t_start)
    if resume_from is not None:   # the run's own clock: t_end is absolute
        t_window = float(t_end) - float(resume_from.t)
    if t_window <= 0:
        raise _options_error(
            "t_end",
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

    adapter_rejected = False
    if bridge11_ok:
        try:
            adapter_rejected = True
            native_adapter = NativeAdapterCls(
                builder, cache,
                b_extra_fn if b_extra_fn is not None else None,
                loss_port=bool(_loss_names))
            adapter_rejected = False
            # 13.2: engine='exact' refuses a segment with no exact step
            native_adapter.set_refuse_implicit(not allow_implicit_segments)
            ic_adjustment = None
            if full_x0 is not None:
                native_adapter.set_mask(initial_mask)
                x0, adj = native_adapter.reduced_state_from_full(full_x0)
                x0 = np.asarray(x0, dtype=float)
                nn = builder.graph.num_nodes
                scale = max(1.0, float(np.abs(full_x0[:nn]).max(initial=0.0)))
                if ambiguous and not np.allclose(x0, full_x0, rtol=0.0, atol=1e-12 * scale):
                    raise _InconsistentInitialState(
                        f"simulate(engine='dsed'): initial_state has {n_state} entries, "
                        "both the reduced state size and the full-MNA size of this "
                        "circuit, and the two readings differ; pass the reduced states "
                        "through the builder's c0= / i0= / set_initial instead.")
                if adj > 1e-9 * scale:
                    raise _InconsistentInitialState(
                        f"simulate(engine='dsed'): initial_state is a full-MNA vector "
                        f"whose capacitor voltages are not consistent with one reduced "
                        f"state of the starting mode (largest adjustment {adj:.3g} V); "
                        f"pass a consistent vector or the {n_state} reduced states.")
            if builder_initial_state is not None:
                native_adapter.set_mask(initial_mask)
                x0, ic_adjustment = native_adapter.reduced_initial_state(
                    builder)
            resume_kw = {}
            if resume_from is not None:
                x0 = _resume_state(resume_from, native_adapter, initial_mask,
                                   builder, n_state)
                resume_kw = {"resume": resume_from,
                             "tick_origins": list(tick_origins or []),
                             "tick_next_k": list(tick_next_k or [])}
            shadow_kw = {}
            if mode_selection in ("shadow", "lcp") and (
                    opts.integrator != "rk45" or _n_diodes == 0):
                raise unsupported(
                    "dsed", "mode_selection", message=
                    f"simulate(engine='dsed', mode_selection="
                    f"{mode_selection!r}): the complementarity selection "
                    f"is wired into the event-driven integrator "
                    f"only; integrator={opts.integrator!r} has no hook. "
                    "Drop mode_selection, or run a circuit with diodes.")
            if opts.integrator == "rk45" and _n_diodes > 0:
                h_e = _shadow_h_e(builder, sf, t_window)
                if mode_selection == "lcp":
                    shadow_kw["mode_lcp_h_e"] = h_e
                else:    # default and shadow: the settle, guarded (or judged, refusing)
                    shadow_kw["mode_guard_h_e"] = h_e
                    shadow_kw["mode_guard_refuse"] = not MODE_GUARD
                if mode_selection == "shadow":
                    shadow_kw["mode_shadow_h_e"] = h_e
            if opts.integrator == "rk45":
                # add-probes-and-recorders R5: the recorders in the loop
                from . import _recorders as _rec
                feed_slot = _rec.ExactFeedSlot(builder)
                _rec.attach_loop_feed(feed_slot, "dsed")
                native_res = run_ped_b11(
                    native_adapter, sf, x0,
                    float(t_end) if resume_from is not None else t_window,
                    **resume_kw,
                    rtol=opts.rtol, atol=opts.atol,
                    dt_init=opts.dt_init, dt_max=opts.dt_max,
                    store_every=opts.store_every, event_log=False,
                    settle_round_cap=SETTLE_ROUND_CAP, diode_law_rel=DIODE_LAW_REL,
                    loop_feed=feed_slot.feed,
                    t_offset=float(opts.t_start or 0.0) if resume_from is None else 0.0,
                    **({"fixed_step": True} if fixed_step else {}),
                    **shadow_kw,
                    **({"output_rtol": float(output_rtol), "output_budget": int(OUTPUT_BUDGET)}
                       if output_rtol else {}),
                    **({"tick_fns": [f for f, _p, _o in ticks],
                        "tick_periods": [float(p) for _f, p, _o in ticks],
                        "tick_offsets": [float(o) for _f, _p, o in ticks],
                        "tick_monitors": [bool(getattr(f, "_pulsim_monitor", False))
                                          for f, _p, _o in ticks]}
                       if ticks else {}))
                if "loop_feed_refused" in native_res:
                    _rec.exact_feed_refused(native_res["loop_feed_refused"])
                if "recorder_closed_form" in native_res:
                    _rec.exact_closed_form(native_res["recorder_closed_form"])
            elif opts.integrator == "bdf2":
                native_res = run_bdf2_b11(
                    native_adapter, sf, x0, t_window,
                    h_fixed=opts.h_bdf2,
                    store_every=opts.store_every, event_log=False)
            else:  # 'auto': the same in-loop feed (its hooks only observe)
                from . import _recorders as _rec
                feed_slot = _rec.ExactFeedSlot(builder)
                _rec.attach_loop_feed(feed_slot, "dsed")
                native_res = run_auto_b11(
                    native_adapter, sf, x0,
                    float(t_end) if resume_from is not None else t_window,
                    rtol=opts.rtol, atol=opts.atol,
                    dt_init=opts.dt_init, dt_max=opts.dt_max,
                    h_bdf2=opts.h_bdf2,
                    stiffness_threshold=opts.stiffness_threshold,
                    store_every=opts.store_every, event_log=False,
                    loop_feed=feed_slot.feed,
                    **({"resume": resume_from} if resume_from is not None else {}))
                if "recorder_closed_form" in native_res:
                    _rec.exact_closed_form(native_res["recorder_closed_form"])
            res = _result_from_native(
                native_res, 0.0 if resume_from is not None else opts.t_start)
            if "final_snapshot" in native_res and (
                    resume_from is not None or not opts.t_start):
                # a fresh run with t_start runs on the engine's own clock from
                # 0, not simulate()'s: its snapshot would carry the wrong time
                res._set_final_snapshot(native_res["final_snapshot"])
            elif resume_from is None:
                _mark_no_snapshot(res, opts.integrator, opts.t_start)
            if "device_losses" in native_res:
                res._set_device_losses(native_res["device_losses"],
                                       native_res["heat_injected"],
                                       native_res["heat_pending"])
            res.initial_condition_adjustment = ic_adjustment
            res.control_tick_counts = [int(n) for n in native_res.get(
                "control_tick_counts", ())]
            if mode_auto is not None:
                res.diagnostics["mode_selection_auto"] = mode_auto
            return res
        except (UnsupportedError, _InconsistentInitialState, ConvergenceError,
                NetlistError):
            # A refusal is an answer, not a reason to try the next
            # bridge (which would run the circuit without the device);
            # so is a give-up of the run the native adapter accepted
            # (the next bridge would re-run it without the refusal of
            # implicit segments).
            raise
        except (RuntimeError, ValueError) as e:
            # Bridge.11 native adapter rejected the circuit (e.g. nonlinear
            # device that the C++ extractor can't handle). Silently fall
            # through to Bridge.10 — the Python adapter has identical
            # semantics + the same error path on the same circuits.
            #
            # EXCEPT for diode circuits (v2.0 Phase 3): only the
            # Bridge.11 path carries the auto-derived predicates, so
            # falling through would freeze the diode bits and
            # reintroduce the silent wrong answer. Re-raise instead.
            #
            # Nor once the native adapter accepted the circuit and the
            # exact engine refuses implicit segments: the fallback would
            # run it numerically (an untyped kernel error mid-run, e.g. a
            # root finder that did not converge, would come back as a
            # silent numeric run; integration batch 9 review).
            if ticks:
                raise unsupported(
                    "dsed", "step_observer", message=
                    "simulate(engine='dsed'): control ticks run on the native "
                    f"adapter only, which rejected this circuit ({e}). Use "
                    "engine='pwl' or 'trbdf2'.") from e
            if _n_diodes > 0 or _loss_names or resume_from is not None or (
                    not adapter_rejected and not allow_implicit_segments
                    and opts.integrator != "bdf2"):
                raise

    if ticks:
        raise unsupported(
            "dsed", "step_observer", message=
            "simulate(engine='dsed'): control ticks run on the native adapter "
            "only, which is not available for this circuit. Use engine='pwl' "
            "or 'trbdf2'.")
    if output_rtol:
        raise unsupported(
            "dsed", "output_rtol", message=
            "simulate(engine='dsed', output_rtol=...): the output-grid rule runs "
            "on the native adapter only, which is not available for this circuit. "
            "Drop output_rtol, or pass dt.")
    if resume_from is not None:
        raise unsupported(
            "dsed", "resume_from", message=
            "simulate(engine='dsed', resume_from=...): the snapshot is read by "
            "the native adapter only, which rejected this circuit. Use "
            "engine='pwl' or 'trbdf2'.")
    if full_x0 is not None and ambiguous:
        x0 = full_x0          # the non-native bridges read it as reduced, as before
    elif full_x0 is not None:
        raise _options_error(
            "initial_state",
            f"simulate(engine='dsed'): initial_state has shape {full_x0.shape}, "
            f"expected ({n_state},); a full-MNA vector is accepted only on the "
            "native adapter.")
    if mode_selection in ("shadow", "lcp"):
        raise unsupported(
            "dsed", "mode_selection", message=
            f"simulate(engine='dsed', mode_selection={mode_selection!r}): "
            "the native adapter rejected this circuit and the fallback "
            "path has no complementarity hook. Drop mode_selection.")

    if not allow_implicit_segments and adapter_rejected and opts.integrator != "bdf2":
        raise unsupported(
            "dsed", "extraction", message=
            "simulate(engine='exact'): the exact kernel's state extraction "
            "rejected this circuit, so every segment would be numeric. Pass "
            "allow_implicit_segments=True, or use engine='pwl' / 'trbdf2'.")

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

    if fixed_step:
        raise unsupported(
            "dsed", "integrator", message=
            "simulate(engine='exact', integrator='fixed'): the native adapter "
            "rejected this circuit and the fallback path has no fixed-step form. "
            "Drop integrator='fixed'.")
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
        out = _exact_result(
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
        if mode_auto is not None:
            out.diagnostics = {"mode_selection_auto": mode_auto}
        _mark_no_snapshot(out, opts.integrator, opts.t_start,
                          bridge="Python adapter fallback")
        return out

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


# A stored sample whose diode mode breaks its own law: the kernel's end-of-run
# check (PEDSimulator::check_diode_law_, every caller gets it), typed with
# `.device` and `.t_failed`. Containment for the zero-band settle of slice 1
# (a boost with V_th = 0 kept its diode ON through -1.07 A; a half-wave RL
# rectifier conducted -0.85 A for half a period, silently).
from ._pulsim import DiodeModeInconsistentError  # noqa: E402, F401  # type: ignore[import-not-found]

