"""The option x engine table, read from the kernel (add-kernel-simulation-object).

`sim::kRunOptionTable` (core/include/pulsim/sim/options.hpp) is the one
declaration of every `simulate()` keyword: the engines that honour it,
which value means "not requested", and the mechanism the other engines
lack. `simulate()` asks `refused(engine, values)` once, above the engine
dispatch, so an option is honoured or refused by name on every engine and
never accepted and ignored. `sim::Defaults` is the one table of default
values (`DEFAULTS`).
"""

from __future__ import annotations

from typing import Any, Mapping

from . import _pulsim  # type: ignore[import-not-found]

#: name -> row dict (engines, unset, default, why), in table order.
TABLE: "dict[str, dict]" = {r["name"]: r for r in _pulsim.run_option_table()}
#: THE default values (sim::Defaults).
DEFAULTS: "dict[str, Any]" = dict(_pulsim.run_option_defaults())

_ENGINES = ("pwl", "trbdf2", "dsed")


def is_requested(name: str, value: Any, engine: str = "") -> bool:
    """True when `value` asks an engine to do something for option `name`
    (so an engine outside its row must refuse it)."""
    kind = TABLE[name]["unset"]
    if kind == "always" or value is None:
        return False
    if kind == "none":
        return True
    if kind == "falsy":
        return bool(value)
    if kind == "none_or_one":
        return int(value) != 1
    if kind == "none_or_zero":
        return int(value) != 0
    if kind == "legacy":
        return value != "legacy"
    if kind == "label":
        return value != "label"
    if kind == "integrator":
        # The fixed engine's own integrator names ask nothing of it.
        return not (engine == "pwl" and value in ("kernel", "default"))
    raise AssertionError(f"unknown unset kind {kind!r}")  # table/py drift


def honoured(name: str, engine: str) -> bool:
    return engine in TABLE[name]["engines"]


def refused(engine: str, values: Mapping[str, Any]) -> "list[str]":
    """The options of `values` that `engine` cannot honour, in table order."""
    return [n for n in TABLE if n in values
            and is_requested(n, values[n], engine) and not honoured(n, engine)]


def why(name: str) -> str:
    return TABLE[name]["why"]


def option_capabilities() -> "dict[str, dict[str, bool]]":
    """The option x engine table: ``{keyword: {engine: honoured}}``.

    One row per :func:`pulsim.simulate` keyword. An engine marked False
    refuses the keyword with :class:`UnsupportedError` (``feature`` = the
    keyword) when it is set to a value that asks for something; it is never
    accepted and ignored. ``pulsim.engine_capabilities()`` is the device
    table. Engines: ``'pwl'`` (fixed step), ``'trbdf2'`` (variable step),
    ``'dsed'`` (the exact engine, ``engine='exact'``).
    """
    return {n: {e: e in r["engines"] for e in _ENGINES}
            for n, r in TABLE.items() if n != "c_blocks"}


def run_options(*, engine: str, t_start: float, t_end: float, dt=None,
                rtol=None, atol=None, dt_init=None, stiffness_threshold=None,
                h_bdf2=None, enable_nonlinear_refresh=None,
                max_newton_iterations=None, tol_newton_dx=None,
                tol_newton_res=None, enable_newton_line_search=None,
                enable_newton_lm=None, newton_roundoff_budget=None,
                max_event_iterations=None, strict_event_iterations=False,
                mode_selection=None, on_impulse="record", mode_guard=True,
                control_tick="label",
                max_dt_halvings=None, store_every=None, start_from_dc_op=False,
                enable_substep_state_correction=None,
                discontinuity_restart=None, resolution_diagnostics=None,
                inductor_freeze_di_max=None, inductor_abs_clamp=None,
                auto_regularize=True, voltage_sanity_factor=None):
    """A kernel `RunOptions` from simulate()'s resolved keywords (None =
    not set; the kernel's `to_simulation_options` / `to_trbdf2_options`
    apply `sim::Defaults`)."""
    o = _pulsim.RunOptions(t_end=float(t_end), dt=None if dt is None
                           else float(dt), engine=engine,
                           t_start=float(t_start))
    for name, v, cast in (
            ("rtol", rtol, float), ("atol", atol, float),
            ("dt_init", dt_init, float),
            ("stiffness_threshold", stiffness_threshold, float),
            ("h_bdf2", h_bdf2, float),
            ("enable_nonlinear_refresh", enable_nonlinear_refresh, bool),
            ("max_newton_iterations", max_newton_iterations, int),
            ("tol_newton_dx", tol_newton_dx, float),
            ("tol_newton_res", tol_newton_res, float),
            ("enable_newton_line_search", enable_newton_line_search, bool),
            ("enable_newton_lm", enable_newton_lm, bool),
            ("newton_roundoff_budget", newton_roundoff_budget, int),
            ("max_event_iterations", max_event_iterations, int),
            ("max_dt_halvings", max_dt_halvings, int),
            ("store_every", store_every, int),
            ("enable_substep_state_correction",
             enable_substep_state_correction, bool),
            ("discontinuity_restart", discontinuity_restart, bool),
            ("resolution_diagnostics", resolution_diagnostics, bool),
            ("inductor_freeze_di_max", inductor_freeze_di_max, float),
            ("inductor_abs_clamp", inductor_abs_clamp, float),
            ("voltage_sanity_factor", voltage_sanity_factor, float)):
        if v is not None:
            setattr(o, name, cast(v))
    o.strict_event_iterations = bool(strict_event_iterations)
    o.mode_selection = mode_selection or "legacy"
    o.on_impulse = on_impulse
    o.mode_guard = bool(mode_guard)
    o.control_tick = control_tick
    o.start_from_dc_op = bool(start_from_dc_op)
    o.auto_regularize = bool(auto_regularize)
    return o


def simulation_options(**kw):
    """The fixed-step driver's `SimulationOptions`, resolved by the kernel
    (`sim::to_simulation_options`) from `run_options(**kw)`."""
    return _pulsim._to_simulation_options(run_options(**kw))


def trbdf2_options(**kw) -> "dict[str, Any]":
    """The variable-step driver's options, resolved by the kernel
    (`sim::to_trbdf2_options`) from `run_options(**kw)`."""
    return dict(_pulsim._to_trbdf2_options(run_options(**kw)))
