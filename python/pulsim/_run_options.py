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

from itertools import compress, repeat
from operator import is_not, truth
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
    if kind == "integrator":
        # The fixed engine's own integrator names ask nothing of it.
        return not (engine == "pwl" and value in ("kernel", "default"))
    raise AssertionError(f"unknown unset kind {kind!r}")  # table/py drift


def honoured(name: str, engine: str) -> bool:
    return engine in TABLE[name]["engines"]


# Per engine, the rows it does not honour, in table order (simulate() asks
# on every call: only these can be refused), split by how "not set" is
# spelled: falsy rows (any falsy value) and the rest (None, or by kind).
def _split(names):
    falsy = tuple(n for n in names if TABLE[n]["unset"] == "falsy")
    return falsy, tuple(n for n in names if TABLE[n]["unset"] != "falsy")


_ANY_ROW = _split(tuple(n for n, r in TABLE.items() if r["unset"] != "always"))
_NOT_HONOURED = {e: _split(tuple(
    n for n, r in TABLE.items()
    if e not in r["engines"] and r["unset"] != "always")) for e in _ENGINES}
_ORDER = {n: k for k, n in enumerate(TABLE)}

#: A pseudo-engine for `refused`: the trbdf2 rows that make `engine='auto'`
#: route to pwl (simulate()'s `_trbdf2_blockers`). Not every fixed-step-only
#: row: `mode_selection` set on a dt-less 'auto' run is refused, not routed.
TRBDF2_ROUTE = "trbdf2 route"
_NOT_HONOURED[TRBDF2_ROUTE] = tuple(
    tuple(n for n in rows if n in (
        "start_from_dc_op", "strict_event_iterations", "max_dt_halvings",
        "store_every", "enable_substep_state_correction",
        "discontinuity_restart", "split_snap", "resolution_diagnostics",
        "inductor_freeze_di_max", "inductor_abs_clamp", "progress",
        "live_stream", "mmc_arms", "on_cache"))
    for rows in _NOT_HONOURED["trbdf2"])


def refused(engine: str, values: Mapping[str, Any]) -> "list[str]":
    """The options of `values` that `engine` cannot honour, in table order."""
    falsy, other = _NOT_HONOURED.get(engine, _ANY_ROW)  # unknown: no row
    # The "not set" filters run in C (simulate() asks on every call and
    # nearly every value is unset); only the set ones reach is_requested.
    get = values.get
    out = list(compress(falsy, map(truth, map(get, falsy))))
    asked = list(compress(other, map(is_not, map(get, other), repeat(None))))
    if asked:
        out += [n for n in asked if is_requested(n, get(n), engine)]
    if len(out) > 1:
        out.sort(key=_ORDER.__getitem__)
    return out


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


def run_options(*, engine: str, t_start: float, t_end: float, dt=None, **kw):
    """A kernel `RunOptions` from simulate()'s resolved keywords (None =
    not set; the kernel's `to_simulation_options` / `to_trbdf2_options`
    apply `sim::Defaults`; a negative count is an `OptionsError`)."""
    return _pulsim._run_options_kw(t_end, dt, engine, t_start, **kw)


def simulation_options(*, engine: str, t_start: float, t_end: float,
                       dt=None, **kw):
    """The fixed-step driver's `SimulationOptions`, resolved by the kernel
    (`sim::to_simulation_options`) from the same keywords."""
    return _pulsim._simulation_options_kw(t_end, dt, engine, t_start, **kw)


def trbdf2_options(*, engine: str, t_start: float, t_end: float, dt=None,
                   **kw) -> "dict[str, Any]":
    """The variable-step driver's options, resolved by the kernel
    (`sim::to_trbdf2_options`) from the same keywords."""
    return dict(_pulsim._trbdf2_options_kw(t_end, dt, engine, t_start, **kw))


def markdown_table() -> str:
    """The option x engine table as the Markdown block of
    docs/known-limitations.md (a test keeps the page equal to it)."""
    rows = ["| Keyword | pwl | trbdf2 | exact (dsed) | default |",
            "|---|---|---|---|---|"]
    for n, r in TABLE.items():
        if n == "c_blocks":
            continue
        cells = " | ".join("yes" if e in r["engines"] else "refused"
                           for e in _ENGINES)
        rows.append(f"| `{n}` | {cells} | {r['default']} |")
    return "\n".join(rows)
