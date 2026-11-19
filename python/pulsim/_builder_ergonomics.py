"""add-python-builder-ergonomics (v1.5) — Python-side builder helpers.

Three GUI papercuts surfaced during PulsimGUI's migration to pulsim
1.4 (PulsimGUI PR #9):

1. Initial conditions on capacitors / inductors had no slot on the
   builder; the v0 ``add_capacitor(name, n1, n2, C, v0)`` 5th
   positional argument was silently dropped by pulsim 1.4.
2. Human node aliases (the GUI's ``"n_dc_plus"`` → ``"node_42"`` map)
   were out-of-band metadata the GUI had to maintain in parallel
   because the builder only knew electrical node names.
3. ``should_continue`` was on ``simulate`` only; the cancel button
   couldn't preempt ``compute_dc_op`` / ``run_ac_sweep`` /
   ``compute_temperature`` mid-call.

Initial conditions (``set_initial``, ``c0=`` / ``i0=``) and aliases
(``set_alias`` / ``aliases``) are stored by the C++ builder, so a
:class:`pulsim.Circuit` carries them (add-immutable-circuit-ir group 3
removed the Python side tables). This module keeps the alias resolver
the result accessors consult and the retired-device guard.

Cancellation: the existing ``simulate(should_continue=…)`` already
preempts the transient solver. The other top-level analyses
(``compute_dc_op``, ``run_ac_sweep``, ``compute_temperature``) gain
the kwarg here as Python-side wrappers that check the callback
between calls but cannot interrupt mid-call (preemption requires
C++ kernel changes, captured in the proposal's tasks 3.1-3.4 for a
follow-up). The wrapper still raises :class:`pulsim.Cancelled` when
the caller signalled cancellation before the call started — useful
for "user clicked Cancel between Bode sweep points" scenarios where
the next call shouldn't fire.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ._pulsim import PulsimError  # type: ignore[import-not-found]

if TYPE_CHECKING:
    from ._pulsim import CircuitBuilder, IdealDiodeParams


class RetiredDeviceError(ValueError):
    """Raised by a builder method whose device was retired because it
    gives wrong answers (today: ``add_nonlinear_diode``). The message
    names the replacements and the explicit opt-in that still builds
    the legacy device."""


class RetiredYamlDeviceError(RetiredDeviceError, RuntimeError):
    """The YAML loader's refusal of a retired device type. A
    ``RetiredDeviceError`` like the builder's, and still a
    ``RuntimeError`` like every other YAML refusal."""


class Cancelled(RuntimeError, PulsimError):
    """Raised by long-running analyses when the user's
    ``should_continue`` callback returns ``False``.

    Subclasses :class:`RuntimeError` so existing
    ``try: ... except RuntimeError: ...`` patterns still catch
    cancellations without forcing migration.

    Attributes
    ----------
    where : str
        Identifier of the analysis that was cancelled
        (``"compute_dc_op"`` / ``"run_ac_sweep"`` / …).
    point_index : int | None
        For sweep-style analyses, the index of the next point that
        would have been computed. ``None`` for non-sweep paths.
    """

    __slots__ = ("where", "point_index")

    def __init__(
        self,
        where: str,
        point_index: "int | None" = None,
    ) -> None:
        self.where = where
        self.point_index = point_index
        loc = f" at point {point_index}" if point_index is not None else ""
        super().__init__(f"{where} cancelled{loc}.")


# ---------------------------------------------------------------------------
# Cancellation wrapper for analyses
# ---------------------------------------------------------------------------
def wrap_with_cancellation(fn, *, where: str):
    """Wrap a long-running analysis ``fn`` so its existing
    ``should_continue`` semantics propagate through a typed
    :class:`Cancelled` exception.

    Most pulsim analyses ignore the kwarg today; the wrapper at
    least preempts re-entry — if a sweep is in a Python-level loop
    over `fn`, callers can pass ``should_continue`` and the next
    iteration won't fire after the user clicked Cancel.

    Note: this is a *partial* implementation of the spec. Mid-call
    preemption requires C++ kernel changes, captured in the
    proposal's tasks 3.1-3.4. We ship this hook so PulsimGUI can
    wire its cancel button today.
    """
    def _wrapped(*args, should_continue=None, **kwargs):
        if should_continue is not None and not should_continue():
            raise Cancelled(where)
        # Forward should_continue to the underlying analysis when it
        # accepts the kwarg (introspect the signature once, cache).
        try:
            import inspect
            params = inspect.signature(fn).parameters
            if "should_continue" in params and should_continue is not None:
                kwargs["should_continue"] = should_continue
        except (TypeError, ValueError):  # pragma: no cover — defensive
            pass
        return fn(*args, **kwargs)
    _wrapped.__name__ = getattr(fn, "__name__", "_wrapped")
    _wrapped.__doc__ = getattr(fn, "__doc__", None)
    return _wrapped


# ---------------------------------------------------------------------------
# Module-level installer
# ---------------------------------------------------------------------------
def install(builder_cls) -> None:
    """Patch the named methods onto ``builder_cls``. Called once at
    ``pulsim`` module-import time so every ``CircuitBuilder`` instance
    transparently gains the helpers.

    ``set_initial`` / ``initial_state`` / ``set_alias`` / ``aliases``
    are C++ (the builder stores them, so a Circuit carries them;
    add-immutable-circuit-ir group 3 deleted the Python side tables
    ``_pulsim_ic`` / ``_pulsim_aliases``). ``_resolve_alias`` is the
    wrapper :meth:`SimulationResult.v` / ``.i`` / ``.power`` consult.
    """
    # _resolve_alias bridges to the builder's (C++) `aliases()`.
    builder_cls._resolve_alias = _resolve_alias_native  # type: ignore[attr-defined]
    native = getattr(builder_cls, "add_nonlinear_diode", None)
    if native is not None and not getattr(native, "_retired", False):
        builder_cls.add_nonlinear_diode = _retired_nonlinear_diode(native)  # type: ignore[attr-defined]


NONLINEAR_DIODE_RETIRED = (
    "add_nonlinear_diode is retired: its smooth-blend law "
    "logistic(kappa*(v - V_F0))*(v - V_F0)/R_d is negative for "
    "0 < v < V_F0, so the device sources power below its knee "
    "(-1.39 A out of a forward-biased anode at +0.636 V with the "
    "default IdealDiodeParams). Use add_diode(name, anode, cathode, "
    "g_on, g_off, V_f=...) for a PWL diode with a forward drop, or "
    "add_shockley_diode(name, anode, cathode, ...) for a smooth "
    "junction. Pass allow_non_passive=True to build the legacy device "
    "anyway (hostile-circuit test fixtures).")


def _retired_nonlinear_diode(native):
    """``add_nonlinear_diode`` that refuses unless the caller opts in
    with ``allow_non_passive=True``; the native call is unchanged."""
    def add_nonlinear_diode(self, name: str, anode: str, cathode: str,
                            params: IdealDiodeParams, *,
                            allow_non_passive: bool = False) -> CircuitBuilder:
        if not allow_non_passive:
            raise RetiredDeviceError(f"{name}: {NONLINEAR_DIODE_RETIRED}")
        return native(self, name, anode, cathode, params)

    add_nonlinear_diode.__doc__ = (
        "Retired. Raises RetiredDeviceError unless "
        "allow_non_passive=True.\n\n" + NONLINEAR_DIODE_RETIRED)
    add_nonlinear_diode.__qualname__ = "CircuitBuilder.add_nonlinear_diode"
    add_nonlinear_diode._retired = True  # type: ignore[attr-defined]
    return add_nonlinear_diode


def shockley_params(builder, name: str) -> "dict | None":
    """The ``add_shockley_diode`` parameters of device ``name``
    (``I_S``, ``n``, ``V_T``, ``G_min``, ``BV``), read from the
    device's own value fields, or ``None`` when ``name`` is not a
    Shockley diode."""
    try:
        kind = builder.pool.kind_of(builder.branch_id_of(str(name)))
    except Exception:  # noqa: BLE001 — not a device of this builder
        return None
    if getattr(kind, "name", None) != "ShockleyDiode":
        return None
    return {f: float(builder.get_value(f"{name}.{f}"))
            for f in ("I_S", "n", "V_T", "G_min", "BV")}


def _resolve_alias_native(self, name: str):
    """Consult the builder's :meth:`aliases` map (C++ or Python
    side) for ``name``. Returns ``(kind, target)`` tuple or ``None``.

    Handles both shapes the underlying dict can take:
    - C++ binding: ``{name: ("node"|"branch", target)}`` tuples.
    - Python fallback: same shape via ``_aliases``.
    """
    try:
        amap = self.aliases()
    except Exception:  # noqa: BLE001 — defensive
        return None
    entry = amap.get(name)
    if entry is None:
        return None
    # entry is either (kind_str, target_str) or AliasTarget-like.
    if isinstance(entry, tuple) and len(entry) == 2:
        return entry
    kind = getattr(entry, "kind", None)
    target = getattr(entry, "target", None)
    if kind is not None and target is not None:
        return (str(kind), str(target))
    return None


__all__ = [
    "Cancelled",
    "install",
    "wrap_with_cancellation",
]
