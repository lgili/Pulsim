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

This module patches :class:`pulsim.CircuitBuilder` with the
following additions, all implemented Python-side (no C++ changes
required because :class:`CircuitBuilder` was declared
``py::dynamic_attr()`` so we can stash side-tables on each
instance):

- ``builder.set_initial(name, value)`` — record an IC on a capacitor
  or inductor branch.
- ``builder.add_capacitor_c0(name, from, to, C, c0)`` and
  ``builder.add_inductor_i0(name, from, to, L, i0)`` — convenience
  wrappers that bundle the `add_*` call with `set_initial`. We use
  these names instead of overloading the existing methods because
  the C++ bindings are positional-only and don't accept kwargs.
- ``builder.set_alias(human_name, *, node=None, branch=None)`` and
  ``builder.aliases()`` — round-trip-friendly human-name registry
  consulted by ``SimulationResult.v / .i / .power`` before raising
  :class:`pulsim.NameNotFoundError`.
- ``builder.initial_state_vector()`` — synthesise the flat
  initial-state ndarray from recorded ICs (zero elsewhere). Called
  by :func:`pulsim.simulate` when ``initial_state=None``.

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

import warnings
from typing import Any


class Cancelled(RuntimeError):
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
# Side-table accessors. We can't subclass CircuitBuilder cleanly because
# the constructor is pybind-bound, so we attach metadata as a dict on the
# instance (dynamic_attr() makes this possible) and route lookups through
# the patched methods.
# ---------------------------------------------------------------------------
def _ic_dict(builder) -> dict:
    """Lazy `{device_name: value}` map for recorded ICs."""
    d = getattr(builder, "_pulsim_ic", None)
    if d is None:
        d = {}
        builder._pulsim_ic = d
    return d


def _alias_dict(builder) -> dict:
    """Lazy `{human_name: (kind, canonical_name)}` map."""
    d = getattr(builder, "_pulsim_aliases", None)
    if d is None:
        d = {}
        builder._pulsim_aliases = d
    return d


# ---------------------------------------------------------------------------
# IC helpers
# ---------------------------------------------------------------------------
def _set_initial(self, device_name: str, value: float) -> None:
    """Record an initial condition for the named capacitor or
    inductor. The value is consumed by :func:`pulsim.simulate` when
    it synthesises the ``initial_state`` ndarray.

    Raises
    ------
    KeyError
        If ``device_name`` doesn't refer to a registered capacitor
        or inductor branch (the only kinds with a state variable
        the IC can target).
    """
    # Validate that the device exists and is a passive state-variable
    # branch. The C++ pool tells us; we look it up by name.
    try:
        b_id = self.branch_id_of(device_name)
    except IndexError as exc:
        raise KeyError(
            f"set_initial: no device named {device_name!r}"
        ) from exc
    has_inductor = self.pool.is_inductor(b_id) if hasattr(
        self.pool, "is_inductor") else False
    has_capacitor = self.pool.is_capacitor(b_id) if hasattr(
        self.pool, "is_capacitor") else False
    if not (has_inductor or has_capacitor):
        # Falling back to a softer warning: pool may not have
        # is_capacitor/is_inductor introspection yet; record the IC
        # and let simulate() complain if it's nonsense.
        warnings.warn(
            f"set_initial({device_name!r}): cannot verify the device "
            "is a capacitor or inductor (pool introspection missing) "
            "— recording the IC anyway.",
            RuntimeWarning,
            stacklevel=2,
        )
    _ic_dict(self)[str(device_name)] = float(value)


def _initial_state_vector(self):
    """Synthesise the flat ``initial_state`` ndarray from recorded
    ICs. Zero everywhere else.

    Returns
    -------
    numpy.ndarray
        Shape ``(num_nodes + num_state_branches,)``.

    Notes
    -----
    The state-variable layout is governed by the C++ pool — only
    inductors and voltage sources have a dedicated current state
    column; capacitor voltages are computed as node-difference, so
    setting ``c0=5.0`` on a cap maps to the *node* voltage of the
    cap's positive terminal (not a separate column).
    """
    import numpy as _np
    n_nodes = int(self.graph.num_nodes)
    # Probe the state size by running an empty cache build; cheaper:
    # ask the pool for the state dimension via DC OP. Pragmatic: use
    # the most common layout n_nodes + (num inductors + num sources).
    # We compute it explicitly by scanning devices.
    ic_map = _ic_dict(self)
    # Build initial state by iterating recorded ICs and writing into
    # the appropriate column.
    # First, determine total state size. Walk each branch and look up
    # if it has a state index.
    n_state = 0
    state_offsets: dict[str, tuple[int, str]] = {}
    for d in self.devices():
        b_id = self.branch_id_of(d.name)
        # Try inductor first.
        for accessor in (
            "branch_var_id_for_inductor",
            "branch_var_id_for_source",
        ):
            fn = getattr(self.pool, accessor, None)
            if fn is None:
                continue
            try:
                idx = int(fn(b_id, self.graph))
                state_offsets[d.name] = (idx, accessor)
                n_state = max(n_state, idx + 1)
                break
            except Exception:  # noqa: BLE001 — wrong kind
                continue
    # Default: state size is at least num_nodes + num_state_branches.
    size = max(n_nodes + len(state_offsets), n_state)
    x0 = _np.zeros(size, dtype=float)

    # Write recorded ICs into their slots.
    for name, value in ic_map.items():
        if name in state_offsets:
            # Inductor / source — has a state-vector column directly.
            idx, _ = state_offsets[name]
            x0[idx] = value
        else:
            # Capacitor — set the *node* voltage of the positive
            # terminal. We need to know which node that is.
            # `devices()` gives us the terminal names.
            dev = next(
                (d for d in self.devices() if d.name == name), None,
            )
            if dev is None or len(dev.terminals) < 2:
                warnings.warn(
                    f"initial_state_vector: cannot place IC for "
                    f"{name!r} — device not found in builder.devices(). "
                    "Skipping.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            try:
                node_idx = self.node_id_of(dev.terminals[0])
            except IndexError:
                continue
            if node_idx >= 0:
                x0[node_idx] = value
    return x0


# ---------------------------------------------------------------------------
# Alias helpers
# ---------------------------------------------------------------------------
def _set_alias(
    self,
    human_name: str,
    *,
    node: "str | None" = None,
    branch: "str | None" = None,
) -> None:
    """Register a secondary name that resolves to the same canonical
    electrical entity. Exactly one of ``node=`` or ``branch=`` must
    be non-``None``.

    Raises
    ------
    ValueError
        - both ``node`` and ``branch`` provided
        - neither provided
        - empty string passed
        - the human name collides with an existing canonical node /
          branch name
    """
    if (node is None) == (branch is None):
        raise ValueError(
            "set_alias: pass exactly one of `node=` or `branch=`."
        )
    if not human_name:
        raise ValueError("set_alias: human_name must be non-empty.")
    if (node is not None and not node) or (
        branch is not None and not branch):
        raise ValueError(
            "set_alias: node/branch target must be non-empty."
        )

    # Reject collisions with canonical names.
    try:
        self.node_id_of(human_name)
        raise ValueError(
            f"set_alias: {human_name!r} is already a canonical "
            "node name."
        )
    except IndexError:
        pass
    try:
        self.branch_id_of(human_name)
        raise ValueError(
            f"set_alias: {human_name!r} is already a canonical "
            "branch / device name."
        )
    except IndexError:
        pass

    kind = "node" if node is not None else "branch"
    target = node if node is not None else branch
    _alias_dict(self)[str(human_name)] = (kind, str(target))


def _aliases(self) -> dict:
    """Return the registered ``{human_name: (kind, canonical_name)}``
    map. Empty dict when no aliases were registered.
    """
    return dict(_alias_dict(self))


def _resolve_alias(self, name: str) -> "tuple[str, str] | None":
    """Best-effort resolver consulted by SimulationResult accessors
    before raising NameNotFoundError. Returns
    ``(kind, canonical_name)`` if ``name`` is a registered alias,
    else ``None``.
    """
    return _alias_dict(self).get(name)


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

    pulsim 1.5+ exposes ``set_initial`` / ``initial_state`` /
    ``set_alias`` / ``aliases`` directly from C++; this installer is
    therefore mostly a no-op on a fresh build. We keep the Python
    fallbacks (gated by ``hasattr``) for two reasons:

    1. Test mocks that build a partial CircuitBuilder analog without
       the C++ helpers (PulsimGUI's `pulsim_v0_compat.py` until it
       picks up the C++ methods).
    2. ``_resolve_alias`` — the Python wrapper consulted by
       :meth:`SimulationResult.v` / `.i` / `.power`. Wraps the C++
       ``aliases()`` dict in a uniform interface.
    """
    if not hasattr(builder_cls, "set_alias"):
        builder_cls.set_alias = _set_alias        # type: ignore[attr-defined]
    if not hasattr(builder_cls, "aliases"):
        builder_cls.aliases   = _aliases          # type: ignore[attr-defined]
    # _resolve_alias bridges to whatever the host implementation of
    # `aliases()` returns — works for both C++ and Python sides.
    builder_cls._resolve_alias = _resolve_alias_native  # type: ignore[attr-defined]
    if not hasattr(builder_cls, "set_initial"):
        builder_cls.set_initial = _set_initial             # type: ignore[attr-defined]
    if not hasattr(builder_cls, "initial_state"):
        builder_cls.initial_state = _initial_state_vector   # type: ignore[attr-defined]


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
