"""add-immutable-circuit-ir — the Python side of :class:`pulsim.Circuit`.

``simulate()`` runs every circuit on its own copy and reports what the
run did to it as a :class:`RunPlan` (``result.plan``): the circuit that
ran (``result.circuit``) and the regularising ties the preflight added.
"""
from __future__ import annotations

import ast
import inspect
import re
import warnings
from dataclasses import dataclass
from typing import Any, NamedTuple

from ._pulsim import Parameter


class Tie(NamedTuple):
    """A regularising resistor the preflight added: ``name`` from node
    index ``node`` to ground, ``resistance`` ohms."""
    name: str
    node: int
    resistance: float


@dataclass(frozen=True)
class RunPlan:
    """What :func:`pulsim.simulate` ran: ``circuit`` (a
    :class:`pulsim.Circuit`, with the ties), ``ties`` and the
    ``preflight`` report (``None`` with ``auto_regularize=False``)."""
    circuit: Any
    ties: tuple
    preflight: Any = None


# ---------------------------------------------------------------------------
# Group 6: named parameters as `add_*` arguments, generic updates.
# ---------------------------------------------------------------------------
_SIG = re.compile(r"^\s*(?:\d+\.\s*)?\w+\((.*?)\)\s*->", re.M)


def _split_args(text: str) -> list:
    """Split a signature's argument list on top-level commas."""
    args, depth, cur = [], 0, ""
    for ch in text:
        depth += (ch in "[(") - (ch in "])")
        if ch == "," and depth == 0:
            args.append(cur)
            cur = ""
        else:
            cur += ch
    return [a.strip() for a in args + [cur] if a.strip()]


def _signature(fn):
    """``inspect.Signature`` of a builder method: its own for a Python
    function, else read from the signature line pybind11 writes at the
    top of ``__doc__`` (names, kinds, defaults; no annotations)."""
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        pass
    m = _SIG.search(getattr(fn, "__doc__", "") or "")
    if not m:
        return None
    P = inspect.Parameter
    params, kind = [], P.POSITIONAL_OR_KEYWORD
    for a in _split_args(m.group(1)):
        head, _, default = a.partition("=")
        name = head.split(":")[0].strip()
        if name == "*":
            kind = P.KEYWORD_ONLY
            continue
        if name.startswith("**"):
            params.append(P(name[2:], P.VAR_KEYWORD))
            continue
        if name.startswith("*"):
            params.append(P(name[1:], P.VAR_POSITIONAL))
            kind = P.KEYWORD_ONLY
            continue
        d = P.empty
        if default.strip():
            try:
                d = ast.literal_eval(default.strip())
            except (ValueError, SyntaxError):
                d = default.strip()
        params.append(P(name, kind, default=d))
    return inspect.Signature(params)


def _arg_names(fn) -> list:
    """Positional argument names of a builder method, ``self`` excluded."""
    sig = _signature(fn)
    if sig is None:
        return []
    P = inspect.Parameter
    return [q.name for q in sig.parameters.values()
            if q.kind in (P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD)][1:]


def _accepting_parameters(native, method: str):
    """``native`` that also takes a :class:`pulsim.Parameter` wherever it
    takes a number: the parameter's value is passed, and the device field
    named like the argument is bound to the parameter, so
    ``Circuit.with_params`` re-values it."""
    names = _arg_names(native)

    def add(self, *args, **kwargs):
        # The common call (numbers only) costs one type test per argument.
        for a in args:
            if type(a) is Parameter:
                break
        else:
            for v in kwargs.values():
                if type(v) is Parameter:
                    break
            else:
                return native(self, *args, **kwargs)
        binds, args = [], list(args)
        for i, a in enumerate(args):
            if isinstance(a, Parameter):
                if i >= len(names):
                    raise TypeError(
                        f"{method}(): pass the parameter {a.name!r} by "
                        "keyword (the argument name is not known)")
                binds.append((names[i], a))
                args[i] = a.value
        for k, v in list(kwargs.items()):
            if isinstance(v, Parameter):
                binds.append((k, v))
                kwargs[k] = v.value
        out = native(self, *args, **kwargs)
        dev = args[0] if args else kwargs.get("name")
        for arg, prm in binds:
            try:
                self.bind_param(f"{dev}.{arg}", prm.name)
            except ValueError as exc:
                raise ValueError(
                    f"{method}({dev!r}): argument {arg!r} cannot take the "
                    f"parameter {prm.name!r}: {exc}") from None
        return out

    add.__doc__ = native.__doc__
    add.__qualname__ = f"CircuitBuilder.{method}"
    sig = _signature(native)
    if sig is not None:  # help(), IDEs and the stub check see the native one
        add.__signature__ = sig  # type: ignore[attr-defined]
    add._accepts_parameters = True  # type: ignore[attr-defined]
    add._native = native  # type: ignore[attr-defined]
    return add


_DEPRECATED_UPDATES = {
    "update_resistor_R": "R", "update_inductor_L": "L",
    "update_capacitor_C": "C", "update_voltage_source_V": "V",
}


def _deprecated_update(native, method: str, field: str):
    def update(self, *args, **kwargs):
        name = args[0] if args else kwargs.get("name", "<device>")
        warnings.warn(
            f"CircuitBuilder.{method} is deprecated: use "
            f"circuit.with_value(\"{name}.{field}\", value) on an immutable "
            f"pulsim.Circuit, or builder.set_value(\"{name}.{field}\", "
            "value) to edit a builder in place (every field of every "
            "kind).", DeprecationWarning, stacklevel=2)
        return native(self, *args, **kwargs)

    update.__doc__ = native.__doc__
    update.__qualname__ = f"CircuitBuilder.{method}"
    sig = _signature(native)
    if sig is not None:
        update.__signature__ = sig  # type: ignore[attr-defined]
    update._deprecated = True  # type: ignore[attr-defined]
    update._native = native  # type: ignore[attr-defined]
    return update


def _checked_build(native):
    def build(self):
        blocks = getattr(self, "_c_blocks", None)
        if blocks:
            from . import UnsupportedError
            names = ", ".join(str(getattr(h, "name", h)) for h in blocks)
            raise UnsupportedError(
                f"build(): this builder carries C blocks ({names}) that a "
                "pulsim.Circuit does not hold yet (add-immutable-circuit-ir "
                "group 3); simulate the builder itself")
        return native(self)

    build.__doc__ = native.__doc__
    build.__qualname__ = "CircuitBuilder.build"
    build._checks_side_state = True  # type: ignore[attr-defined]
    return build


def install(builder_cls) -> None:
    """Patch the ``add_*`` family (Parameter arguments), the four
    deprecated ``update_*`` methods and ``build`` (refuses what a
    Circuit cannot hold yet) onto ``builder_cls``. ``_snapshot`` is the
    unchecked ``build``: ``simulate`` keeps the caller's Python side
    state beside it.

    ``scripts/generate_stubs.py`` sets ``PULSIM_STUBGEN_NATIVE``: the
    stub then shows the native signatures these wrappers forward to (a
    Parameter is ``SupportsFloat``), not ``(*args, **kwargs)``."""
    import os
    if os.environ.get("PULSIM_STUBGEN_NATIVE"):
        return
    native = builder_cls.build
    if not getattr(native, "_checks_side_state", False):
        builder_cls._snapshot = native
        builder_cls.build = _checked_build(native)
    for name in dir(builder_cls):
        fn = getattr(builder_cls, name)
        # shortcut: an overloaded binding (add_nonlinear_capacitor) keeps
        # its native form — one signature cannot stand for its overloads;
        # it refuses a Parameter (Parameter is not a float).
        if (name.startswith("add_") and callable(fn)
                and not getattr(fn, "_accepts_parameters", False)
                and "Overloaded function" not in (fn.__doc__ or "")):
            setattr(builder_cls, name, _accepting_parameters(fn, name))
    for name, field in _DEPRECATED_UPDATES.items():
        fn = getattr(builder_cls, name, None)
        if fn is not None and not getattr(fn, "_deprecated", False):
            setattr(builder_cls, name, _deprecated_update(fn, name, field))


__all__ = ["RunPlan", "Tie", "install"]
