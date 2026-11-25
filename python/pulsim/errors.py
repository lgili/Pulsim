"""The pulsim exception hierarchy (add-kernel-simulation-object, design D5).

Every error pulsim raises on purpose is a :class:`PulsimError`, and each
keeps the standard base its raise sites used before, so an existing
``except ValueError`` / ``except RuntimeError`` / ``except KeyError`` still
catches it:

=========================  ==================================================
class                      bases
=========================  ==================================================
``PulsimError``            ``Exception``
``NetlistError``           ``PulsimError``, ``ValueError`` (``.device``, ``.node``)
``OptionsError``           ``PulsimError``, ``ValueError`` (``.option``)
``UnsupportedError``       ``PulsimError``, ``ValueError`` (``.engine``, ``.feature``,
                           ``.device``, ``.kind``)
``ConvergenceError``       ``PulsimError``, ``RuntimeError``
``SimulationAborted``      ``ConvergenceError`` (``.partial``, ``.t_failed``)
``SingularError``          ``SimulationAborted`` (a singular matrix)
``Cancelled``              ``PulsimError``, ``RuntimeError``
``NameNotFoundError``      ``PulsimError``, ``KeyError``
``InternalError``          ``PulsimError``, ``RuntimeError`` (a pulsim bug)
=========================  ==================================================

The kernel's classes are created by the extension (its exception
translators name them at throw time); this module is their one Python
home, and ``pulsim.<Class>`` is the same object. ``NameNotFoundError`` keeps
``KeyError`` and does not take ``NetlistError``'s ``ValueError``: a name
lookup caught by an ``except ValueError`` fallback would be swallowed
(DECISIONS 28).
"""

from __future__ import annotations

from ._builder_ergonomics import Cancelled
from ._pulsim import (  # type: ignore[import-not-found]
    ConvergenceError,
    InternalError,
    NetlistError,
    NewtonRoundoffBudgetExceeded,
    OptionsError,
    PulsimError,
    SimulationAborted,
    SingularError,
    UnsupportedError,
)

__all__ = [
    "PulsimError", "NetlistError", "OptionsError", "UnsupportedError",
    "ConvergenceError", "SimulationAborted", "SingularError",
    "NewtonRoundoffBudgetExceeded", "Cancelled", "NameNotFoundError",
    "InternalError", "options_error", "netlist_error",
]


class NameNotFoundError(KeyError, PulsimError):
    """Raised by :meth:`SimulationResult.v` / `.i` / `.power` and the
    :class:`CircuitBuilder` lookup helpers when a requested name
    isn't registered. Carries ``name``, ``kind`` (one of
    ``"node"``, ``"branch"``, ``"switch"``), and up to three
    fuzzy-matched ``suggestions`` so the caller can hint at a
    likely typo.

    Subclasses :class:`KeyError` so existing
    ``try: result.v(name) except KeyError`` patterns keep
    working.
    """

    __slots__ = ("name", "kind", "suggestions")

    def __init__(
        self,
        name: str,
        kind: str,
        suggestions: "list[str] | None" = None,
    ) -> None:
        self.name = name
        self.kind = kind
        self.suggestions = list(suggestions or [])
        hint = (
            f" Did you mean {self.suggestions}?"
            if self.suggestions else ""
        )
        super().__init__(
            f"{kind} {name!r} is not registered.{hint}"
        )


NameNotFoundError.__module__ = "pulsim"   # its public name, as before


def options_error(option: str, message: str) -> OptionsError:
    """Build (not raise) an :class:`OptionsError` naming ``option``."""
    err = OptionsError(message)
    err.option = option
    return err


def netlist_error(message: str, *, device: str = "",
                  node: str = "") -> NetlistError:
    """Build (not raise) a :class:`NetlistError` naming the device / node."""
    err = NetlistError(message)
    err.device, err.node = device, node
    return err
