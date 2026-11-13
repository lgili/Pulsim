"""add-immutable-circuit-ir — the Python side of :class:`pulsim.Circuit`.

``simulate()`` runs every circuit on its own copy and reports what the
run did to it as a :class:`RunPlan` (``result.plan``): the circuit that
ran (``result.circuit``) and the regularising ties the preflight added.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple


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


__all__ = ["RunPlan", "Tie"]
