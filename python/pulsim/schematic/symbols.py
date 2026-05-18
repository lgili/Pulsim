"""Mapping from canonical Pulsim device kinds to schemdraw elements.

Phase 3 of the ``add-schematic-rendering`` change. The mapping covers the
common power-electronics inventory; anything missing renders as a labeled
rectangle with a one-shot ``UserWarning`` so the schematic doesn't crash
on a new device kind.

The map values are *factories* (callables returning a fresh element each
time) rather than element instances, because schemdraw elements are
single-use — once added to a Drawing they cannot be reused.
"""

from __future__ import annotations

import warnings
from typing import Callable


def _build_symbol_factories() -> dict[str, Callable]:
    """Build the kind → element-factory map, importing schemdraw lazily.

    Called once at the top of :func:`get_symbol_factory` (which caches the
    result) so importing ``pulsim.schematic.symbols`` itself never imports
    schemdraw at module load time.
    """
    from schemdraw import elements as elm

    return {
        # Two-terminal passives
        "resistor": elm.Resistor,
        "inductor": elm.Inductor,
        "capacitor": elm.Capacitor,
        # Two-terminal nonlinear
        "diode": elm.Diode,
        "switch": elm.Switch,
        # Sources
        "voltage_source": elm.SourceV,
        "current_source": elm.SourceI,
        "sine_voltage_source": elm.SourceSin,
        "pulse_voltage_source": elm.SourcePulse,
        "pwm_voltage_source": elm.SourcePulse,
        # Three-terminal active devices. NFet covers MOSFET cleanly;
        # IGBT has no first-class schemdraw element in 0.21 so we reuse
        # NFet (visually close — both have gate/drain/source-ish anchors).
        # vcswitch (voltage-controlled switch) maps to NFet because that's
        # the closest schematic appearance.
        "mosfet": elm.NFet,
        "igbt": elm.NFet,
        "vcswitch": elm.NFet,
        # Four-terminal magnetic
        "transformer": elm.Transformer,
    }


_FACTORY_CACHE: dict[str, Callable] | None = None
_WARNED_KINDS: set[str] = set()


def _ensure_factories() -> dict[str, Callable]:
    global _FACTORY_CACHE
    if _FACTORY_CACHE is None:
        _FACTORY_CACHE = _build_symbol_factories()
    return _FACTORY_CACHE


def get_symbol_factory(kind: str) -> Callable | None:
    """Return a schemdraw element factory for the given canonical kind.

    Returns ``None`` when the kind isn't mapped — callers fall back to a
    generic labeled rectangle and emit a one-shot ``UserWarning`` per
    unknown kind (so a circuit with twenty unknown devices doesn't spam
    twenty warnings).
    """
    factories = _ensure_factories()
    factory = factories.get(kind)
    if factory is None and kind not in _WARNED_KINDS:
        _WARNED_KINDS.add(kind)
        warnings.warn(
            f"pulsim.schematic: no symbol mapping for kind {kind!r}; "
            "rendering as a labeled rectangle. Add a mapping in "
            "pulsim/schematic/symbols.py to fix.",
            UserWarning,
            stacklevel=3,
        )
    return factory
