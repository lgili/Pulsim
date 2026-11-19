"""``pulsim.device_fields``: a device kind's value fields in one shape.

add-unified-device-interface group 12. A built-in kind reads the C++ field
table (``ir::kKindFields`` through ``_pulsim.device_field_table``); an
external device model (``pulsim.xdev.Device``, or anything with its
``fields`` list of ``(name, default, unit, lo, hi)``) is mapped onto the
same keys. What differs: an external field has numeric closed bounds
``[lo, hi]`` and no description, a built-in field a bound class
("a finite number > 0") and a description.
"""

from __future__ import annotations

from typing import Any


def device_fields(kind: Any) -> list[dict]:
    """The value fields of device kind ``kind`` (its YAML / ``components()``
    kind name, e.g. ``"shockley_diode"``) or of an external device model.

    Each entry: ``name`` (the YAML / ``with_value`` name), ``key`` (the
    builder argument and ``components()`` key), ``unit``, ``bound``,
    ``default``, ``description``, ``listed`` (in ``components()``),
    ``derived`` (stored as another field, e.g. R as G = 1/R).
    """
    if not isinstance(kind, str):
        rows = getattr(kind, "fields", None)
        if rows is None:
            raise TypeError("device_fields: give a kind name or an external device model "
                            "(an object with a `fields` list)")
        return [{"name": n, "key": n, "unit": u, "bound": f"[{lo!r}, {hi!r}]",
                 "default": float(d), "description": "", "listed": True,
                 "derived": False} for (n, d, u, lo, hi) in rows]
    from ._pulsim import device_field_table
    for row in device_field_table():
        if row["kind"] == kind:
            return [dict(f) for f in row["fields"]]
    raise KeyError(f"device_fields: no device kind {kind!r}")
