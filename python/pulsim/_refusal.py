"""The typed refusal, raised from Python (add-engine-conformance-matrix).

`UnsupportedError` itself is created by the extension so the kernel and
Python raise the same class. It is a `ValueError`; its `engine`,
`feature`, `device` and `kind` attributes say which engine refused
which device kind or keyword.
"""

from ._pulsim import UnsupportedError  # type: ignore[import-not-found]


def unsupported(engine: str, feature: str, message: str, *,
                device: str = "", kind: str = "",
                cls: type = UnsupportedError) -> UnsupportedError:
    """Build (not raise) an `UnsupportedError` with its four fields."""
    err = cls(message)
    err.engine, err.feature, err.device, err.kind = (
        engine, feature, device, kind)
    return err


class _UnsupportedNotImplemented(UnsupportedError, NotImplementedError):
    """Refusal at the two DSED extractor sites that raised
    `NotImplementedError` before; keeps `except NotImplementedError`
    working. Remove in the next major."""
