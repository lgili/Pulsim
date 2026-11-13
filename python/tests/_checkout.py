"""Is the imported pulsim this checkout (its source tree or a wheel built from it)?

On a developer machine a stale installed pulsim (1.8.0) is imported unless
this tree is on the path; CI installs a wheel built from the checkout into
site-packages. Both are accepted, a foreign or stale install is not.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

CHECKOUT_PKG = Path(__file__).resolve().parents[1] / "pulsim"
# Files whose bytes a wheel copies verbatim from the checkout.
FINGERPRINT = ("__init__.py", "design.py", "_pulsim.pyi")


def _digest(pkg: Path) -> dict[str, str | None]:
    return {f: hashlib.sha256((pkg / f).read_bytes()).hexdigest()
            if (pkg / f).is_file() else None for f in FINGERPRINT}


def assert_pulsim_is_this_checkout(mod) -> None:
    pkg = Path(mod.__file__).resolve().parent
    if pkg.is_relative_to(CHECKOUT_PKG.parent.parent):
        return
    installed, checkout = _digest(pkg), _digest(CHECKOUT_PKG)
    # __init__.py carries __version__, so its hash also pins the version.
    stale = [f for f in FINGERPRINT if installed[f] != checkout[f]]
    assert not stale, (
        f"the imported pulsim is not this checkout: {mod.__file__} "
        f"(version {getattr(mod, '__version__', '?')}) differs from "
        f"{CHECKOUT_PKG} in {stale}; build this tree or put "
        f"{CHECKOUT_PKG.parent} on PYTHONPATH")
