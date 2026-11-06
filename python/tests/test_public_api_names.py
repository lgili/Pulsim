"""Public names must be the objects the docs say they are.

Regression: ``pulsim/__init__.py`` imported the ``scope`` plotting
function from ``.plot`` and then ran ``from .scope import LiveScope``.
Importing a submodule binds it to the package attribute of the same
name, so ``pulsim.scope`` silently became the ``pulsim.scope`` *module*
and every documented ``p.scope(builder, result, ...)`` call raised
``TypeError: 'module' object is not callable``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types

import pytest

import pulsim as p

# Names in ``__all__`` that are deliberately submodules. Empty today:
# every exported name is a function, class or constant. Add a name
# here only if exporting a module is intended.
_INTENDED_MODULE_EXPORTS: frozenset[str] = frozenset()

_HAS_PYQTGRAPH = importlib.util.find_spec("pyqtgraph") is not None


def test_scope_is_the_plot_function():
    assert callable(p.scope)
    assert not isinstance(p.scope, types.ModuleType)
    assert p.scope is p.plot.scope


def test_no_exported_name_is_an_unintended_module():
    modules = sorted(
        name for name in p.__all__
        if isinstance(getattr(p, name, None), types.ModuleType)
        and name not in _INTENDED_MODULE_EXPORTS
    )
    assert modules == [], (
        f"these public names resolve to submodules, not the documented "
        f"objects: {modules}"
    )


@pytest.mark.skipif(not _HAS_PYQTGRAPH, reason="LiveScope needs pyqtgraph")
def test_live_scope_submodule_still_importable():
    import pulsim.scope  # noqa: F401
    from pulsim.scope import LiveScope

    # Re-importing the submodule must not re-bind the package attribute.
    assert p.scope is p.plot.scope
    assert LiveScope is p.LiveScope
    assert sys.modules["pulsim.scope"].LiveScope is LiveScope


@pytest.mark.skipif(not _HAS_PYQTGRAPH, reason="LiveScope needs pyqtgraph")
def test_scope_callable_in_a_fresh_interpreter_after_submodule_import():
    # The in-process tests above run after pulsim is already cached;
    # check the cold-import order a user script actually sees.
    code = (
        "import pulsim as p\n"
        "import pulsim.scope\n"
        "from pulsim.scope import LiveScope\n"
        "assert callable(p.scope) and p.scope is p.plot.scope\n"
        "assert LiveScope is p.LiveScope\n"
    )
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
