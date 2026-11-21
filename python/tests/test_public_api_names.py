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


# The clash only happens when ``from .scope import LiveScope`` succeeds,
# i.e. when pyqtgraph imports. CI installs ``.[dev]``, which has no
# pyqtgraph, so a plain in-process test would pass against the broken
# ``__init__`` there. Run the cold import in a fresh interpreter with a
# stub ``pyqtgraph`` (just enough for ``pulsim/scope.py`` to import), so
# the regression is exercised on every machine; also run it against the
# real pyqtgraph when that is installed.
_PYQTGRAPH_STUB = """
import sys, types

class _AnyClass(types.ModuleType):
    # Every attribute is a fresh class: enough for module-level
    # ``class _ClickFilter(QtCore.QObject)`` in pulsim/scope.py.
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        cls = type(name, (), {})
        setattr(self, name, cls)
        return cls

_pg = types.ModuleType("pyqtgraph")
_qt = types.ModuleType("pyqtgraph.Qt")
for _n in ("QtWidgets", "QtCore", "QtGui"):
    setattr(_qt, _n, _AnyClass("pyqtgraph.Qt." + _n))
_pg.Qt = _qt
sys.modules["pyqtgraph"] = _pg
sys.modules["pyqtgraph.Qt"] = _qt
"""

_COLD_IMPORT_CHECK = """
import sys
import types
import pulsim as p
assert p._HAS_SCOPE, "pyqtgraph not found"
assert "pulsim.scope" not in sys.modules, "import pulsim loaded the GUI stack"
assert callable(p.scope), f"p.scope is {p.scope!r}"
assert not isinstance(p.scope, types.ModuleType), "p.scope is the submodule"
assert p.scope is p.plot.scope
import pulsim.scope
from pulsim.scope import LiveScope
# Re-importing the submodule must not re-bind the package attribute.
assert p.scope is p.plot.scope
assert LiveScope is p.LiveScope
assert sys.modules["pulsim.scope"].LiveScope is LiveScope
"""


@pytest.mark.parametrize("qt", [
    "stub",
    pytest.param("real", marks=pytest.mark.skipif(
        not _HAS_PYQTGRAPH, reason="real pyqtgraph not installed")),
])
def test_scope_callable_after_live_scope_import_in_fresh_interpreter(qt):
    code = (_PYQTGRAPH_STUB if qt == "stub" else "") + _COLD_IMPORT_CHECK
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr


# ---- lazy import of the scope GUI stack ----------------------------------

def _run(code):
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr


def test_import_pulsim_leaves_the_gui_stack_unloaded():
    _run("import sys, pulsim\n"
         "bad = [m for m in ('pyqtgraph', 'PySide6', 'pulsim.scope')\n"
         "       if m in sys.modules]\n"
         "assert not bad, bad\n")


@pytest.mark.parametrize("first", ["attribute", "from_import"])
def test_live_scope_resolves_on_demand(first):
    access = ("L = p.LiveScope" if first == "attribute"
              else "from pulsim.scope import LiveScope as L")
    _run(_PYQTGRAPH_STUB + "import sys, pulsim as p\n" + access + "\n"
         "assert 'pyqtgraph' in sys.modules and 'pulsim.scope' in sys.modules\n"
         "assert L is p.LiveScope is sys.modules['pulsim.scope'].LiveScope\n"
         "assert callable(p.scope) and p.scope is p.plot.scope\n")


def test_live_scope_without_pyqtgraph_names_the_dependency():
    _run("import sys\n"
         "sys.modules['pyqtgraph'] = None\n"
         "import pulsim as p\n"
         "assert not p._HAS_SCOPE and 'LiveScope' not in p.__all__\n"
         "assert not hasattr(p, 'LiveScope')\n"
         "try:\n"
         "    p.LiveScope\n"
         "except AttributeError as e:\n"
         "    assert 'pyqtgraph' in str(e), e\n"
         "assert callable(p.scope) and p.simulate\n")
