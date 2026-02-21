"""Ensure the locally built pulsim extension is used for all tests."""

import importlib
import os
import sys


def _ensure_pulsim_path() -> None:
    build_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
    )
    if os.path.exists(build_path):
        sys.path = [p for p in sys.path if os.path.abspath(p) != build_path]
        sys.path.insert(0, build_path)


def _ensure_benchmarks_path() -> None:
    benchmarks_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks")
    )
    if os.path.exists(benchmarks_path):
        sys.path = [p for p in sys.path if os.path.abspath(p) != benchmarks_path]
        sys.path.insert(0, benchmarks_path)


def _reload_pulsim_if_needed() -> None:
    if "pulsim" in sys.modules:
        mod = sys.modules["pulsim"]
        mod_file = os.path.abspath(getattr(mod, "__file__", ""))
        if mod_file and "build/python" not in mod_file:
            # Drop cached modules so import uses the build tree.
            for name in list(sys.modules):
                if name == "pulsim" or name.startswith("pulsim."):
                    del sys.modules[name]
    importlib.invalidate_caches()
    importlib.import_module("pulsim")


_ensure_pulsim_path()
_ensure_benchmarks_path()
_reload_pulsim_if_needed()
