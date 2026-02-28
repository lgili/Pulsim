"""Ensure the locally built pulsim extension is used for all tests."""

import importlib
import glob
import os
import sys


def _build_tree_has_extension(build_path: str) -> bool:
    package_dir = os.path.join(build_path, "pulsim")
    if not os.path.isdir(package_dir):
        return False
    for pattern in ("_pulsim*.so", "_pulsim*.pyd", "_pulsim*.dylib"):
        if glob.glob(os.path.join(package_dir, pattern)):
            return True
    return False


def _clear_cached_pulsim_modules() -> None:
    for name in list(sys.modules):
        if name == "pulsim" or name.startswith("pulsim."):
            del sys.modules[name]


def _remove_path(path: str) -> None:
    abs_path = os.path.abspath(path)
    sys.path = [p for p in sys.path if os.path.abspath(p) != abs_path]


def _ensure_pulsim_path() -> tuple[bool, str, str]:
    build_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
    )
    source_python_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    # Prefer local build tree only when native extension is actually present.
    if _build_tree_has_extension(build_path):
        _remove_path(build_path)
        _remove_path(source_python_path)
        sys.path.insert(0, build_path)
        return True, build_path, source_python_path
    _remove_path(build_path)
    # When tests are executed from python/tests, pytest can prepend
    # <repo>/python, which shadows the installed wheel package.
    _remove_path(source_python_path)
    return False, build_path, source_python_path


def _ensure_benchmarks_path() -> None:
    benchmarks_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks")
    )
    if os.path.exists(benchmarks_path):
        sys.path = [p for p in sys.path if os.path.abspath(p) != benchmarks_path]
        sys.path.insert(0, benchmarks_path)


def _reload_pulsim_if_needed(
    prefer_build_tree: bool,
    build_path: str,
    source_python_path: str,
) -> None:
    if "pulsim" in sys.modules:
        mod = sys.modules["pulsim"]
        mod_file = os.path.abspath(getattr(mod, "__file__", ""))
        is_build_mod = "build/python" in mod_file
        source_pkg_path = os.path.join(source_python_path, "pulsim")
        is_source_mod = mod_file.startswith(source_pkg_path + os.sep)
        if (prefer_build_tree and not is_build_mod) or (
            not prefer_build_tree and (is_build_mod or is_source_mod)
        ):
            _clear_cached_pulsim_modules()
    importlib.invalidate_caches()
    try:
        importlib.import_module("pulsim")
    except ModuleNotFoundError as exc:
        # If local source/build trees are stale, fall back to installed package.
        if exc.name == "pulsim._pulsim":
            _remove_path(build_path)
            _remove_path(source_python_path)
            _clear_cached_pulsim_modules()
            importlib.invalidate_caches()
            importlib.import_module("pulsim")
        else:
            raise


_prefer_build_tree, _build_path, _source_python_path = _ensure_pulsim_path()
_ensure_benchmarks_path()
_reload_pulsim_if_needed(_prefer_build_tree, _build_path, _source_python_path)
