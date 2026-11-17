"""Ensure the locally built pulsim extension is used for all tests."""

import glob
import hashlib
import importlib
import os
import sys

import pytest


def _normalize_path(path: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def _is_within(path: str, root: str) -> bool:
    path_norm = _normalize_path(path)
    root_norm = _normalize_path(root)
    return path_norm == root_norm or path_norm.startswith(root_norm + os.sep)


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
    target = _normalize_path(path)
    sys.path = [p for p in sys.path if _normalize_path(p) != target]


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
        _remove_path(benchmarks_path)
        sys.path.insert(0, benchmarks_path)


def _reload_pulsim_if_needed(
    prefer_build_tree: bool,
    build_path: str,
    source_python_path: str,
) -> None:
    if "pulsim" in sys.modules:
        mod = sys.modules["pulsim"]
        mod_file = os.path.abspath(getattr(mod, "__file__", ""))
        source_pkg_path = os.path.join(source_python_path, "pulsim")
        build_pkg_path = os.path.join(build_path, "pulsim")
        is_build_mod = _is_within(mod_file, build_pkg_path)
        is_source_mod = _is_within(mod_file, source_pkg_path)
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


# The pulsim under test must be this checkout: its source tree, or an
# installed build of it (CI installs a wheel into site-packages). A stale
# install (the developer Mac has pulsim 1.8.0 in site-packages) is refused.
_CHECKOUT_PKG = os.path.join(_source_python_path, "pulsim")
# Files a wheel copies verbatim; __init__.py carries __version__.
_FINGERPRINT = ("__init__.py", "design.py", "_pulsim.pyi")


def _digest(pkg: str) -> dict:
    out = {}
    for f in _FINGERPRINT:
        path = os.path.join(pkg, f)
        out[f] = (hashlib.sha256(open(path, "rb").read()).hexdigest()
                  if os.path.isfile(path) else None)
    return out


def pulsim_checkout_mismatch(mod) -> str:
    """'' when ``mod`` is this checkout's pulsim, else why it is not."""
    pkg = os.path.dirname(os.path.realpath(mod.__file__))
    if _is_within(pkg, os.path.realpath(_CHECKOUT_PKG)):
        return ""
    stale = [f for f, h in _digest(pkg).items() if h != _digest(_CHECKOUT_PKG)[f]]
    if not stale:
        return ""
    return (f"the imported pulsim is not this checkout: {mod.__file__} "
            f"(version {getattr(mod, '__version__', '?')}) differs from "
            f"{_CHECKOUT_PKG} in {stale}; build this tree or put "
            f"{_source_python_path} on PYTHONPATH")


@pytest.fixture(scope="session", autouse=True)
def _pulsim_is_this_checkout():
    msg = pulsim_checkout_mismatch(sys.modules["pulsim"])
    if msg:
        pytest.fail(msg, pytrace=False)
