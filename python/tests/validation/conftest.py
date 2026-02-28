"""Pytest fixtures for validation framework."""

import pytest
import sys
import os
import glob


def _ensure_pulsim_path():
    """Put locally built extension at the front of sys.path before imports."""
    build_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "build", "python"
    )
    build_path = os.path.abspath(build_path)

    source_python_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

    package_dir = os.path.join(build_path, "pulsim")
    has_native_extension = any(
        glob.glob(os.path.join(package_dir, pattern))
        for pattern in ("_pulsim*.so", "_pulsim*.pyd", "_pulsim*.dylib")
    )

    # Remove any existing occurrence before deciding which package root to use.
    sys.path = [p for p in sys.path if os.path.abspath(p) != build_path]
    sys.path = [p for p in sys.path if os.path.abspath(p) != source_python_path]

    if has_native_extension:
        sys.path.insert(0, build_path)


# Make the compiled module discoverable during test collection
_ensure_pulsim_path()

# Preload pulsim from the build directory so later imports use the same module
import importlib  # noqa: E402
importlib.invalidate_caches()
importlib.import_module("pulsim")


@pytest.fixture(autouse=True)
def setup_pulsim_path():
    """Keep pulsim path present for any dynamic imports."""
    _ensure_pulsim_path()
    if os.environ.get("PULSIM_DEBUG_PATH"):
        import pulsim  # noqa: WPS433
        print(f"Using pulsim from: {getattr(pulsim, '__file__', 'unknown')}")
        print(f"sys.path[0:5]: {sys.path[:5]}")


@pytest.fixture
def tolerance_linear():
    """Default tolerance for linear circuits: 1%."""
    return 0.01


@pytest.fixture
def tolerance_dc():
    """Default tolerance for DC analysis: 0.01%."""
    return 0.0001


@pytest.fixture
def tolerance_nonlinear():
    """Default tolerance for nonlinear circuits: 5%."""
    return 0.05


@pytest.fixture
def ngspice_available():
    """Expose PySpice/ngspice availability to tests."""
    try:
        from .framework.spice_runner import is_pyspice_available
    except Exception:
        return False
    return is_pyspice_available()


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "level1: Linear circuits with analytical solutions"
    )
    config.addinivalue_line(
        "markers", "level2: DC analysis tests"
    )
    config.addinivalue_line(
        "markers", "level3: Nonlinear component tests"
    )
    config.addinivalue_line(
        "markers", "level4: Power converter tests"
    )
