"""Pytest configuration and fixtures for Pulsim tests."""

import pytest
import sys
import os
import glob

# Add the build directory to path if running tests before installation
# This allows testing the compiled module directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
build_patterns = [
    os.path.join(project_root, 'build', 'cp*', 'python'),  # scikit-build-core location
    os.path.join(project_root, 'build', 'python'),
]

path_added = False
for pattern in build_patterns:
    for path in glob.glob(pattern):
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
            path_added = True
            break
    if path_added:
        break


@pytest.fixture
def default_tolerances():
    """Default solver tolerances."""
    import pulsim as ps
    return ps.Tolerances.defaults()


@pytest.fixture
def newton_options():
    """Default Newton solver options."""
    import pulsim as ps
    opts = ps.NewtonOptions()
    opts.max_iterations = 50
    opts.auto_damping = True
    return opts


@pytest.fixture
def timestep_config():
    """Default timestep configuration."""
    import pulsim as ps
    return ps.TimestepConfig.defaults()


@pytest.fixture
def rc_analytical():
    """RC circuit analytical solution (1k, 1uF, 0V to 5V step)."""
    import pulsim as ps
    return ps.RCAnalytical(1000.0, 1e-6, 0.0, 5.0)


@pytest.fixture
def rl_analytical():
    """RL circuit analytical solution (1k, 1mH, 10V source)."""
    import pulsim as ps
    return ps.RLAnalytical(1000.0, 1e-3, 10.0, 0.0)


@pytest.fixture
def rlc_underdamped():
    """Underdamped RLC circuit analytical solution."""
    import pulsim as ps
    return ps.RLCAnalytical(10.0, 1e-3, 1e-6, 10.0, 0.0, 0.0)
