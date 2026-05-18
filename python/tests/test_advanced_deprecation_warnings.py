"""simplify-and-harden-numerical-surface — Phase 3.4 tests.

Verifies the opt-in DeprecationWarning shim on the 9 flat
sub-block fields of `SimulationOptions`. Defaults to silent
(internal kernel + un-migrated user code stays quiet); flips on
via `pulsim.enable_advanced_deprecation_warnings()`.
"""

import warnings

import pulsim as ps


def test_default_is_silent():
    """Without enabling the shim, no warning fires on flat-field access."""
    opts = ps.SimulationOptions()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opts.newton_options.max_iterations = 100
        opts.timestep_config.dt_initial = 1e-6
        opts.lte_config.voltage_tolerance = 1e-4
    assert len(caught) == 0


def test_enable_emits_deprecation_warning_per_field():
    """Once enabled, the 9 fields emit DeprecationWarning on read/write."""
    ps.enable_advanced_deprecation_warnings()

    # Reset the per-attribute "warned once" set so this test is stable.
    ps._advanced_deprecation_warned.clear()

    opts = ps.SimulationOptions()

    fields = [
        "newton_options",
        "timestep_config",
        "lte_config",
        "bdf_config",
        "dc_config",
        "stiffness_config",
        "fallback_policy",
        "formulation_mode",
        "linear_solver",
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for name in fields:
            # Trigger one read on each — that's enough to fire the warning.
            _ = getattr(opts, name)

    warned_fields = {
        name for name in fields
        if any(name in str(w.message) for w in caught)
    }
    # All 9 fields should have warned at least once.
    assert warned_fields == set(fields), f"missing warnings for {set(fields) - warned_fields}"
    # Each warning is a DeprecationWarning.
    for w in caught:
        assert issubclass(w.category, DeprecationWarning)


def test_warning_fires_once_per_attribute():
    """After the first read/write, further accesses on the same attribute stay silent."""
    ps.enable_advanced_deprecation_warnings()
    ps._advanced_deprecation_warned.clear()

    opts = ps.SimulationOptions()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # 4 accesses, but should only emit 1 warning.
        _ = opts.newton_options
        opts.newton_options.max_iterations = 100
        _ = opts.newton_options
        opts.newton_options.max_iterations = 200

    newton_warnings = [w for w in caught if "newton_options" in str(w.message)]
    assert len(newton_warnings) == 1


def test_advanced_is_canonical_path_silent():
    """`opts.advanced().newton` does NOT emit a warning."""
    ps.enable_advanced_deprecation_warnings()
    ps._advanced_deprecation_warned.clear()

    opts = ps.SimulationOptions()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opts.advanced().newton.max_iterations = 100
        _ = opts.advanced().timestep.dt_initial
    # `opts.advanced()` itself doesn't fire warnings.
    # Note: the inner access to `newton_options` from the C++ `advanced()`
    # view goes through the C++ field directly, NOT the Python property,
    # so no Python-level warning fires.
    advanced_warnings = [w for w in caught
                         if issubclass(w.category, DeprecationWarning)]
    assert len(advanced_warnings) == 0


def test_enable_is_idempotent():
    """Calling enable() multiple times is a no-op."""
    ps.enable_advanced_deprecation_warnings()
    ps.enable_advanced_deprecation_warnings()  # second call: no exception
    ps.enable_advanced_deprecation_warnings()  # third call: still fine
    # No assertion needed — the test is "doesn't throw".
