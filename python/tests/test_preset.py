"""simplify-and-harden-numerical-surface — Phase 2 Python tests.

Covers OpenSpec task 2.9: same coverage as `test_preset.cpp` but
exercised through the Python bindings. Verifies that:

  - The `pulsim.Preset` enum is reachable and iterable.
  - `pulsim.SimulationOptions.from_preset(...)` materializes each
    profile's documented field values.
  - Explicit field overrides win over the preset's defaults.
  - The raw `pulsim.SimulationOptions()` constructor still works.
"""

import math

import pytest

import pulsim as ps


def test_preset_enum_has_four_values():
    """`pulsim.Preset.__members__` exposes exactly the four documented
    profiles. pybind11 enums are not directly iterable, but `__members__`
    is the canonical introspection path."""
    members = ps.Preset.__members__
    assert len(members) == 4
    assert set(members.keys()) == {"Auto", "Fast", "Robust", "HighFidelity"}
    # And the values are reachable via dot access too.
    assert ps.Preset.Auto in members.values()
    assert ps.Preset.Fast in members.values()
    assert ps.Preset.Robust in members.values()
    assert ps.Preset.HighFidelity in members.values()


def test_preset_fast_targets_pure_switching():
    opts = ps.SimulationOptions.from_preset(ps.Preset.Fast,
                                             dt=1e-6, tstop=1e-3)
    assert opts.tstop == pytest.approx(1e-3)
    assert opts.dt == pytest.approx(1e-6)
    assert opts.switching_mode == ps.SwitchingMode.Ideal
    assert opts.integrator == ps.Integrator.Trapezoidal
    assert opts.step_mode == ps.StepMode.Fixed
    assert opts.enable_bdf_order_control is False
    assert opts.stiffness_config.enable is False
    assert opts.max_step_retries == 2
    assert opts.dt_max == pytest.approx(1e-6)


def test_preset_robust_targets_mixed_domain():
    opts = ps.SimulationOptions.from_preset(ps.Preset.Robust,
                                             dt=1e-6, tstop=1e-3)
    assert opts.integrator == ps.Integrator.TRBDF2
    assert opts.step_mode == ps.StepMode.Variable
    assert opts.enable_bdf_order_control is True
    assert opts.bdf_config.min_order == 1
    assert opts.bdf_config.max_order == 2
    assert opts.stiffness_config.enable is True
    assert opts.stiffness_config.switch_integrator is True
    assert opts.stiffness_config.stiff_integrator == ps.Integrator.BDF1
    assert opts.max_step_retries == 12
    assert opts.fallback_policy.trace_retries is True
    assert opts.fallback_policy.enable_transient_gmin is True


def test_preset_auto_aliases_robust_today():
    auto = ps.SimulationOptions.from_preset(ps.Preset.Auto, 1e-6, 1e-3)
    robust = ps.SimulationOptions.from_preset(ps.Preset.Robust, 1e-6, 1e-3)
    assert auto.integrator == robust.integrator
    assert auto.step_mode == robust.step_mode
    assert auto.stiffness_config.enable == robust.stiffness_config.enable
    assert auto.max_step_retries == robust.max_step_retries


def test_preset_high_fidelity_tightens_tolerances():
    robust = ps.SimulationOptions.from_preset(ps.Preset.Robust, 1e-6, 1e-3)
    hi = ps.SimulationOptions.from_preset(ps.Preset.HighFidelity,
                                           dt=1e-6, tstop=1e-3)
    assert hi.integrator == ps.Integrator.TRBDF2
    # 10× tighter LTE tolerance
    assert hi.timestep_config.error_tolerance == pytest.approx(
        robust.timestep_config.error_tolerance / 10.0
    )
    assert hi.lte_config.voltage_tolerance == pytest.approx(
        robust.lte_config.voltage_tolerance / 10.0
    )
    # Stricter step ceiling + more retries
    assert hi.timestep_config.dt_max < robust.timestep_config.dt_max
    assert hi.max_step_retries > robust.max_step_retries


def test_explicit_override_wins_over_preset():
    """After `from_preset(Robust)`, setting `integrator = BDF1` wins."""
    opts = ps.SimulationOptions.from_preset(ps.Preset.Robust, 1e-6, 1e-3)
    opts.integrator = ps.Integrator.BDF1
    assert opts.integrator == ps.Integrator.BDF1
    # Rest of the Robust profile still applies.
    assert opts.stiffness_config.enable is True
    assert opts.max_step_retries == 12


def test_raw_simulation_options_unchanged():
    """The legacy raw `SimulationOptions()` ctor still works."""
    opts = ps.SimulationOptions()
    # Raw defaults differ from any preset on the structural fields.
    assert opts.integrator == ps.Integrator.Trapezoidal
    assert opts.step_mode == ps.StepMode.Variable
    assert opts.max_step_retries == 6
    assert opts.enable_bdf_order_control is False
