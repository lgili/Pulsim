"""simplify-and-harden-numerical-surface — Phase 2 + 14 Python YAML test.

End-to-end smoke covering the documented user pattern: pick a preset
in YAML, override specific fields, run the simulation. This is the
canonical recipe published in `docs/numerical-configuration.md` and
`docs/getting-started.md`.
"""

import pulsim as ps


def test_yaml_preset_robust_with_integrator_override():
    """Robust preset materializes TRBDF2; explicit override → BDF1."""
    yaml_in = """
schema: pulsim-v1
version: 1
simulation:
  preset: robust
  integrator: bdf1
  tstop: 1e-4
  dt: 1e-6
components:
  - { type: resistor, name: R1, nodes: [a, 0], value: 1000.0 }
  - { type: capacitor, name: C1, nodes: [a, 0], value: 1e-6, ic: 0.0 }
  - { type: voltage_source, name: V1, nodes: [a, 0], value: 5.0 }
"""
    parser = ps.YamlParser()
    ckt, opts = parser.load_string(yaml_in)
    assert parser.errors == []

    # Override wins over Robust's TRBDF2.
    assert opts.integrator == ps.Integrator.BDF1
    # Rest of the Robust profile still applies.
    assert opts.stiffness_config.enable is True
    assert opts.max_step_retries == 12
    assert opts.enable_bdf_order_control is True


def test_yaml_preset_fast_for_pure_switching():
    yaml_in = """
schema: pulsim-v1
version: 1
simulation:
  preset: fast
  tstop: 1e-3
  dt: 1e-6
components:
  - { type: resistor, name: R1, nodes: [a, 0], value: 1000.0 }
"""
    parser = ps.YamlParser()
    ckt, opts = parser.load_string(yaml_in)
    assert parser.errors == []
    assert opts.integrator == ps.Integrator.Trapezoidal
    assert opts.switching_mode == ps.SwitchingMode.Ideal
    assert opts.step_mode == ps.StepMode.Fixed
    assert opts.stiffness_config.enable is False


def test_yaml_deprecated_adaptive_timestep_emits_warning():
    yaml_in = """
schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-3
  dt: 1e-6
  adaptive_timestep: true
components:
  - { type: resistor, name: R1, nodes: [a, 0], value: 1000.0 }
"""
    parser = ps.YamlParser()
    ckt, opts = parser.load_string(yaml_in)
    assert parser.errors == []
    # Should have emitted at least one deprecation warning mentioning
    # adaptive_timestep + the canonical step_mode replacement.
    assert any("adaptive_timestep" in w.lower() for w in parser.warnings)


def test_yaml_deprecated_integrator_bdf5_emits_warning():
    yaml_in = """
schema: pulsim-v1
version: 1
simulation:
  preset: robust
  integrator: bdf5
  tstop: 1e-3
  dt: 1e-6
components:
  - { type: resistor, name: R1, nodes: [a, 0], value: 1000.0 }
"""
    parser = ps.YamlParser()
    ckt, opts = parser.load_string(yaml_in)
    assert parser.errors == []
    # Should warn about bdf5 being deprecated.
    bdf5_warnings = [w for w in parser.warnings if "bdf5" in w.lower()]
    assert len(bdf5_warnings) >= 1
