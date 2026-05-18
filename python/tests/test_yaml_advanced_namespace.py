"""simplify-and-harden-numerical-surface — Phase 3.5 tests.

Verifies the YAML parser accepts `simulation.advanced.*` as a
canonical namespace for advanced numerical knobs. Both the flat
form (`simulation.newton.*`) and the namespaced form
(`simulation.advanced.newton.*`) target the same underlying fields.
"""

import pulsim as ps


def test_yaml_advanced_newton_max_iterations():
    yaml_in = """
schema: pulsim-v1
version: 1
simulation:
  preset: robust
  tstop: 1e-4
  dt: 1e-6
  advanced:
    newton:
      max_iterations: 777
components:
  - { type: resistor, name: R1, nodes: [a, 0], value: 1000.0 }
"""
    parser = ps.YamlParser()
    ckt, opts = parser.load_string(yaml_in)
    assert parser.errors == []
    assert opts.newton_options.max_iterations == 777


def test_yaml_advanced_integrator_override():
    yaml_in = """
schema: pulsim-v1
version: 1
simulation:
  preset: robust
  tstop: 1e-4
  dt: 1e-6
  advanced:
    integrator: bdf1
components:
  - { type: resistor, name: R1, nodes: [a, 0], value: 1000.0 }
"""
    parser = ps.YamlParser()
    ckt, opts = parser.load_string(yaml_in)
    assert parser.errors == []
    assert opts.integrator == ps.Integrator.BDF1
    # Rest of the Robust preset still applies.
    assert opts.stiffness_config.enable is True


def test_yaml_advanced_and_flat_both_work():
    """Setting newton.max_iterations via both flat AND advanced paths.
    The last one written wins — but both write to the same field."""
    yaml_in = """
schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-4
  dt: 1e-6
  newton:
    max_iterations: 100
  advanced:
    newton:
      max_iterations: 200
components:
  - { type: resistor, name: R1, nodes: [a, 0], value: 1000.0 }
"""
    parser = ps.YamlParser()
    ckt, opts = parser.load_string(yaml_in)
    assert parser.errors == []
    # The advanced.* path wins because it's processed later in the
    # parser's dispatch (advanced is the canonical path).
    assert opts.newton_options.max_iterations == 200
