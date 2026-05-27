"""YAML `simulation: integrator:` option (Phase 2.4 schema, v1.6 wiring).

Verifies that the adaptive-RK integrator selector round-trips through
`SimulationOptions` + YAML and that `simulate(integrator=...)` rejects
non-kernel choices with a clear `NotImplementedError` until the v1.6
cache refactor lands (DAE-aware continuous-time assembly).

The schema-side support keeps user YAMLs forward-compatible today.
"""
from __future__ import annotations

import pytest

pulsim = pytest.importorskip("pulsim")


_RC_YAML = """
circuit:
  devices:
    - {type: voltage_source, name: V1, from: a, to: gnd, V: 5.0}
    - {type: resistor,       name: R1, from: a, to: b,   R: 1.0}
    - {type: capacitor,      name: C1, from: b, to: gnd, C: 1.0e-6}
simulation:
  t_end: 1.0e-3
  dt: 1.0e-5
  integrator: radau
  rtol: 1.5e-6
  atol: 2.5e-9
  dt_init: 1.0e-7
"""


def test_simulation_options_has_integrator_defaults():
    opts = pulsim.SimulationOptions(t_start=0.0, t_end=1e-3, dt=1e-5)
    assert opts.integrator == "kernel"
    assert opts.rtol == pytest.approx(1e-5)
    assert opts.atol == pytest.approx(1e-8)
    assert opts.dt_init == pytest.approx(0.0)


def test_simulation_options_integrator_round_trip():
    opts = pulsim.SimulationOptions(t_start=0.0, t_end=1e-3, dt=1e-5)
    opts.integrator = "dopri5"
    opts.rtol = 1e-7
    opts.atol = 1e-10
    opts.dt_init = 2e-8
    assert opts.integrator == "dopri5"
    assert opts.rtol == pytest.approx(1e-7)
    assert opts.atol == pytest.approx(1e-10)
    assert opts.dt_init == pytest.approx(2e-8)


def test_yaml_loads_integrator_block():
    """A `simulation:` block in YAML populates the integrator
    selector on the returned `LoadedCircuit.options`."""
    loaded = pulsim.load_yaml_string(_RC_YAML)
    opts = loaded.options
    assert opts.integrator == "radau"
    assert opts.rtol == pytest.approx(1.5e-6)
    assert opts.atol == pytest.approx(2.5e-9)
    assert opts.dt_init == pytest.approx(1e-7)


def test_yaml_integrator_defaults_when_omitted():
    """YAMLs without explicit RK fields stay on the default
    `kernel` integrator + reference tolerances."""
    yaml_no_rk = """
circuit:
  devices:
    - {type: voltage_source, name: V1, from: a, to: gnd, V: 1.0}
    - {type: resistor,       name: R1, from: a, to: gnd, R: 1.0}
simulation:
  t_end: 1.0e-3
  dt: 1.0e-5
"""
    loaded = pulsim.load_yaml_string(yaml_no_rk)
    opts = loaded.options
    assert opts.integrator == "kernel"
    assert opts.rtol == pytest.approx(1e-5)
    assert opts.atol == pytest.approx(1e-8)
    assert opts.dt_init == pytest.approx(0.0)


def test_simulate_kernel_path_unchanged():
    """`integrator='kernel'` (and the default) runs the existing
    trap-companion kernel path without surprises."""
    b = pulsim.CircuitBuilder()
    b.add_voltage_source("V1", "a", "gnd", 5.0)
    b.add_resistor("R1", "a", "gnd", 1.0)

    # Default — no integrator kwarg.
    res = pulsim.simulate(b, t_end=1e-4, dt=1e-5)
    assert res.num_steps() > 0

    # Explicit kernel.
    res = pulsim.simulate(b, t_end=1e-4, dt=1e-5, integrator="kernel")
    assert res.num_steps() > 0


@pytest.mark.parametrize("name", ["dopri5", "radau"])
def test_simulate_rk_raises_not_implemented(name: str):
    """RK integrators are reserved for v1.6 — must raise
    `NotImplementedError` with a clear pointer until then."""
    b = pulsim.CircuitBuilder()
    b.add_voltage_source("V1", "a", "gnd", 5.0)
    b.add_resistor("R1", "a", "gnd", 1.0)

    with pytest.raises(NotImplementedError, match="v1.6"):
        pulsim.simulate(b, t_end=1e-4, dt=1e-5, integrator=name)


def test_simulate_unknown_integrator_raises_value_error():
    b = pulsim.CircuitBuilder()
    b.add_voltage_source("V1", "a", "gnd", 5.0)
    b.add_resistor("R1", "a", "gnd", 1.0)

    with pytest.raises(ValueError, match="unknown"):
        pulsim.simulate(b, t_end=1e-4, dt=1e-5, integrator="bogus")


def test_yaml_integrator_flow_into_simulate():
    """End-to-end: read an integrator selection from YAML and
    forward it to `simulate()`. The v1.6 path is gated, so
    `radau` must surface the same `NotImplementedError` even when
    the option comes from a YAML file."""
    loaded = pulsim.load_yaml_string(_RC_YAML)
    o = loaded.options
    # YAML chose `radau` — forwarded as a kwarg, must raise.
    with pytest.raises(NotImplementedError, match="v1.6"):
        pulsim.simulate(
            loaded.builder,
            t_end=o.t_end,
            dt=o.dt,
            integrator=o.integrator,
            rtol=o.rtol,
            atol=o.atol,
            dt_init=o.dt_init,
        )
