"""YAML `simulation: integrator:` option (Phase 2.4 schema, v1.6 wiring).

Verifies that the adaptive-RK integrator selector round-trips through
`SimulationOptions` + YAML and that `simulate(integrator=...)` rejects
non-kernel choices with a clear `NotImplementedError` until the v1.6
cache refactor lands (DAE-aware continuous-time assembly).

The schema-side support keeps user YAMLs forward-compatible today.
"""
from __future__ import annotations

import numpy as np
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


# ---------------------------------------------------------------------------
# add-ai-usage-guide 12.15: simulate(LoadedCircuit) forwards every
# `simulation:` key, not only the run window.
# ---------------------------------------------------------------------------
_RC_TRBDF2_YAML = """
circuit:
  devices:
    - {type: voltage_source, name: V1, from: a, to: gnd, V: 1.0}
    - {type: resistor,       name: R1, from: a, to: b,   R: 1.0e3}
    - {type: capacitor,      name: C1, from: b, to: gnd, C: 1.0e-6}
simulation:
  t_end: 5.0e-3
  dt: 5.0e-4
  rtol: %s
  atol: 1.0e-10
"""


def test_yaml_simulation_keys_are_listed_in_file_order():
    loaded = pulsim.load_yaml_string(_RC_YAML)
    assert list(loaded.simulation_keys) == [
        "t_end", "dt", "integrator", "rtol", "atol", "dt_init"]


def test_simulate_loaded_circuit_forwards_the_integrator():
    """The YAML's `integrator: radau` reached no engine before: the run
    went through on the kernel integrator. On the engine that takes an
    integrator it is now the keyword's own refusal."""
    with pytest.raises(ValueError, match="integrator='radau'"):
        pulsim.simulate(pulsim.load_yaml_string(_RC_YAML), engine="exact")


def test_keys_the_landing_engine_cannot_honour_warn_and_run():
    """The same file on its dt (the fixed step): integrator, rtol, atol and
    dt_init mean nothing there. It ran before 12.15 and still runs, now
    with one RuntimeWarning naming the unused keys, byte-equal to the run
    without them."""
    loaded = pulsim.load_yaml_string(_RC_YAML)
    with pytest.warns(RuntimeWarning, match="integrator='radau'.*dt_init"):
        r = pulsim.simulate(loaded)
    ref = pulsim.simulate(loaded.builder, 1e-3, dt=1e-5)
    assert r.engine_used == "pwl"
    np.testing.assert_array_equal(np.asarray(r.states), np.asarray(ref.states))


_DOC_BLOCK_YAML = _RC_YAML.replace("""  integrator: radau
  rtol: 1.5e-6
  atol: 2.5e-9
  dt_init: 1.0e-7
""", """  enable_newton_line_search: false
  max_event_iterations: 16
  max_newton_iterations: 50
  tol_newton_dx: 1.0e-9
  integrator: kernel
  dt_init: 0.0
""")


@pytest.mark.parametrize("engine", ["auto", "pwl", "trbdf2", "exact"])
def test_the_documented_simulation_block_runs_on_every_engine(engine):
    """docs/yaml-chain.md's block: the loader's placeholders (`integrator:
    kernel`, `dt_init: 0`) and kernel-default Newton values ask nothing,
    so no engine refuses or warns."""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*YAML simulation")
        r = pulsim.simulate(pulsim.load_yaml_string(_DOC_BLOCK_YAML),
                            engine=engine)
    assert len(r.times) > 10


def test_simulate_loaded_circuit_forwards_tolerances():
    """rtol/atol from the YAML give the run of the same keywords, and
    the error against the RC closed form follows the YAML's rtol."""
    import numpy as np

    def err(r):
        t = np.asarray(r.times)
        return float(np.max(np.abs(r.v("b") - (1.0 - np.exp(-t / 1e-3)))))

    loose = pulsim.load_yaml_string(_RC_TRBDF2_YAML % "1.0e-2")
    r = pulsim.simulate(loose, engine="trbdf2")
    kw = pulsim.simulate(loose.builder, 5e-3, dt=5e-4, engine="trbdf2",
                         rtol=1e-2, atol=1e-10)
    np.testing.assert_array_equal(np.asarray(r.times), np.asarray(kw.times))
    np.testing.assert_array_equal(r.v("b"), kw.v("b"))
    tight = pulsim.simulate(
        pulsim.load_yaml_string(_RC_TRBDF2_YAML % "1.0e-7"), engine="trbdf2")
    assert len(tight.times) > 3 * len(r.times)
    assert err(tight) < 1e-5 < err(r)
    # a keyword still wins over the file
    k2 = pulsim.simulate(loose, engine="trbdf2", rtol=1e-7)
    np.testing.assert_array_equal(np.asarray(k2.times),
                                  np.asarray(tight.times))


def test_a_changed_options_field_without_a_yaml_key_is_refused():
    loaded = pulsim.load_yaml_string(_RC_TRBDF2_YAML % "1.0e-3")
    loaded.options.split_snap = 0.5
    with pytest.raises(pulsim.OptionsError, match="split_snap") as e:
        pulsim.simulate(loaded)
    assert e.value.option == "split_snap"
