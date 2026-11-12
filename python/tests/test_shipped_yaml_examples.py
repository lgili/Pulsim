"""Every shipped ``examples/*.yaml`` loads, carries its run settings and runs.

Until 2.0.x, 9 of the 26 shipped examples indented ``simulation:`` under
``circuit:`` and 3 more had no ``simulation:`` block at all. The loader
ignored both, so the 12 files loaded with ``t_end = dt = 0`` and no
diagnostic; ``simulate(lc.builder, t_end=lc.options.t_end,
dt=lc.options.dt)`` then died on an engine message about the step
ceiling. This module pins the fixed files, the loader's key checks and
``simulate(LoadedCircuit)``.

A pip install does not ship ``examples/``; the file-driven tests skip
there. The key-check tests use inline YAML and always run.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pytest

import pulsim as p

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
YAMLS = sorted(EXAMPLES.glob("*.yaml")) if EXAMPLES.is_dir() else []

# Kept on the v1 schema on purpose (single-phase induction motor is a
# documented v2 gap); the loader must refuse it by name.
V1_ONLY = {"refrigerator_compressor_cc.yaml"}

# Large ladders: loading/stamping 2000 devices is the point of the file.
HEAVY = {"stress_large_rc_ladder.yaml", "stress_large_resistive_ladder.yaml"}

# Cap per run: at most this many fixed steps of the file's own dt.
MAX_STEPS = 200

needs_examples = pytest.mark.skipif(
    not YAMLS, reason="examples/ is not shipped with a pip install")


def _ids(paths):
    return [q.name for q in paths]


V2_YAMLS = [q for q in YAMLS if q.name not in V1_ONLY]


@needs_examples
def test_the_example_inventory_is_what_this_module_covers():
    # 26 files: 25 v2 netlists + 1 v1 reference. A new file is covered
    # automatically; this pins that none is silently skipped.
    assert len(YAMLS) == 26
    assert {q.name for q in YAMLS} >= V1_ONLY | HEAVY


@needs_examples
@pytest.mark.parametrize("path", V2_YAMLS, ids=_ids(V2_YAMLS))
def test_example_loads_with_its_run_settings(path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # no loader warning either
        lc = p.load_yaml_file(str(path))
    assert lc.source == str(path)
    assert lc.has_simulation, f"{path.name}: no top-level simulation: block"
    assert lc.t_end > lc.t_start >= 0.0
    assert lc.dt > 0.0
    assert lc.t_end == lc.options.t_end and lc.dt == lc.options.dt
    assert lc.warnings == []


@needs_examples
@pytest.mark.parametrize("path", V2_YAMLS, ids=_ids(V2_YAMLS))
def test_example_runs_a_short_fixed_step_window(path):
    lc = p.load_yaml_file(str(path))
    t_end = min(lc.t_end, lc.t_start + MAX_STEPS * lc.dt)
    # Gate drive is not in the YAML. All gates OFF: a defined state
    # with no shoot-through (the default, all closed, shorts the legs).
    n_sw = lc.builder.graph.num_switches
    off = p.SwitchStateMask(n_sw)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = p.simulate(lc, t_end=t_end, engine="pwl",
                         switch_fn=lambda t: off)
    elapsed = time.perf_counter() - t0
    times = np.asarray(res.times)
    assert res.engine_used == "pwl"
    assert times[-1] == pytest.approx(t_end, rel=1e-9, abs=lc.dt * 1e-6)
    # dt came from the file: a uniform grid of exactly that step.
    assert np.allclose(np.diff(times), lc.dt, rtol=1e-6)
    for name in lc.node_names:
        assert np.all(np.isfinite(res.v(name))), f"{path.name}: v({name})"
    budget = 60.0 if path.name in HEAVY else 20.0
    assert elapsed < budget, f"{path.name}: {elapsed:.1f} s"


@needs_examples
@pytest.mark.parametrize("name", sorted(V1_ONLY))
def test_v1_example_is_refused_by_name(name):
    with pytest.raises(RuntimeError, match="pulsim v1 netlist") as ei:
        p.load_yaml_file(str(EXAMPLES / name))
    assert name in str(ei.value)


# ---------------------------------------------------------------------------
# Loader key checks (inline YAML)
# ---------------------------------------------------------------------------

_DEVICES = """\
  devices:
    - {type: voltage_source, name: V1, from: a, to: gnd, V: 5.0}
    - {type: resistor,       name: R1, from: a, to: b,   R: 1.0e3}
    - {type: capacitor,      name: C1, from: b, to: gnd, C: 1.0e-6}
"""


def test_simulation_nested_under_circuit_is_refused():
    text = "circuit:\n" + _DEVICES + "  simulation:\n    t_end: 1.0e-3\n    dt: 1.0e-6\n"
    with pytest.raises(RuntimeError, match="nested under 'circuit:'"):
        p.load_yaml_string(text)


def test_nested_simulation_in_a_file_names_the_path(tmp_path):
    f = tmp_path / "nested.yaml"
    f.write_text("circuit:\n" + _DEVICES + "  simulation:\n    t_end: 1.0e-3\n")
    with pytest.raises(RuntimeError, match="nested.yaml") as ei:
        p.load_yaml_file(str(f))
    assert "TOP-LEVEL" in str(ei.value)


@pytest.mark.parametrize("text, key", [
    ("circuit:\n" + _DEVICES + "simulaton:\n  t_end: 1.0e-3\n", "simulaton"),
    ("circuit:\n" + _DEVICES + "chain: []\n", "chain"),
    ("circuit:\n" + _DEVICES + "  device:\n    - {}\n", "device"),
])
def test_unknown_top_level_or_circuit_key_is_refused(text, key):
    with pytest.raises(RuntimeError, match=f"unknown .* key '{key}'"):
        p.load_yaml_string(text)


def test_empty_simulation_block_is_refused():
    with pytest.raises(RuntimeError, match="'simulation:' must be a map"):
        p.load_yaml_string("circuit:\n" + _DEVICES + "simulation:\n")


def test_unknown_simulation_key_warns_and_is_recorded():
    text = ("circuit:\n" + _DEVICES +
            "simulation:\n  tstop: 1.0e-3\n  dt: 1.0e-6\n")
    with pytest.warns(UserWarning, match=r"simulation\.tstop .* 't_end'"):
        lc = p.load_yaml_string(text)
    assert len(lc.warnings) == 1
    assert lc.t_end == 0.0                     # tstop was not read
    with pytest.raises(ValueError, match="without a t_end"):
        p.simulate(lc)


def test_unknown_simulation_key_warning_can_be_an_error():
    text = "circuit:\n" + _DEVICES + "simulation:\n  t_end: 1.0e-3\n  foo: 1\n"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match=r"simulation\.foo"):
            p.load_yaml_string(text)


# ---------------------------------------------------------------------------
# simulate(LoadedCircuit) and the empty-window refusal
# ---------------------------------------------------------------------------

def test_simulate_takes_the_run_settings_from_a_loaded_circuit():
    lc = p.load_yaml_string(
        "circuit:\n" + _DEVICES + "simulation:\n  t_end: 2.0e-3\n  dt: 1.0e-5\n")
    res = p.simulate(lc)
    ref = p.simulate(lc.builder, t_end=2e-3, dt=1e-5)
    assert res.engine_used == ref.engine_used == "pwl"
    assert np.array_equal(np.asarray(res.times), np.asarray(ref.times))
    assert np.array_equal(res.v("b"), ref.v("b"))
    # A keyword wins over the file.
    assert np.asarray(p.simulate(lc, t_end=1e-3).times)[-1] == pytest.approx(1e-3)


def test_simulate_forwards_store_every_from_the_yaml():
    lc = p.load_yaml_string(
        "circuit:\n" + _DEVICES +
        "simulation:\n  t_end: 1.0e-3\n  dt: 1.0e-5\n  store_every: 10\n")
    assert len(p.simulate(lc).times) == 11


def test_simulate_without_a_simulation_block_names_the_yaml(tmp_path):
    f = tmp_path / "no_sim.yaml"
    f.write_text("circuit:\n" + _DEVICES)
    lc = p.load_yaml_file(str(f))
    assert not lc.has_simulation and lc.t_end == 0.0 and lc.dt == 0.0
    with pytest.raises(ValueError, match="no_sim.yaml") as ei:
        p.simulate(lc)
    assert "no top-level 'simulation:' block" in str(ei.value)
    # Supplying t_end runs it (no dt -> the variable-step engine).
    assert p.simulate(lc, t_end=1e-3).engine_used == "trbdf2"


def test_simulate_from_options_of_an_empty_yaml_says_why():
    lc = p.load_yaml_string("circuit:\n" + _DEVICES)
    with pytest.raises(ValueError, match="no top-level simulation: block"):
        p.simulate(lc.builder, t_end=lc.options.t_end, dt=lc.options.dt)


def test_simulate_builder_still_requires_t_end():
    lc = p.load_yaml_string("circuit:\n" + _DEVICES)
    with pytest.raises(TypeError, match="t_end"):
        p.simulate(lc.builder)


# ---------------------------------------------------------------------------
# LoadedCircuit read-only properties
# ---------------------------------------------------------------------------

_BUCK = """\
circuit:
  devices:
    - {type: voltage_source,         name: Vin,  from: vin, to: gnd, V: 24.0}
    - {type: mosfet_with_body_diode, name: Q1,   drain: vin, source: sw,
       R_on: 1.0e-3, R_off: 1.0e9, V_F: 0.7}
    - {type: diode,                  name: D_FW, anode: gnd, cathode: sw,
       g_on: 1.0e3, g_off: 1.0e-9, V_th: 0.7}
    - {type: inductor,               name: L1,   from: sw, to: vout, L: 100.0e-6}
    - {type: capacitor,              name: Cout, from: vout, to: gnd, C: 47.0e-6}
    - {type: resistor,               name: R_L,  from: vout, to: gnd, R: 5.0}
simulation:
  t_start: 0.0
  t_end: 5.0e-3
  dt: 1.0e-7
"""


def test_loaded_circuit_properties():
    lc = p.load_yaml_string(_BUCK)
    assert lc.source == "<string>"
    assert lc.has_simulation
    assert (lc.t_start, lc.t_end, lc.dt) == (0.0, 5e-3, 1e-7)
    assert lc.node_names == ["vin", "sw", "vout"]
    assert lc.switch_names == ["Q1", "Q1_body", "D_FW"]
    for k, name in enumerate(lc.switch_names):
        assert lc.builder.switch_index_of(name) == k
    assert len(lc.switch_names) == lc.builder.graph.num_switches
    assert lc.warnings == []
    assert "has_simulation=True" in repr(lc)
    for attr in ("source", "t_end", "dt", "switch_names", "node_names"):
        with pytest.raises(AttributeError):
            setattr(lc, attr, None)
